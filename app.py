import hashlib
import io
import os

import streamlit as st
from docx import Document
from dotenv import load_dotenv

from agent import analyze
from validator import STATUSES

load_dotenv()

MODEL_NAME = os.getenv("GROQ_MODEL") or os.getenv("GROK_MODEL", "openai/gpt-oss-120b")
API_KEY = (
    os.getenv("GROQ_API_KEY")
    or os.getenv("GROK_API_KEY")
    or os.getenv("grok-api")
    or os.getenv("groq-api")
    or os.getenv("XAI_API_KEY")
    or ""
).strip()

st.set_page_config(page_title="Requirement Gap Agent", page_icon="📄", layout="wide")
st.title("Requirement Gap Agent")
st.write(
    "Upload a requirements file (one requirement per line) and the document to check. An AI agent searches the "
    "document with Python tools, verifies its own evidence, and proposes edits. You decide which edits to apply."
)


def read_upload(uploaded_file) -> tuple[str, str, bytes]:
    """Return extracted text, extension, and original file bytes."""
    raw = uploaded_file.getvalue()
    name = uploaded_file.name.lower()
    if name.endswith(".docx"):
        doc = Document(io.BytesIO(raw))
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return text, "docx", raw
    if name.endswith(".txt"):
        return raw.decode("utf-8", errors="replace"), "txt", raw
    raise ValueError("Please upload a .txt or .docx file.")


def make_revised_docx(original_bytes, original_ext, document_text, findings, approved_indexes) -> bytes:
    if original_ext == "docx" and original_bytes:
        doc = Document(io.BytesIO(original_bytes))
    else:
        doc = Document()
        for line in document_text.splitlines():
            doc.add_paragraph(line)

    # Apply exact paragraph replacements only. If an exact match is not found, skip it safely.
    for index in sorted(approved_indexes, reverse=True):
        finding = findings[index]
        action = finding.get("action")
        suggestion = str(finding.get("suggested_text", "")).strip()
        if not suggestion:
            continue
        if action == "replace":
            original = str(finding.get("original_text", "")).strip()
            if not original:
                continue
            for paragraph in doc.paragraphs:
                if paragraph.text.strip() == original:
                    paragraph.text = suggestion
                    break
        elif action == "append":
            doc.add_paragraph(suggestion)

    output = io.BytesIO()
    doc.save(output)
    return output.getvalue()


with st.sidebar:
    st.header("Settings")
    available_models = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
    if MODEL_NAME not in available_models:
        available_models.insert(0, MODEL_NAME)
    MODEL_NAME = st.selectbox("Model", options=available_models, index=available_models.index(MODEL_NAME))

if not API_KEY:
    st.error("API key is missing. Add 'grok-api' or 'GROQ_API_KEY' to your .env file in the project folder, then restart the app.")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    requirements_file = st.file_uploader("1. Requirements file (.txt or .docx)", type=["txt", "docx"], key="requirements_file")
with col2:
    target_file = st.file_uploader("2. Document to check (.txt or .docx)", type=["txt", "docx"], key="target_file")

if st.button("Analyze document", type="primary", disabled=not (requirements_file and target_file)):
    try:
        requirements_text, _, _ = read_upload(requirements_file)
        document_text, document_ext, original_bytes = read_upload(target_file)
        if not requirements_text.strip() or not document_text.strip():
            st.error("Both files must contain text.")
        else:
            key = hashlib.sha256((MODEL_NAME + "\x00" + requirements_text + "\x00" + document_text).encode()).hexdigest()
            cache = st.session_state.setdefault("cache", {})
            if key in cache:
                result = cache[key]
            else:
                bar = st.progress(0.0, text="Agents are reviewing the requirements...")
                result = analyze(API_KEY, MODEL_NAME, requirements_text, document_text,
                                 on_progress=lambda done, total, status: bar.progress(done / total, text=f"Reviewed {done} of {total} requirements" + (f" | {status}" if status else "")))
                bar.empty()
                if not any(f.get("error") for f in result["findings"]):
                    cache[key] = result  # failed runs are not cached, so you can simply click Analyze again
            st.session_state["analysis_result"] = result
            st.session_state["document_text"] = document_text
            st.session_state["document_ext"] = document_ext
            st.session_state["original_bytes"] = original_bytes
            st.session_state["target_filename"] = target_file.name
            st.session_state.pop("revised_bytes", None)
            st.success(f"Analysis complete in {result['timing']['total_seconds']} s. Review the findings below.")
    except Exception as exc:
        st.error(f"Could not analyze the files: {exc}")

result = st.session_state.get("analysis_result")
if result:
    st.subheader("Analysis summary")
    st.write(result.get("summary", "No summary returned."))
    findings = result.get("findings", [])
    metric_cols = st.columns(5)
    for col, status in zip(metric_cols, STATUSES):
        col.metric(status.title(), sum(1 for f in findings if f.get("status") == status))
    metric_cols[4].metric("Verified", f"{sum(1 for f in findings if f.get('verified'))}/{len(findings)}")

    st.subheader("Review findings and approve edits")
    approved = set()
    for i, finding in enumerate(findings):
        status = finding.get("status", "UNCLEAR")
        badge = "✅" if finding.get("verified") else "⚠️"
        with st.expander(f"{badge} {i + 1}. [{status}] {finding.get('requirement', 'Requirement')}", expanded=(status != "MET")):
            st.caption("Verified by independent Python checks" if finding.get("verified")
                       else "Could not be fully verified automatically. Status and edit were made safe. Check manually.")
            st.markdown("**Evidence**")
            st.write(finding.get("evidence", "No evidence provided."))
            st.markdown("**Why this status?**")
            st.write(finding.get("reason", ""))
            action = finding.get("action", "none")
            if action == "replace":
                st.markdown("**Paragraph to replace**")
                st.code(finding.get("original_text", ""))
            st.markdown("**Suggested edit**")
            st.write(finding.get("suggested_text", "") or "No edit suggested.")
            if action in ("replace", "append") and finding.get("suggested_text", "").strip():
                default_on = status in ("PARTIAL", "MISSING") and finding.get("verified", False)
                if st.checkbox("Approve this edit", value=default_on, key=f"approve_{i}"):
                    approved.add(i)
            else:
                st.caption("No safe automatic edit proposed for this finding.")

    with st.expander("Agent trace (tool calls, attempts and verification)"):
        for i, finding in enumerate(findings):
            st.markdown(f"**{i + 1}. {finding.get('requirement', '')}**  "
                        f"({finding.get('model_calls', 0)} model calls, {finding.get('attempts', 0)} submission(s), {finding.get('seconds', 0)} s)")
            st.code("\n".join(finding.get("trace", [])) or "(no trace)", language=None)

    st.warning("Approved edits are applied to a new DOCX file. The original upload is not changed. Exact paragraph matches are required for replacements.")
    if st.button("Apply approved edits and create DOCX", type="primary", disabled=(len(approved) == 0)):
        try:
            st.session_state["revised_bytes"] = make_revised_docx(
                st.session_state.get("original_bytes"), st.session_state.get("document_ext", "txt"),
                st.session_state.get("document_text", ""), findings, approved)
        except Exception as exc:
            st.error(f"Could not create the revised document: {exc}")

    if st.session_state.get("revised_bytes"):
        base_name = os.path.splitext(st.session_state.get("target_filename", "document"))[0]
        st.success("Revised DOCX is ready. Check the downloaded file before using it.")
        st.download_button("Download revised document", data=st.session_state["revised_bytes"],
                           file_name=f"{base_name}_revised.docx",
                           mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

st.divider()
st.caption("Prototype note: review all AI findings before relying on them. Do not upload confidential documents or treat proposed metrics/placeholders as verified facts.")