# Requirement Gap Agent (student MVP)

A single-agent document reviewer. It compares a requirements file with a target document, returns evidence-based findings, suggests changes, and lets the user approve edits before creating a new DOCX file.

## 1. Requirements
- Python 3.10+
- A Gemini API key
- Internet access for the Gemini API

## 2. Setup on Windows
Open Command Prompt or the VS Code terminal in this folder:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Set your key for this terminal session:

```bash
set GEMINI_API_KEY=YOUR_API_KEY_HERE
```

Then run:

```bash
streamlit run app.py
```

The app opens in your browser. You can also paste your key into the sidebar. Do not commit API keys to GitHub.

## 3. Use it
1. Upload one file from `samples/requirements/`.
2. Upload the matching file with the same number from `samples/documents/`.
3. Click **Analyze document**.
4. Review the evidence and proposed edits. Approve only edits you agree with.
5. Click **Apply approved edits and create DOCX** and download the new document.

Both `.txt` and `.docx` inputs are supported. The sample pairs are numbered 01–10.

## 4. Important prototype limitations
- This MVP uses the Gemini API; analysis requires an internet connection and may incur API usage.
- It uses a single LLM call for the first version; the agent/tool loop, retrieval index, audit log and formal evaluation can be added in later iterations.
- Replacement edits require an exact paragraph match. If the paragraph cannot be matched exactly, the app skips the replacement instead of guessing.
- New paragraphs are appended to the end of the document. This is intentionally safer than guessing where they belong.
- The app does not change the original upload.
- Do not upload private or confidential documents while using this prototype.
