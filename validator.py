"""Pure-Python document index, agent tools and validation rules.

Nothing in this file calls a language model, so every rule can be unit tested.
The AI agent (agent.py) calls these functions as tools, and the application
re-runs them independently on the final answer.
"""
import math
import re
from collections import Counter
from typing import Any

STATUSES = ("MET", "PARTIAL", "MISSING", "UNCLEAR")
ACTIONS = ("replace", "append", "none")
NO_SUPPORT = "no supporting text found"
MIN_FRAGMENT = 8          # shortest quote (characters) that counts as evidence
MAX_REQUIREMENTS = 40

STOPWORDS = {
    "the", "and", "for", "that", "with", "this", "from", "are", "was", "were", "has", "have", "had",
    "not", "but", "all", "any", "can", "may", "must", "shall", "should", "will", "would", "could",
    "its", "their", "them", "they", "you", "your", "our", "out", "into", "than", "then", "also",
    "each", "per", "via", "use", "used", "using", "able", "system", "document",
}


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
def normalize(text: str) -> str:
    """Lower-case, unify quotes and collapse whitespace so quotes can be compared."""
    t = (text or "").replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s+", " ", t).strip().lower()


def stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def tokenize(text: str) -> list[str]:
    return [stem(w) for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(w) > 2 and w not in STOPWORDS]


def split_requirements(text: str, max_items: int = MAX_REQUIREMENTS) -> tuple[list[str], int]:
    """One requirement per non-empty line. Headings and very short lines are skipped.

    Returns (requirements, number_skipped_because_of_the_limit).
    """
    items: list[str] = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*\u2022]+|\d+[.)])\s*", "", line).strip()
        if not line or line.endswith(":") or len(line.split()) < 3:
            continue
        items.append(line)
    return items[:max_items], max(0, len(items) - max_items)


def evidence_fragments(evidence: str) -> list[str]:
    """Pieces of the evidence text that must be found verbatim in the document."""
    ev = (evidence or "").strip().replace("\u201c", '"').replace("\u201d", '"')
    quoted = [q.strip() for q in re.findall(r'"([^"]+)"', ev) if len(q.strip()) >= MIN_FRAGMENT]
    if quoted:
        return quoted
    ev = re.sub(r"^(?:paragraph|para|section)\s*\d+\s*[:\-]\s*", "", ev, flags=re.I)
    parts = re.split(r"\.\.\.|\u2026|\n|\s\|\s", ev)
    cleaned = [p.strip(" .;:-\"'") for p in parts]
    return [p for p in cleaned if len(p) >= MIN_FRAGMENT]


# ---------------------------------------------------------------------------
# Document index (used by the tools)
# ---------------------------------------------------------------------------
class DocIndex:
    def __init__(self, text: str):
        self.paragraphs = [p.strip() for p in text.splitlines() if p.strip()]
        self.norm = [normalize(p) for p in self.paragraphs]
        self.tokens = [set(tokenize(p)) for p in self.paragraphs]
        df: Counter = Counter()
        for toks in self.tokens:
            df.update(toks)
        n = max(len(self.paragraphs), 1)
        self.idf = {w: math.log(1 + n / c) for w, c in df.items()}

    def search(self, query: str, k: int = 4) -> list[dict[str, Any]]:
        q = set(tokenize(query))
        scored = []
        for i, toks in enumerate(self.tokens):
            score = sum(self.idf.get(w, 0.0) for w in q & toks)
            if score > 0:
                scored.append((score, i))
        scored.sort(key=lambda s: (-s[0], s[1]))
        return [{"paragraph_id": i + 1, "text": self.paragraphs[i][:700]} for _, i in scored[:k]]

    def find_paragraph(self, text: str) -> int | None:
        """1-based id of the paragraph that equals `text` exactly (ignoring outer spaces)."""
        target = (text or "").strip()
        if not target:
            return None
        for i, p in enumerate(self.paragraphs):
            if p == target:
                return i + 1
        return None

    def check_quote(self, evidence: str) -> dict[str, Any]:
        fragments = evidence_fragments(evidence)
        missing, ids = [], []
        for frag in fragments:
            nf = normalize(frag)
            hits = [i + 1 for i, p in enumerate(self.norm) if nf in p]
            if hits:
                ids.extend(hits)
            else:
                missing.append(frag)
        return {"fragments": fragments, "missing": missing, "paragraph_ids": sorted(set(ids)),
                "found": bool(fragments) and not missing}


# ---------------------------------------------------------------------------
# Validation rules
# ---------------------------------------------------------------------------
def clean_finding(raw: dict[str, Any]) -> dict[str, str]:
    f = {k: str(raw.get(k, "") or "").strip() for k in
         ("requirement", "status", "evidence", "reason", "action", "original_text", "suggested_text")}
    f["status"] = f["status"].upper()
    f["action"] = f["action"].lower()
    return f


def validate_finding(finding: dict[str, Any], doc: DocIndex) -> list[tuple[str, str]]:
    """Return (code, message) pairs. An empty list means the finding passes."""
    f = clean_finding(finding)
    v: list[tuple[str, str]] = []
    status, action = f["status"], f["action"]

    if status not in STATUSES:
        v.append(("status", f"status must be one of {', '.join(STATUSES)}."))
    if action not in ACTIONS:
        v.append(("action", f"action must be one of {', '.join(ACTIONS)}."))
    if not f["reason"]:
        v.append(("field", "reason is empty; explain briefly why this status was chosen."))

    evidence = f["evidence"]
    if not evidence:
        v.append(("evidence", "evidence is empty; quote the document verbatim or write 'No supporting text found'."))
    elif NO_SUPPORT in normalize(evidence):
        if status in ("MET", "PARTIAL"):
            v.append(("evidence", f"status {status} needs a verbatim quote from the document, but the evidence says "
                                  "no supporting text was found. Search again, or use MISSING/UNCLEAR."))
    else:
        q = doc.check_quote(evidence)
        if not q["fragments"]:
            v.append(("evidence", "evidence has no quotable text. Put the exact words from the document in double quotes."))
        elif q["missing"]:
            shown = "; ".join(f'"{m[:80]}"' for m in q["missing"][:3])
            v.append(("evidence", f"these quotes were not found verbatim in the document: {shown}. "
                                  "Copy the exact words from a search result."))

    if status == "MET" and action != "none":
        v.append(("action", "a MET requirement needs no edit; use action 'none'."))
    if action == "replace":
        if not f["original_text"]:
            v.append(("edit", "action 'replace' needs original_text copied from one paragraph of the document."))
        elif doc.find_paragraph(f["original_text"]) is None:
            v.append(("edit", "original_text does not exactly match one complete paragraph of the document."))
        if not f["suggested_text"]:
            v.append(("edit", "action 'replace' needs suggested_text."))
    elif action == "append":
        if f["original_text"]:
            v.append(("edit", "action 'append' needs original_text to be an empty string."))
        if not f["suggested_text"]:
            v.append(("edit", "action 'append' needs suggested_text."))
    return v


def finalize_finding(finding: dict[str, Any], doc: DocIndex) -> dict[str, Any]:
    """Independent verification. The `verified` flag never depends on the model's own claim."""
    f: dict[str, Any] = clean_finding(finding)
    violations = validate_finding(f, doc)
    codes = {c for c, _ in violations}
    f["verified"] = not violations
    f["violations"] = [m for _, m in violations]
    if violations:
        if f["status"] not in STATUSES or codes & {"evidence", "field"}:
            f["status"] = "UNCLEAR"
        if f["action"] not in ACTIONS or codes & {"action", "edit"}:
            f["action"] = "none"
        if "evidence" in codes and f["evidence"] and not f["evidence"].startswith("[Unverified]"):
            f["evidence"] = "[Unverified] " + f["evidence"]
        f["reason"] = (f["reason"] + " " if f["reason"] else "") + "(Automatic check: " + " ".join(f["violations"]) + ")"
    return f


# ---------------------------------------------------------------------------
# Tools exposed to the model
# ---------------------------------------------------------------------------
def tool_search_document(doc: DocIndex, query: str) -> dict[str, Any]:
    results = doc.search(query)
    if not results:
        return {"results": [], "note": "No paragraph matched. Try different keywords."}
    return {"results": results}


def tool_verify_evidence(doc: DocIndex, quote: str) -> dict[str, Any]:
    q = doc.check_quote(quote)
    if q["found"]:
        return {"found": True, "paragraph_ids": q["paragraph_ids"]}
    if not q["fragments"]:
        return {"found": False, "note": "No quotable text. Use exact words from the document in double quotes."}
    return {"found": False, "not_found": q["missing"][:3], "note": "Quote not verbatim in the document."}


def tool_check_finding(doc: DocIndex, args: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    violations = validate_finding(args, doc)
    if not violations:
        return {"valid": True, "report": "VALIDATION PASSED: the finding meets all rules."}, True
    return {"valid": False, "violations": [m for _, m in violations]}, False
