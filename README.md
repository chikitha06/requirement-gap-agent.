# Requirement Gap Agent

Streamlit app with a Grok/Groq tool-using agent that checks a document against a list of requirements.

- `app.py` - Streamlit interface and DOCX editing
- `agent.py` - Grok/Groq function-calling loop (one agent per requirement, run in parallel)
- `validator.py` - document index, tools (`search_document`, `verify_evidence`, `check_finding`) and validation rules
- `tests/` - unit tests (no API key needed)

Setup: `pip install -r requirements.txt`, copy `.env.example` to `.env`, add your key, then `streamlit run app.py`.
Requirements file format: one requirement per line (bullets and numbers are removed; headings and lines under 3 words are skipped).
