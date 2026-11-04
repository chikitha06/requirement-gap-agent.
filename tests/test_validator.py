from validator import (DocIndex, evidence_fragments, finalize_finding, split_requirements, tool_check_finding,
                       tool_search_document, tool_verify_evidence, validate_finding)

DOC = "\n".join([
    "Users can register with an email address and a password.",
    "Users can log in with their registered credentials.",
    "The application validates all user input before processing.",
    "Reports are generated every Monday morning.",
])


def doc():
    return DocIndex(DOC)


def finding(**kw):
    base = dict(requirement="Users must be able to log in.", status="MET",
                evidence='"Users can log in with their registered credentials."', reason="Login is described.",
                action="none", original_text="", suggested_text="")
    base.update(kw)
    return base


def test_split_requirements_skips_headings_and_bullets():
    reqs, skipped = split_requirements("Requirements:\n1. Users must be able to register.\n- The system must validate input.\n\nShort")
    assert reqs == ["Users must be able to register.", "The system must validate input."]
    assert skipped == 0


def test_split_requirements_limit():
    text = "\n".join(f"The system must do thing number {i}." for i in range(50))
    reqs, skipped = split_requirements(text, max_items=40)
    assert len(reqs) == 40 and skipped == 10


def test_search_finds_relevant_paragraph_with_stemming():
    results = tool_search_document(doc(), "register users with passwords")["results"]
    assert results and results[0]["paragraph_id"] == 1


def test_search_no_match():
    out = tool_search_document(doc(), "blockchain")
    assert out["results"] == [] and "note" in out


def test_verify_evidence_found_and_not_found():
    d = doc()
    assert tool_verify_evidence(d, '"Users can log in with their registered credentials."')["found"] is True
    assert tool_verify_evidence(d, '"Users can reset their forgotten passwords by email."')["found"] is False


def test_evidence_fragments_extracts_quotes():
    assert evidence_fragments('Paragraph 2: "Users can log in" and "registered credentials"') == ["Users can log in", "registered credentials"]


def test_valid_finding_passes():
    assert validate_finding(finding(), doc()) == []


def test_fabricated_evidence_is_rejected():
    codes = [c for c, _ in validate_finding(finding(evidence='"Users receive a confirmation SMS after login."'), doc())]
    assert "evidence" in codes


def test_met_with_no_support_is_rejected():
    codes = [c for c, _ in validate_finding(finding(evidence="No supporting text found"), doc())]
    assert "evidence" in codes


def test_missing_with_no_support_is_valid():
    f = finding(status="MISSING", evidence="No supporting text found", action="append",
                suggested_text="Add a password recovery workflow [confirm details].")
    assert validate_finding(f, doc()) == []


def test_replace_requires_exact_paragraph():
    f = finding(status="PARTIAL", action="replace", original_text="Users can log in.", suggested_text="New text.")
    assert ("edit" in [c for c, _ in validate_finding(f, doc())])
    f["original_text"] = "Users can log in with their registered credentials."
    assert validate_finding(f, doc()) == []


def test_append_needs_empty_original_and_suggestion():
    f = finding(status="MISSING", evidence="No supporting text found", action="append", original_text="x", suggested_text="")
    messages = " ".join(m for _, m in validate_finding(f, doc()))
    assert "empty string" in messages and "suggested_text" in messages


def test_met_must_not_edit():
    assert "action" in [c for c, _ in validate_finding(finding(action="append", suggested_text="x"), doc())]


def test_invalid_status_and_action():
    codes = [c for c, _ in validate_finding(finding(status="GREAT", action="delete"), doc())]
    assert "status" in codes and "action" in codes


def test_check_finding_tool_report():
    ok_result, ok = tool_check_finding(doc(), finding())
    bad_result, bad_ok = tool_check_finding(doc(), finding(evidence='"made up quote that is not there"'))
    assert ok and ok_result["valid"] and not bad_ok and bad_result["violations"]


def test_finalize_downgrades_unverified_finding():
    out = finalize_finding(finding(status="MET", evidence='"Users can sign in with a fingerprint scanner."'), doc())
    assert out["verified"] is False
    assert out["status"] == "UNCLEAR"
    assert out["evidence"].startswith("[Unverified]")


def test_finalize_makes_bad_edit_safe():
    out = finalize_finding(finding(status="PARTIAL", action="replace", original_text="not in doc", suggested_text="New."), doc())
    assert out["verified"] is False and out["action"] == "none"
