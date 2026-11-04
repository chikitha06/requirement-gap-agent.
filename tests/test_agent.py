"""Tests the agent loop with a scripted fake client (no network, no API key)."""
import json
from types import SimpleNamespace

from agent import run_agent, make_configs
from validator import DocIndex

DOC = DocIndex("Users can register with an email address.\nUsers can log in with their registered credentials.")


def call(name, **args):
    return SimpleNamespace(id=f"call_{name}", type="function",
                           function=SimpleNamespace(name=name, arguments=json.dumps(args)))


class FakeClient:
    def __init__(self, scripted):
        self.scripted, self.calls = list(scripted), 0
        self.chat = SimpleNamespace(completions=self)

    def create(self, model, messages, **kwargs):
        self.calls += 1
        calls = self.scripted.pop(0) if self.scripted else []
        msg = SimpleNamespace(role="assistant", content=None, tool_calls=calls)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


GOOD = dict(requirement="Users must log in.", status="MET", evidence='"Users can log in with their registered credentials."',
            reason="Described.", action="none", original_text="", suggested_text="")


def test_agent_passes_on_first_attempt():
    client = FakeClient([[call("search_document", query="log in")], [call("check_finding", **GOOD)]])
    out = run_agent(client, "fake", "Users must log in.", DOC, make_configs())
    assert out["verified"] and out["status"] == "MET" and out["attempts"] == 1 and out["model_calls"] == 2
    assert any(t.startswith("[SEARCH]") for t in out["trace"])


def test_agent_revises_after_violation():
    bad = dict(GOOD, evidence='"Users log in by fingerprint."')
    client = FakeClient([[call("search_document", query="log in")], [call("check_finding", **bad)], [call("check_finding", **GOOD)]])
    out = run_agent(client, "fake", "Users must log in.", DOC, make_configs())
    assert out["verified"] and out["attempts"] == 2


def test_agent_gives_up_after_two_failed_submissions_and_downgrades():
    bad = dict(GOOD, evidence='"Users log in by fingerprint."')
    client = FakeClient([[call("check_finding", **bad)], [call("check_finding", **bad)]])
    out = run_agent(client, "fake", "Users must log in.", DOC, make_configs())
    assert out["verified"] is False and out["status"] == "UNCLEAR" and out["attempts"] == 2
    assert client.calls == 2


def test_agent_without_any_submission_returns_safe_default():
    client = FakeClient([[], []])
    out = run_agent(client, "fake", "Users must log in.", DOC, make_configs())
    assert out["verified"] is False and out["status"] == "UNCLEAR" and out["action"] == "none"
