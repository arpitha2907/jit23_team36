import json
from types import SimpleNamespace

import pytest

from app.core.config import settings
from app.services import correction, llm_correction


def _word(text, line=1, label=None):
    return {
        "text": text,
        "block_num": 1,
        "par_num": 1,
        "line_num": line,
        "field_label": label,
    }


class FakeClient:
    def __init__(self, content=None, error=None):
        self.content = content
        self.error = error
        self.calls = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    def sent_payload(self):
        return json.loads(self.calls[0]["messages"][1]["content"])


@pytest.fixture()
def enabled(monkeypatch):
    monkeypatch.setattr(settings, "llm_suggestions_enabled", True)
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")


def _use_client(monkeypatch, client):
    monkeypatch.setattr(llm_correction, "_client", lambda: client)


def test_disabled_makes_no_call(monkeypatch):
    monkeypatch.setattr(settings, "llm_suggestions_enabled", False)
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    client = FakeClient('{"suggestions": {"0": ["issued"]}}')
    _use_client(monkeypatch, client)

    assert llm_correction.suggest_batch([_word("igsued")], [0]) == {}
    assert client.calls == []


def test_enabled_without_key_makes_no_call(monkeypatch):
    monkeypatch.setattr(settings, "llm_suggestions_enabled", True)
    monkeypatch.setattr(settings, "openai_api_key", "")
    client = FakeClient('{"suggestions": {"0": ["issued"]}}')
    _use_client(monkeypatch, client)

    assert llm_correction.suggest_batch([_word("igsued")], [0]) == {}
    assert client.calls == []


def test_parses_and_filters_suggestions(enabled, monkeypatch):
    words = [_word("igsued"), _word("by"), _word("Ramesh", line=2, label="Name")]
    client = FakeClient(
        json.dumps(
            {
                "suggestions": {
                    "0": ["issued", "Issued", "igsued", "", "i#sued", "issued", 5],
                    "1": ["not requested"],
                    "2": [],
                    "nope": ["x"],
                }
            }
        )
    )
    _use_client(monkeypatch, client)

    result = llm_correction.suggest_batch(words, [0, 2], k=3)

    assert result == {0: ["issued", "Issued"], 2: []}
    assert [t["id"] for t in client.sent_payload()["tokens"]] == [0, 2]


def test_digits_never_sent(enabled, monkeypatch):
    words = [
        _word("Issued"),
        _word("igsued"),
        _word("on"),
        _word("1O/02/2024."),
        _word("Aadhaar", line=2),
        _word("4521", line=2),
    ]
    client = FakeClient('{"suggestions": {}}')
    _use_client(monkeypatch, client)

    llm_correction.suggest_batch(words, [1, 3])

    tokens = client.sent_payload()["tokens"]
    assert [t["id"] for t in tokens] == [1]
    assert tokens[0]["token"] == "igsued"
    assert tokens[0]["line"] == "Issued igsued on #O/##/####."


def test_only_digit_tokens_skips_call(enabled, monkeypatch):
    client = FakeClient('{"suggestions": {}}')
    _use_client(monkeypatch, client)

    assert llm_correction.suggest_batch([_word("56OO01")], [0]) == {}
    assert client.calls == []


@pytest.mark.parametrize(
    "client",
    [FakeClient(error=RuntimeError("401 invalid key")), FakeClient("not json")],
)
def test_failures_fall_back_to_empty(enabled, monkeypatch, client):
    _use_client(monkeypatch, client)
    assert llm_correction.suggest_batch([_word("igsued")], [0]) == {}


def test_llm_candidates_rank_first():
    suggestions = correction.suggest_corrections(
        ["it", "was", "igsued"],
        2,
        contextual_candidates=["sent"],
        language_codes=["eng"],
        llm_candidates=["issued"],
    )
    assert suggestions[0] == "issued"
    assert "sent" in suggestions


def test_reasoning_effort_is_sent_only_when_set(enabled, monkeypatch):
    client = FakeClient('{"suggestions": {}}')
    _use_client(monkeypatch, client)

    monkeypatch.setattr(settings, "openai_reasoning_effort", "minimal")
    llm_correction.suggest_batch([_word("igsued")], [0])
    monkeypatch.setattr(settings, "openai_reasoning_effort", "")
    llm_correction.suggest_batch([_word("igsued")], [0])

    assert client.calls[0]["reasoning_effort"] == "minimal"
    assert "reasoning_effort" not in client.calls[1]


def test_gpt_left_alone_hides_local_candidates():
    assert correction.suggest_corrections(
        ["Name:", "Ramesh"],
        1,
        contextual_candidates=["rates"],
        language_codes=["eng"],
        llm_candidates=[],
    ) == []


def test_without_gpt_local_candidates_remain():
    suggestions = correction.suggest_corrections(
        ["it", "was", "igsued"],
        2,
        contextual_candidates=["sent"],
        language_codes=["eng"],
        llm_candidates=None,
    )
    assert "sent" in suggestions


def test_numeric_tokens_ignore_llm_candidates():
    assert correction.suggest_corrections(
        ["ID", "45O1"],
        1,
        contextual_candidates=[],
        llm_candidates=["4501x"],
    ) == ["4501"]
