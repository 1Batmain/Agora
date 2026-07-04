"""local_llm_client — client redéclaré vers un endpoint OpenAI-compatible local."""
import json

import httpx
import pytest

from pipeline.cluster import local_llm_client as llc
from pipeline.cluster import mistral_client


class _Resp:
    status_code = 200

    def __init__(self, content="pong"):
        self._content = content

    def json(self):
        return {"choices": [{"message": {"content": self._content}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3}}


def test_chat_posts_to_local_endpoint_without_key(monkeypatch):
    seen = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        seen.update(url=url, payload=json, headers=headers or {})
        return _Resp()

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    out = llc.chat([{"role": "user", "content": "ping"}])
    assert out == "pong"
    assert seen["url"] == llc.API_URL
    assert "Authorization" not in seen["headers"]  # vLLM local : pas de clé requise
    assert seen["payload"]["model"] == llc.MODEL


def test_chat_json_mode_and_usage_accounting(monkeypatch):
    monkeypatch.setattr(httpx, "post",
                        lambda url, json=None, headers=None, timeout=None: _Resp('{"a": 1}'))
    mistral_client.reset_usage()
    out = llc.chat([{"role": "user", "content": "x"}], json_mode=True)
    assert json.loads(out) == {"a": 1}
    # Les tokens locaux alimentent le MÊME accumulateur que l'API (coût du build).
    usage = mistral_client.get_usage()
    assert usage["calls"] == 1
    assert usage["prompt_tokens"] == 7


def test_chat_folds_system_when_asked(monkeypatch):
    seen = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        seen["messages"] = json["messages"]
        return _Resp()

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(llc, "FOLD_SYSTEM", True)
    llc.chat([{"role": "system", "content": "règles"},
              {"role": "user", "content": "question"}])
    assert seen["messages"] == [{"role": "user", "content": "règles\n\nquestion"}]


def test_chat_raises_mistral_error_on_http_error(monkeypatch):
    class _Err:
        status_code = 500
        text = "boom"

        def json(self):
            return {"error": "boom"}

    monkeypatch.setattr(httpx, "post",
                        lambda url, json=None, headers=None, timeout=None: _Err())
    with pytest.raises(mistral_client.MistralError):
        llc.chat([{"role": "user", "content": "x"}])


def test_env_exports_route_all_stages():
    exports = llc.env_exports()
    joined = "\n".join(exports)
    for var in ("AGORA_MISTRAL_URL", "MISTRAL_API_KEY", "AGORA_MISTRAL_MODEL",
                "AGORA_MISTRAL_SYNTH_MODEL", "AGORA_CLAIMS_API_MODEL",
                "AGORA_OPINION_MODEL", "AGORA_ENRICH_MODEL"):
        assert var in joined
