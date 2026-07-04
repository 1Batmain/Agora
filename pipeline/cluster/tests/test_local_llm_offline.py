"""local_llm_offline — LLM() vLLM in-process branché sur le seam mistral_client.chat."""
import pytest

from pipeline.cluster import local_llm_offline as llo
from pipeline.cluster import mistral_client


def _fake_complete(text="pong", usage=None, seen=None):
    def complete(messages, temperature, max_tokens, json_mode):
        if seen is not None:
            seen.append(dict(messages=messages, temperature=temperature,
                             max_tokens=max_tokens, json_mode=json_mode))
        return text, (usage or {"prompt_tokens": 5, "completion_tokens": 2})
    return complete


def test_offline_chat_contract_and_usage():
    seen = []
    chat = llo.make_offline_chat(_fake_complete(seen=seen), model_id="gemma-local")
    mistral_client.reset_usage()
    out = chat([{"role": "user", "content": "ping"}],
               model="ignoré-le-modèle-est-local", temperature=0.0,
               max_tokens=64, json_mode=True, timeout=30)
    assert out == "pong"
    assert seen[0]["json_mode"] is True
    assert seen[0]["max_tokens"] == 64
    usage = mistral_client.get_usage()
    assert usage["calls"] == 1 and usage["prompt_tokens"] == 5
    assert "gemma-local" in usage["by_model"]


def test_offline_chat_folds_system_when_asked():
    seen = []
    chat = llo.make_offline_chat(_fake_complete(seen=seen), fold_system=True)
    chat([{"role": "system", "content": "règles"}, {"role": "user", "content": "q"}])
    assert seen[0]["messages"] == [{"role": "user", "content": "règles\n\nq"}]


def test_offline_chat_wraps_errors_as_mistral_error():
    def broken(messages, temperature, max_tokens, json_mode):
        raise RuntimeError("CUDA boom")

    chat = llo.make_offline_chat(broken)
    with pytest.raises(mistral_client.MistralError):
        chat([{"role": "user", "content": "x"}])


def test_install_rebinds_the_seam(monkeypatch):
    # Enregistre les valeurs d'origine pour restauration automatique.
    monkeypatch.setattr(mistral_client, "chat", mistral_client.chat)
    monkeypatch.setattr(mistral_client, "available", mistral_client.available)
    monkeypatch.setattr(mistral_client, "load_api_key", mistral_client.load_api_key)

    chat = llo.make_offline_chat(_fake_complete())
    llo.install(chat)
    assert mistral_client.chat is chat
    assert mistral_client.available() is True
    assert mistral_client.load_api_key() == "local-offline"


def test_set_env_routes_every_stage(monkeypatch):
    for var in llo._MODEL_ENV_VARS + ("MISTRAL_API_KEY",):
        monkeypatch.delenv(var, raising=False)
    llo.set_env("google/gemma-4-12B-it")
    import os
    for var in llo._MODEL_ENV_VARS:
        assert os.environ[var] == "google/gemma-4-12B-it"
    assert os.environ["MISTRAL_API_KEY"] == "local-offline"
