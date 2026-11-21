from core import providers
from core.providers import embedding_mode, embedding_model_id, resolve_embedding_provider


def test_auto_prefers_local_model_when_cached(monkeypatch):
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    monkeypatch.setattr(providers, "local_model_available", lambda: True)
    assert resolve_embedding_provider() == "local"
    assert embedding_model_id() == "local:BAAI/bge-small-en-v1.5" and not embedding_mode()["mock"]


def test_auto_falls_back_to_clearly_labelled_mock(monkeypatch):
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    monkeypatch.setattr(providers, "local_model_available", lambda: False)
    mode = embedding_mode()
    assert mode["mock"] and "MOCK" in mode["label"] and embedding_model_id() == "mock:hash-256"


def test_explicit_provider_overrides_auto(monkeypatch):
    monkeypatch.setattr(providers, "local_model_available", lambda: True)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openrouter")
    assert embedding_model_id() == "openrouter:baai/bge-large-en-v1.5"
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hf")
    assert embedding_model_id().startswith("hf:")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "azure_openai")
    monkeypatch.setenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "emb1")
    assert embedding_model_id() == "azure_openai:emb1"
