"""Provider factories. RAG code depends only on LangChain interfaces, so
switching OpenRouter -> Azure OpenAI is an env change (LLM_PROVIDER / EMBEDDING_PROVIDER)."""
import os

from core.config import EMBEDDING_MODEL_DEFAULT


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def get_llm(temperature: float = 0.0, callbacks=None):
    provider = _env("LLM_PROVIDER", "openrouter").lower()
    if provider == "openrouter":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=_env("OPENROUTER_MODEL", "anthropic/claude-sonnet-4.5"),
            api_key=_env("OPENROUTER_API_KEY"),
            base_url=_env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            temperature=temperature,
            callbacks=callbacks,
        )
    if provider == "azure_openai":
        from langchain_openai import AzureChatOpenAI

        return AzureChatOpenAI(
            azure_endpoint=_env("AZURE_OPENAI_ENDPOINT"),
            api_key=_env("AZURE_OPENAI_API_KEY"),
            api_version=_env("AZURE_OPENAI_API_VERSION", "2024-06-01"),
            azure_deployment=_env("AZURE_OPENAI_CHAT_DEPLOYMENT"),
            temperature=temperature,
            callbacks=callbacks,
        )
    raise ValueError(f"Unknown LLM_PROVIDER: {provider}")


OPENROUTER_EMBEDDING_DEFAULT = "baai/bge-large-en-v1.5"  # same model as spec, hosted (no torch)


def local_model_available() -> bool:
    try:
        from core.local_embed import find_model_dir

        return find_model_dir() is not None
    except Exception:  # numpy/tokenizers missing
        return False


def resolve_embedding_provider() -> str:
    """'auto' (default) = already-cached local bge-small if present, otherwise labelled MOCK.
    Hosted/other providers are used only when explicitly selected."""
    p = _env("EMBEDDING_PROVIDER", "auto").lower()
    if p == "auto":
        return "local" if local_model_available() else "mock"
    return p


def embedding_mode() -> dict:
    """What the UI shows so nobody mistakes a stand-in for the spec's bge-large (1024-d)."""
    p = resolve_embedding_provider()
    if p == "mock":
        from core.mock_embed import MOCK_LABEL

        return {"provider": p, "mock": True, "label": MOCK_LABEL, "dim": 256}
    if p == "local":
        return {"provider": p, "mock": False, "dim": 384,
                "label": "Local bge-small-en-v1.5 (384-d, numpy runtime) - stand-in for the spec's bge-large 1024-d"}
    return {"provider": p, "mock": False, "dim": None, "label": f"Embeddings via {p}"}


def embedding_model_id() -> str:
    """Identifier stored in index metadata; a mismatch forces re-indexing."""
    provider = resolve_embedding_provider()
    if provider == "mock":
        return "mock:hash-256"
    if provider == "local":
        return "local:BAAI/bge-small-en-v1.5"
    if provider == "hf":
        return f"hf:{_env('HF_EMBEDDING_MODEL', EMBEDDING_MODEL_DEFAULT)}"
    if provider == "openrouter":
        return f"openrouter:{_env('OPENROUTER_EMBEDDING_MODEL', OPENROUTER_EMBEDDING_DEFAULT)}"
    return f"azure_openai:{_env('AZURE_OPENAI_EMBEDDING_DEPLOYMENT')}"


def get_embeddings():
    provider = resolve_embedding_provider()
    if provider == "mock":
        from core.mock_embed import MockEmbeddings

        return MockEmbeddings()
    if provider == "local":
        from core.local_embed import LocalBgeSmallEmbeddings

        return LocalBgeSmallEmbeddings()
    if provider == "openrouter":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=_env("OPENROUTER_EMBEDDING_MODEL", OPENROUTER_EMBEDDING_DEFAULT),
            api_key=_env("OPENROUTER_API_KEY"),
            base_url=_env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            # Send raw text: OpenAI's tiktoken pre-splitting is wrong for non-OpenAI models.
            check_embedding_ctx_length=False,
        )
    if provider == "hf":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=_env("HF_EMBEDDING_MODEL", EMBEDDING_MODEL_DEFAULT),
            encode_kwargs={"normalize_embeddings": True},
        )
    if provider == "azure_openai":
        from langchain_openai import AzureOpenAIEmbeddings

        return AzureOpenAIEmbeddings(
            azure_endpoint=_env("AZURE_OPENAI_ENDPOINT"),
            api_key=_env("AZURE_OPENAI_API_KEY"),
            api_version=_env("AZURE_OPENAI_API_VERSION", "2024-06-01"),
            azure_deployment=_env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        )
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {provider}")
