"""
LLM 팩토리.

환경변수로 백엔드와 모델을 선택한다.

  LLM_BACKEND=api     → OpenAI / Anthropic API (기본값)
  LLM_BACKEND=ollama  → 로컬 Ollama 서버

모델 오버라이드:
  INTENT_MODEL=llama3.2
  PRODUCT_MODEL=llama3.1:8b
  RESPONSE_MODEL=llama3.1:8b
  CONTEXT_MODEL=llama3.2:1b
  OLLAMA_BASE_URL=http://localhost:11434
"""
import os
from langchain_core.language_models import BaseChatModel

_DEFAULT_MODELS: dict[str, dict[str, str]] = {
    "api": {
        "intent":   "gpt-4o-mini",
        "product":  "claude-sonnet-4-6",
        "response": "claude-sonnet-4-6",
        "context":  "claude-haiku-4-5-20251001",
        "recipe":   "claude-haiku-4-5-20251001",
    },
    "ollama": {
        "intent":   "llama3.2",
        "product":  "llama3.1:8b",
        "response": "llama3.1:8b",
        "context":  "llama3.2:1b",
        "recipe":   "llama3.2",
    },
}

_ENV_KEYS: dict[str, str] = {
    "intent":   "INTENT_MODEL",
    "product":  "PRODUCT_MODEL",
    "response": "RESPONSE_MODEL",
    "context":  "CONTEXT_MODEL",
    "recipe":   "RECIPE_MODEL",
}


def get_llm(agent: str, **kwargs) -> BaseChatModel:
    """
    agent: "intent" | "product" | "response" | "context"
    kwargs: temperature, max_tokens 등 — 백엔드별로 자동 변환
    """
    backend = os.getenv("LLM_BACKEND", "api")
    model = os.getenv(_ENV_KEYS[agent], _DEFAULT_MODELS[backend][agent])

    if backend == "ollama":
        from langchain_ollama import ChatOllama
        # Ollama는 max_tokens 대신 num_predict 사용
        ollama_kwargs = {k: v for k, v in kwargs.items() if k != "max_tokens"}
        if "max_tokens" in kwargs:
            ollama_kwargs["num_predict"] = kwargs["max_tokens"]
        return ChatOllama(
            model=model,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            **ollama_kwargs,
        )

    if agent == "intent":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, **kwargs)

    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(model=model, **kwargs)  # product / response / context / recipe


def reset_all_llm_caches() -> None:
    """LLM 교체 실험 시 모든 에이전트 모듈의 싱글톤 캐시를 초기화한다."""
    try:
        import src.agents.intent_agent as ia
        ia._llm = None
        ia._structured_llm = None
    except Exception:
        pass
    try:
        import src.agents.product_agent as pa
        pa._llm = None
    except Exception:
        pass
    try:
        import src.agents.response_agent as ra
        ra._llm = None
    except Exception:
        pass
    try:
        import src.agents.context_agent as ca
        ca._context_llm = None
    except Exception:
        pass
