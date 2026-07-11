"""
LangSmith 실험 실행기.

사용법:
  python -m evals.run_experiment --agent intent
  python -m evals.run_experiment --agent product --backend ollama --model xlam-2:32b-fc-r
  python -m evals.run_experiment --agent response --backend ollama --model qwen2.5:7b
  python -m evals.run_experiment --agent context --backend ollama --model qwen2.5:14b
  python -m evals.run_experiment --agent all --backend api
  python -m evals.run_experiment --agent e2e
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time

from dotenv import load_dotenv
from langchain_core.callbacks.base import BaseCallbackHandler

load_dotenv()

# ── 에이전트 키 집합 ──────────────────────────────────────────────────────
_AGENT_MODEL_KEYS = {"intent", "product", "response", "context", "recipe"}

# ── 토큰 사용량 캡처 ─────────────────────────────────────────────────────
_tls = threading.local()


class _UsageCapture(BaseCallbackHandler):
    """LLM on_llm_end 마다 thread-local에 input/output 토큰 누적."""

    def on_llm_end(self, response, **kwargs):  # noqa: ANN001
        store = getattr(_tls, 'usage', None)
        if store is None:
            return
        inp = out = 0
        # Anthropic / standard: llm_output.usage
        llm_out = getattr(response, 'llm_output', None) or {}
        usage = llm_out.get('usage', {}) or {}
        inp += usage.get('input_tokens', 0) or usage.get('prompt_tokens', 0)
        out += usage.get('output_tokens', 0) or usage.get('completion_tokens', 0)
        # Ollama: generation_info.prompt_eval_count / eval_count
        if inp == 0 and out == 0:
            for gen_list in (getattr(response, 'generations', None) or []):
                for gen in gen_list:
                    gi = getattr(gen, 'generation_info', None) or {}
                    inp += gi.get('prompt_eval_count', 0)
                    out += gi.get('eval_count', 0)
        store['input_tokens'] += inp
        store['output_tokens'] += out


_USAGE_CAPTURE = _UsageCapture()


def _init_usage() -> None:
    """predict() 시작 시 thread-local 카운터 초기화."""
    _tls.usage = {'input_tokens': 0, 'output_tokens': 0}


def _read_usage() -> tuple[int, int]:
    u = getattr(_tls, 'usage', {}) or {}
    return u.get('input_tokens', 0), u.get('output_tokens', 0)


# ── 과금 단가 (per 1M tokens, USD) ───────────────────────────────────────
_PRICING: dict[str, dict[str, float]] = {
    "claude-haiku-4-5-20251001": {"input": 0.80,  "output": 4.00},
    "claude-sonnet-4-6":         {"input": 3.00,  "output": 15.00},
    "claude-opus-4-8":           {"input": 15.00, "output": 75.00},
    "gpt-4o-mini":               {"input": 0.15,  "output": 0.60},
    "gpt-4o":                    {"input": 5.00,  "output": 15.00},
}


def _calc_cost(model: str, inp: int, out: int, backend: str) -> float:
    if backend in ("ollama", "vllm"):
        return 0.0
    p = _PRICING.get(model, {"input": 0.0, "output": 0.0})
    return (inp * p["input"] + out * p["output"]) / 1_000_000


def _effective_model(agent: str, backend: str) -> str:
    from configs.llm_config import _DEFAULT_MODELS, _ENV_KEYS
    default = _DEFAULT_MODELS.get(backend, {}).get(agent, "")
    return os.getenv(_ENV_KEYS.get(agent, ""), default)


# ── 백엔드 설정 ──────────────────────────────────────────────────────────
def _set_backend(backend: str, model: str | None, agent: str) -> None:
    from configs.llm_config import _DEFAULT_MODELS, reset_all_llm_caches, set_eval_callbacks
    os.environ["LLM_BACKEND"] = backend
    # load_dotenv()가 나중에 재실행되어도 .env 값을 복원하지 못하도록
    # pop 대신 backend 기본값으로 명시적 설정 (override=False인 load_dotenv가 기존 값을 덮지 않음)
    for ag in _AGENT_MODEL_KEYS:
        default = _DEFAULT_MODELS.get(backend, {}).get(ag)
        if default:
            os.environ[f"{ag.upper()}_MODEL"] = default
        else:
            os.environ.pop(f"{ag.upper()}_MODEL", None)
    # 명시적 모델 지정 시 해당 에이전트만 오버라이드
    if model and agent in _AGENT_MODEL_KEYS:
        os.environ[f"{agent.upper()}_MODEL"] = model
    reset_all_llm_caches()
    set_eval_callbacks([_USAGE_CAPTURE])


def _print_summary(results, metrics: list[str]) -> None:
    try:
        df = results.to_pandas()
        print(f"\n{'─'*56}")
        for metric in metrics:
            col = f"feedback.{metric}"
            if col in df.columns:
                series = df[col].dropna()
                if series.empty:
                    continue
                if metric == "cost_usd":
                    print(f"  {metric:<30} total=${series.sum():.6f}  n={len(series)}")
                elif metric in ("latency_ms", "ttft_ms"):
                    print(f"  {metric:<30} mean={series.mean():.1f}ms  p90={series.quantile(.9):.1f}ms")
                elif metric in ("input_tokens", "output_tokens"):
                    print(f"  {metric:<30} mean={series.mean():.0f}  total={int(series.sum())}")
                else:
                    print(f"  {metric:<30} mean={series.mean():.3f}  n={len(series)}")
        print(f"{'─'*56}")
    except Exception as e:
        print(f"[summary] 출력 실패: {e}")


# ── 공통 latency/cost 필드 추가 헬퍼 ────────────────────────────────────
def _attach_metrics(result: dict, elapsed_ms: float, agent: str, backend: str) -> dict:
    inp, out = _read_usage()
    model = _effective_model(agent, backend)
    result["latency_ms"]    = elapsed_ms
    result["ttft_ms"]       = elapsed_ms   # 스트리밍 미적용 시 latency와 동일
    result["input_tokens"]  = inp
    result["output_tokens"] = out
    result["cost_usd"]      = _calc_cost(model, inp, out, backend)
    return result


# ── Intent 실험 ──────────────────────────────────────────────────────────
def run_intent_experiment(backend: str, model: str | None) -> None:
    from langsmith import evaluate
    from langsmith import tracing_context

    from src.state.schema import get_default_shopping_state
    from src.agents.intent_agent import intent_agent_node
    from evals.evaluators import INTENT_EVALUATORS
    from evals.dataset import get_dataset_name

    dataset_name = get_dataset_name("intent")

    def predict(inputs: dict) -> dict:
        with tracing_context(enabled=False):
            _init_usage()
            started = time.perf_counter()
            state = get_default_shopping_state("user_eval", "eval-session")
            state["messages"] = [{"role": "user", "content": inputs["user_input"]}]
            state["stage"] = inputs.get("stage", "idle")
            if pending := inputs.get("pending_action"):
                state["pending_action"] = {"type": pending}
            try:
                result = intent_agent_node(state)
                result["_schema_ok"] = True
            except Exception as e:
                result = {"intent": "unclear", "keywords": [], "_schema_ok": False, "_error": str(e)}
            return _attach_metrics(result, (time.perf_counter() - started) * 1000, "intent", backend)

    prefix = f"intent-{backend}-{model or 'baseline'}"
    print(f"\n[experiment] 시작: {prefix}  dataset={dataset_name}")
    results = evaluate(
        predict,
        data=dataset_name,
        evaluators=INTENT_EVALUATORS,
        experiment_prefix=prefix,
        metadata={"backend": backend, "model": model or "default", "agent": "intent"},
    )
    _print_summary(results, [
        "intent_accuracy", "keyword_overlap", "schema_compliance", "schema_loose",
        "latency_ms", "ttft_ms", "input_tokens", "output_tokens", "cost_usd",
    ])


# ── Product 실험 ─────────────────────────────────────────────────────────
def run_product_experiment(backend: str, model: str | None) -> None:
    from langsmith import evaluate
    from langsmith import tracing_context

    from src.agents.product_agent import _filter_results, _rank_with_metadata
    from evals.evaluators import PRODUCT_EVALUATORS
    from evals.dataset import get_dataset_name

    dataset_name = get_dataset_name("product")

    def predict(inputs: dict) -> dict:
        with tracing_context(enabled=False):
            _init_usage()
            started = time.perf_counter()
            try:
                candidates = _filter_results(inputs["candidates"], inputs.get("exclude_keywords", []))
                ranked_result = _rank_with_metadata(
                    candidates=candidates,
                    keywords=inputs.get("keywords", []),
                    condition=inputs.get("condition"),
                    preference_context=inputs.get("preference_context", {}),
                )
                ranked = ranked_result["ranked_products"]
                result = {
                    "ranked_products": ranked,
                    "top_product": ranked[0] if ranked else None,
                    "tool_call_success": ranked_result.get("tool_call_success", False),
                    "tool_call_error": ranked_result.get("tool_call_error"),
                }
            except Exception as e:
                result = {
                    "ranked_products": [],
                    "top_product": None,
                    "tool_call_success": False,
                    "tool_call_error": str(e),
                    "_error": str(e),
                }
            return _attach_metrics(result, (time.perf_counter() - started) * 1000, "product", backend)

    prefix = f"product-{backend}-{model or 'baseline'}"
    print(f"\n[experiment] 시작: {prefix}  dataset={dataset_name}")
    results = evaluate(
        predict,
        data=dataset_name,
        evaluators=PRODUCT_EVALUATORS,
        experiment_prefix=prefix,
        metadata={"backend": backend, "model": model or "default", "agent": "product"},
    )
    _print_summary(results, [
        "tool_call_success_rate", "mrr", "ndcg_at_3", "condition_adherence",
        "hallucination_free_rate",
        "latency_ms", "ttft_ms", "input_tokens", "output_tokens", "cost_usd",
    ])


# ── Response 실험 ────────────────────────────────────────────────────────
def run_response_experiment(backend: str, model: str | None) -> None:
    from langsmith import evaluate
    from langsmith import tracing_context

    from src.state.schema import get_default_shopping_state
    from src.agents.response_agent import response_agent_node
    from evals.evaluators import RESPONSE_EVALUATORS
    from evals.dataset import get_dataset_name

    dataset_name = get_dataset_name("response")

    def predict(inputs: dict) -> dict:
        with tracing_context(enabled=False):
            _init_usage()
            started = time.perf_counter()
            state = get_default_shopping_state("user_eval", "eval-session")
            state["intent"] = "ask" if inputs.get("task") == "qa" else "buy"
            state["selected_product"] = inputs["product"]
            state["keywords"] = inputs.get("keywords", [])
            state["condition"] = inputs.get("condition")
            state["messages"] = [{"role": "user", "content": inputs.get("question") or "이 상품 설명해줘"}]
            state["recommendation_context"] = {"preference_context": inputs.get("preference_context", {})}
            try:
                result = response_agent_node(state)
            except Exception as e:
                result = {"explanation": "", "_error": str(e)}
            return _attach_metrics(result, (time.perf_counter() - started) * 1000, "response", backend)

    prefix = f"response-{backend}-{model or 'baseline'}"
    print(f"\n[experiment] 시작: {prefix}  dataset={dataset_name}")
    results = evaluate(
        predict,
        data=dataset_name,
        evaluators=RESPONSE_EVALUATORS,
        experiment_prefix=prefix,
        metadata={"backend": backend, "model": model or "default", "agent": "response"},
    )
    _print_summary(results, [
        "reflection_pass_rate", "haiku_fallback_rate", "g_eval_elderly",
        "latency_ms", "ttft_ms", "input_tokens", "output_tokens", "cost_usd",
    ])


# ── Context 실험 ─────────────────────────────────────────────────────────
def run_context_experiment(backend: str, model: str | None) -> None:
    from langsmith import evaluate
    from langsmith import tracing_context

    from src.agents.context_agent import build_preference_context
    from src.tools.mock_tools import mock_get_purchase_history
    from evals.evaluators import CONTEXT_EVALUATORS
    from evals.dataset import get_dataset_name

    dataset_name = get_dataset_name("context")

    def predict(inputs: dict) -> dict:
        with tracing_context(enabled=False):
            _init_usage()
            started = time.perf_counter()
            user_id = inputs.get("user_id", "")
            keywords = inputs.get("keywords", [])
            histories = mock_get_purchase_history(user_id)
            pref = build_preference_context(user_id, keywords)

            retrieval_context = [
                f"{h.get('product_name', '')} / 브랜드:{h.get('brand', '')} / "
                f"가격:{h.get('price_at_purchase', '')}원 / 플랫폼:{h.get('platform', '')}"
                for h in histories
            ] or ["구매이력 없음"]

            result = {
                **pref,
                "summary": pref.get("keyword_summary") or pref.get("summary", ""),
                "preference_context": pref,
                "purchase_count": len(histories),
                "keyword_history_count": len(pref.get("keyword_history") or []),
                "_retrieval_context": retrieval_context,
            }
            return _attach_metrics(result, (time.perf_counter() - started) * 1000, "context", backend)

    prefix = f"context-{backend}-{model or 'baseline'}"
    print(f"\n[experiment] 시작: {prefix}  dataset={dataset_name}")
    results = evaluate(
        predict,
        data=dataset_name,
        evaluators=CONTEXT_EVALUATORS,
        experiment_prefix=prefix,
        metadata={"backend": backend, "model": model or "default", "agent": "context"},
    )
    _print_summary(results, [
        "coverage", "faithfulness", "cold_start_pass", "conciseness",
        "latency_ms", "ttft_ms", "input_tokens", "output_tokens", "cost_usd",
    ])


# ── E2E 실험 (기존 유지) ─────────────────────────────────────────────────
def run_e2e_experiment(backend: str, model: str | None) -> None:
    import uuid
    from langsmith import evaluate

    from src.graph.builder import build_graph
    from src.state.schema import get_default_shopping_state
    from evals.evaluators import E2E_EVALUATORS
    from evals.dataset import get_dataset_name

    dataset_name = get_dataset_name("intent")
    graph = build_graph()

    def predict(inputs: dict) -> dict:
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        state = get_default_shopping_state("user_eval", str(uuid.uuid4()))
        state["messages"] = [{"role": "user", "content": inputs["user_input"]}]
        try:
            final = graph.invoke(state, config)
            return {
                "intent": final.get("intent", ""),
                "explanation": final.get("explanation", ""),
                "stage": final.get("stage", ""),
            }
        except Exception as e:
            return {"intent": "", "explanation": "", "_error": str(e)}

    prefix = f"e2e-{backend}-{model or 'baseline'}"
    print(f"\n[experiment] 시작: {prefix}  dataset={dataset_name}")
    results = evaluate(
        predict,
        data=dataset_name,
        evaluators=E2E_EVALUATORS,
        experiment_prefix=prefix,
        metadata={"backend": backend, "model": model or "default", "agent": "e2e"},
    )
    _print_summary(results, ["intent_accuracy", "elderly_friendliness"])


# ── CLI ──────────────────────────────────────────────────────────────────
_RUNNERS = {
    "intent":   run_intent_experiment,
    "product":  run_product_experiment,
    "response": run_response_experiment,
    "context":  run_context_experiment,
    "e2e":      run_e2e_experiment,
}
_ALL_AGENTS = ["intent", "product", "response", "context"]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="딸랑구 LangSmith 실험 실행기")
    parser.add_argument(
        "--agent",
        choices=[*_ALL_AGENTS, "e2e", "all"],
        default="intent",
        help="실험 대상 에이전트 (all=전체 순차 실행)",
    )
    parser.add_argument("--backend", choices=["api", "ollama", "vllm"], default="api", help="LLM 백엔드")
    parser.add_argument("--model", default=None, help="모델 오버라이드")
    args = parser.parse_args()

    targets = _ALL_AGENTS if args.agent == "all" else [args.agent]
    for ag in targets:
        _set_backend(args.backend, args.model, ag)
        _RUNNERS[ag](args.backend, args.model)


if __name__ == "__main__":
    main()
