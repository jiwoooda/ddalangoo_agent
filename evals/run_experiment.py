"""
LangSmith 실험 실행기.

사용법:
  python -m evals.run_experiment --mode intent
  python -m evals.run_experiment --mode product --backend ollama --model llama3.1:8b
  python -m evals.run_experiment --mode response --backend ollama --model llama3.1:8b
  python -m evals.run_experiment --mode context --backend ollama --model llama3.2:1b
  python -m evals.run_experiment --mode e2e
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()


def _set_backend(backend: str, model: str | None, agent: str) -> None:
    os.environ["LLM_BACKEND"] = backend
    if model:
        os.environ[f"{agent.upper()}_MODEL"] = model
    from configs.llm_config import reset_all_llm_caches
    reset_all_llm_caches()


def _print_summary(results, metrics: list[str]) -> None:
    try:
        df = results.to_pandas()
        print(f"\n{'─'*40}")
        for metric in metrics:
            col = f"feedback.{metric}"
            if col in df.columns:
                series = df[col].dropna()
                if not series.empty:
                    print(f"  {metric}: mean={series.mean():.3f}  n={len(series)}")
        print(f"{'─'*40}")
    except Exception as e:
        print(f"[summary] 출력 실패: {e}")


def run_intent_experiment(backend: str, model: str | None) -> None:
    from langsmith import evaluate

    from src.state.schema import get_default_shopping_state
    from src.agents.intent_agent import intent_agent_node
    from evals.evaluators import INTENT_EVALUATORS
    from evals.dataset import get_dataset_name

    dataset_name = get_dataset_name("intent")

    def predict(inputs: dict) -> dict:
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
        result["latency_ms"] = (time.perf_counter() - started) * 1000
        return result

    prefix = f"intent-{backend}-{model or 'default'}"
    print(f"\n[experiment] 시작: {prefix}  dataset={dataset_name}")
    results = evaluate(
        predict,
        data=dataset_name,
        evaluators=INTENT_EVALUATORS,
        experiment_prefix=prefix,
        metadata={"backend": backend, "model": model or "default", "agent": "intent"},
    )
    _print_summary(results, ["intent_accuracy", "keyword_overlap", "schema_compliance", "latency_ms"])


def run_product_experiment(backend: str, model: str | None) -> None:
    from langsmith import evaluate

    from src.agents.product_agent import _filter_results, _rank_with_metadata
    from evals.evaluators import PRODUCT_EVALUATORS
    from evals.dataset import get_dataset_name

    dataset_name = get_dataset_name("product")

    def predict(inputs: dict) -> dict:
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
        result["latency_ms"] = (time.perf_counter() - started) * 1000
        return result

    prefix = f"product-{backend}-{model or 'default'}"
    print(f"\n[experiment] 시작: {prefix}  dataset={dataset_name}")
    results = evaluate(
        predict,
        data=dataset_name,
        evaluators=PRODUCT_EVALUATORS,
        experiment_prefix=prefix,
        metadata={"backend": backend, "model": model or "default", "agent": "product"},
    )
    _print_summary(results, ["tool_call_success_rate", "mrr", "ndcg_at_3", "condition_adherence", "latency_ms"])


def run_response_experiment(backend: str, model: str | None) -> None:
    from langsmith import evaluate

    from src.state.schema import get_default_shopping_state
    from src.agents.response_agent import response_agent_node
    from evals.evaluators import RESPONSE_EVALUATORS
    from evals.dataset import get_dataset_name

    dataset_name = get_dataset_name("response")

    def predict(inputs: dict) -> dict:
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
        elapsed = (time.perf_counter() - started) * 1000
        result["latency_ms"] = elapsed
        result["ttft_ms"] = elapsed
        return result

    prefix = f"response-{backend}-{model or 'default'}"
    print(f"\n[experiment] 시작: {prefix}  dataset={dataset_name}")
    results = evaluate(
        predict,
        data=dataset_name,
        evaluators=RESPONSE_EVALUATORS,
        experiment_prefix=prefix,
        metadata={"backend": backend, "model": model or "default", "agent": "response"},
    )
    _print_summary(results, ["reflection_pass_rate", "haiku_fallback_rate", "g_eval_elderly", "ttft_ms"])


def run_context_experiment(backend: str, model: str | None) -> None:
    from langsmith import evaluate

    from src.agents.context_agent import build_preference_context
    from src.tools.mock_tools import mock_get_purchase_history
    from evals.evaluators import CONTEXT_EVALUATORS
    from evals.dataset import get_dataset_name

    dataset_name = get_dataset_name("context")

    def predict(inputs: dict) -> dict:
        started = time.perf_counter()
        user_id = inputs.get("user_id", "")
        keywords = inputs.get("keywords", [])
        histories = mock_get_purchase_history(user_id)
        pref = build_preference_context(user_id, keywords)
        return {
            "summary": pref.get("keyword_summary") or pref.get("summary", ""),
            "preference_context": pref,
            "purchase_count": len(histories),
            "keyword_history_count": len(pref.get("keyword_history") or []),
            "latency_ms": (time.perf_counter() - started) * 1000,
        }

    prefix = f"context-{backend}-{model or 'default'}"
    print(f"\n[experiment] 시작: {prefix}  dataset={dataset_name}")
    results = evaluate(
        predict,
        data=dataset_name,
        evaluators=CONTEXT_EVALUATORS,
        experiment_prefix=prefix,
        metadata={"backend": backend, "model": model or "default", "agent": "context"},
    )
    _print_summary(results, ["faithfulness", "coverage", "cold_start_pass", "conciseness", "latency_ms"])


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
            return {"intent": final.get("intent", ""), "explanation": final.get("explanation", ""), "stage": final.get("stage", "")}
        except Exception as e:
            return {"intent": "", "explanation": "", "_error": str(e)}

    prefix = f"e2e-{backend}-{model or 'default'}"
    print(f"\n[experiment] 시작: {prefix}  dataset={dataset_name}")
    results = evaluate(
        predict,
        data=dataset_name,
        evaluators=E2E_EVALUATORS,
        experiment_prefix=prefix,
        metadata={"backend": backend, "model": model or "default", "agent": "e2e"},
    )
    _print_summary(results, ["intent_accuracy", "elderly_friendliness"])


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="딸랑구 LangSmith 실험 실행기")
    parser.add_argument(
        "--mode",
        choices=["intent", "product", "response", "context", "e2e"],
        default="intent",
        help="실험 대상 에이전트",
    )
    parser.add_argument("--backend", choices=["api", "ollama"], default="api", help="LLM 백엔드")
    parser.add_argument("--model", default=None, help="모델 오버라이드")
    args = parser.parse_args()

    _set_backend(args.backend, args.model, args.mode)
    runners = {
        "intent": run_intent_experiment,
        "product": run_product_experiment,
        "response": run_response_experiment,
        "context": run_context_experiment,
        "e2e": run_e2e_experiment,
    }
    runners[args.mode](args.backend, args.model)


if __name__ == "__main__":
    main()
