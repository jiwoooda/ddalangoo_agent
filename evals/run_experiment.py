"""
LangSmith 실험 실행기.

사용법:
  python -m evals.run_experiment --mode intent
  python -m evals.run_experiment --mode intent --backend ollama --model llama3.2
  python -m evals.run_experiment --mode product
  python -m evals.run_experiment --mode response
  python -m evals.run_experiment --mode e2e

실험 이름 prefix: {mode}-{backend}-{model}  예) intent-ollama-llama3.2
결과는 LANGCHAIN_PROJECT 환경변수에 지정한 LangSmith 프로젝트에 기록된다.
"""
from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv

load_dotenv()


def _set_backend(backend: str, model: str | None, agent: str) -> None:
    """환경변수 세팅 + 모든 에이전트 LLM 캐시 초기화."""
    os.environ["LLM_BACKEND"] = backend
    if model:
        os.environ[f"{agent.upper()}_MODEL"] = model
    from configs.llm_config import reset_all_llm_caches
    reset_all_llm_caches()


def _print_summary(results, metrics: list[str]) -> None:
    try:
        df = results.to_pandas()
        print(f"\n{'─'*40}")
        for m in metrics:
            col = f"feedback.{m}"
            if col in df.columns:
                series = df[col].dropna()
                if not series.empty:
                    print(f"  {m}: mean={series.mean():.3f}  n={len(series)}")
        print(f"{'─'*40}")
    except Exception as e:
        print(f"[summary] 출력 실패: {e}")


# ── Intent 실험 ─────────────────────────────────────────────────────────

def run_intent_experiment(backend: str, model: str | None) -> None:
    from langsmith import evaluate

    from src.state.schema import get_default_shopping_state
    from src.agents.intent_agent import intent_agent_node
    from evals.evaluators import INTENT_EVALUATORS
    from evals.dataset import get_dataset_name

    dataset_name = get_dataset_name("intent")

    def predict(inputs: dict) -> dict:
        state = get_default_shopping_state("user_eval", "eval-session")
        state["messages"] = [{"role": "user", "content": inputs["user_input"]}]
        # 신포맷: stage / pending_action 필드 활용
        state["stage"] = inputs.get("stage", "idle")
        pending = inputs.get("pending_action")
        if pending:
            state["pending_action"] = {"type": pending}
        try:
            return intent_agent_node(state)
        except Exception as e:
            return {"intent": "unclear", "keywords": [], "_error": str(e)}

    prefix = f"intent-{backend}-{model or 'default'}"
    print(f"\n[experiment] 시작: {prefix}  dataset={dataset_name}")

    results = evaluate(
        predict,
        data=dataset_name,
        evaluators=INTENT_EVALUATORS,
        experiment_prefix=prefix,
        metadata={"backend": backend, "model": model or "default", "agent": "intent"},
    )
    _print_summary(results, ["intent_accuracy", "keyword_overlap", "needs_clarification_accuracy"])


# ── Product 실험 ─────────────────────────────────────────────────────────

def run_product_experiment(backend: str, model: str | None) -> None:
    """
    product_agent의 _rank() 함수를 직접 평가한다.
    JSON 파일에 포함된 candidates를 주입해 LLM 랭킹 품질만 측정.
    """
    from langsmith import evaluate

    from src.agents.product_agent import _rank
    from evals.evaluators import PRODUCT_EVALUATORS
    from evals.dataset import get_dataset_name

    dataset_name = get_dataset_name("product")

    def predict(inputs: dict) -> dict:
        try:
            ranked = _rank(
                candidates=inputs["candidates"],
                keywords=inputs.get("keywords", []),
                condition=inputs.get("condition"),
                preference_context=inputs.get("preference_context", {}),
            )
            return {
                "ranked_products": ranked,
                "top_product": ranked[0] if ranked else None,
            }
        except Exception as e:
            return {"ranked_products": [], "top_product": None, "_error": str(e)}

    prefix = f"product-{backend}-{model or 'default'}"
    print(f"\n[experiment] 시작: {prefix}  dataset={dataset_name}")

    results = evaluate(
        predict,
        data=dataset_name,
        evaluators=PRODUCT_EVALUATORS,
        experiment_prefix=prefix,
        metadata={"backend": backend, "model": model or "default", "agent": "product"},
    )
    _print_summary(results, ["ranking_accuracy"])


# ── Response 실험 ────────────────────────────────────────────────────────

def run_response_experiment(backend: str, model: str | None) -> None:
    from langsmith import evaluate

    from src.state.schema import get_default_shopping_state
    from src.agents.response_agent import response_agent_node
    from evals.evaluators import RESPONSE_EVALUATORS
    from evals.dataset import get_dataset_name

    dataset_name = get_dataset_name("response")

    def predict(inputs: dict) -> dict:
        state = get_default_shopping_state("user_eval", "eval-session")
        state["intent"] = "buy"
        state["selected_product"] = inputs["product"]
        state["keywords"] = inputs.get("keywords", [])
        state["condition"] = inputs.get("condition")
        state["messages"] = [{"role": "user", "content": "이 상품 설명해줘"}]
        state["recommendation_context"] = {
            "preference_context": inputs.get("preference_context", {})
        }
        try:
            return response_agent_node(state)
        except Exception as e:
            return {"explanation": "", "_error": str(e)}

    prefix = f"response-{backend}-{model or 'default'}"
    print(f"\n[experiment] 시작: {prefix}  dataset={dataset_name}")

    results = evaluate(
        predict,
        data=dataset_name,
        evaluators=RESPONSE_EVALUATORS,
        experiment_prefix=prefix,
        metadata={"backend": backend, "model": model or "default", "agent": "response"},
    )
    _print_summary(results, ["elderly_friendliness"])


# ── E2E 실험 ────────────────────────────────────────────────────────────

def run_e2e_experiment(backend: str, model: str | None) -> None:
    """전체 그래프 단일 턴 실행. intent 정확도 + 노인 친화도 동시 측정."""
    import uuid
    from langsmith import evaluate

    from src.graph.builder import build_graph
    from src.state.schema import get_default_shopping_state
    from evals.evaluators import E2E_EVALUATORS
    from evals.dataset import get_dataset_name

    dataset_name = get_dataset_name("intent")  # intent 데이터셋으로 E2E 평가
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


# ── CLI ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="딸랑구 LangSmith 실험 실행기")
    parser.add_argument(
        "--mode", choices=["intent", "product", "response", "e2e"], default="intent",
        help="실험 대상 에이전트 (기본: intent)",
    )
    parser.add_argument(
        "--backend", choices=["api", "ollama"], default="api",
        help="LLM 백엔드 (기본: api)",
    )
    parser.add_argument(
        "--model", default=None,
        help="모델 오버라이드 (예: llama3.2, gpt-4o, claude-sonnet-4-6)",
    )
    args = parser.parse_args()

    _set_backend(args.backend, args.model, args.mode)

    runners = {
        "intent":   run_intent_experiment,
        "product":  run_product_experiment,
        "response": run_response_experiment,
        "e2e":      run_e2e_experiment,
    }
    runners[args.mode](args.backend, args.model)


if __name__ == "__main__":
    main()
