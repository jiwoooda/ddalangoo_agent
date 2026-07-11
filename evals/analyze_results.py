"""
1차 실험 결과 분석기.

주요 기능:
  1. Strict fail + Loose pass 케이스 추출 → 2차(prompt 튜닝) 대상 후보 리스트업
  2. 두 실험 메트릭 비교 (API vs Ollama, 또는 A후보 vs B후보)
  3. 교체 기준 달성 여부 체크

비교 플로우:
  Step 1) 각 후보를 API baseline과 비교 → 기준선 대비 격차 측정
    python -m evals.analyze_results --compare intent-api-baseline intent-ollama-qwen3:32b
    python -m evals.analyze_results --compare intent-api-baseline intent-ollama-qwen2.5:72b

  Step 2) 기준 달성한 후보들끼리 A vs B 비교 → 최종 선택
    python -m evals.analyze_results --compare intent-ollama-qwen3:32b intent-ollama-qwen2.5:72b

  Step 3) 교체 기준 체크
    python -m evals.analyze_results --check-thresholds intent-ollama-qwen3:32b

  Step 4) Intent/Product 실패 케이스 → 2차 실험(prompt 튜닝) 후보 추출
    python -m evals.analyze_results --experiment intent-ollama-qwen3:32b
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# ── 교체 기준 (스펙 4번 참고) ────────────────────────────────────────────
# _ttft_vs_api: True → 단독 임계값 없음, API baseline 대비 낮아야 교체 가능
#               --compare 명령으로 delta 확인 후 사람이 판단
REPLACEMENT_THRESHOLDS: dict[str, Any] = {
    "intent": {
        "intent_accuracy":   0.90,   # Exact Match 90% 이상
        "schema_compliance": 0.95,   # Schema Strict 95% 이상
    },
    "product": {
        "mrr":                    0.80,   # MRR 0.8 이상 (⭐메인)
        "tool_call_success_rate": 0.95,   # Tool Call Success 95% 이상
        "hallucination_free_rate": 0.98,  # 환각 방지율 98% 이상
    },
    "response": {
        "reflection_pass_rate": 0.85,   # Reflection Pass Rate 85% 이상
        "_ttft_vs_api": True,           # TTFT는 API baseline 대비 개선 여부 비교 필요
    },
    "context": {
        "faithfulness": 0.80,   # Faithfulness 0.8 이상
        "coverage":     0.50,   # expected_terms 50% 이상 포함 (Coverage 통과)
    },
}


def _get_runs(experiment_name: str) -> list[dict[str, Any]]:
    """LangSmith에서 실험 결과 row 리스트를 가져온다."""
    from langsmith import Client
    client = Client()

    # experiment_name으로 프로젝트 내 실험 찾기
    runs = list(client.list_runs(
        project_name=None,
        filter=f'has(tags, "{experiment_name}")',
        run_type="chain",
    ))
    if not runs:
        # experiment_prefix로 등록된 실험에서 결과 가져오기
        try:
            results = client.get_test_results(project_name=experiment_name)
            return [r.__dict__ for r in results]
        except Exception:
            pass
    return [r.__dict__ for r in runs]


def _fetch_experiment_rows(experiment_name: str) -> list[dict[str, Any]]:
    """
    LangSmith evaluate() 결과를 row 단위로 반환.
    각 row: {input, output, reference_output, feedback: {metric: score}}
    """
    from langsmith import Client
    client = Client()

    rows = []
    try:
        # evaluate()가 생성하는 experiment project에서 결과 가져오기
        for run in client.list_runs(
            project_name=experiment_name,
            run_type="chain",
            is_root=True,
        ):
            feedback = {}
            for fb in client.list_feedback(run_ids=[str(run.id)]):
                feedback[fb.key] = fb.score
            rows.append({
                "run_id": str(run.id),
                "inputs": run.inputs or {},
                "outputs": run.outputs or {},
                "feedback": feedback,
            })
    except Exception as e:
        print(f"[analyze] LangSmith 조회 실패: {e}")
    return rows


# ── 1. Strict fail + Loose pass 추출 ────────────────────────────────────

def extract_strict_loose_mismatch(experiment_name: str) -> None:
    """
    schema_compliance(Strict)=0 이면서 schema_loose(Loose)=1인 케이스 출력.
    → 2차 실험(prompt 튜닝) 후보.
    """
    print(f"\n=== Strict fail + Loose pass 추출: {experiment_name} ===")
    rows = _fetch_experiment_rows(experiment_name)
    if not rows:
        print("  결과 없음 (experiment 이름 확인)")
        return

    mismatches = [
        r for r in rows
        if r["feedback"].get("schema_compliance") == 0.0
        and r["feedback"].get("schema_loose") == 1.0
    ]

    print(f"전체 {len(rows)}개 중 Strict=0 + Loose=1: {len(mismatches)}개")
    print(f"→ 2차 실험(prompt 튜닝) 대상: {len(mismatches)}개\n")

    for i, r in enumerate(mismatches, 1):
        user_input = r["inputs"].get("user_input", r["inputs"])
        out = r["outputs"]
        fb = r["feedback"]
        print(f"[{i}] run_id={r['run_id'][:8]}...")
        print(f"     input : {str(user_input)[:80]}")
        print(f"     intent: {out.get('intent')}  keywords: {out.get('keywords')}")
        print(f"     error : {out.get('_error', '없음')}")
        print(f"     scores: strict={fb.get('schema_compliance')} loose={fb.get('schema_loose')} "
              f"accuracy={fb.get('intent_accuracy')}")
        print()

    if mismatches:
        print("→ 위 케이스들은 모델 체급 한계가 아니라 prompt 미스매치 가능성 높음.")
        print("  2차 실험: experiment_prefix에 '-tuned' 붙여서 별도 실험으로 진행할 것.")


# ── 2. API vs Ollama 비교 ────────────────────────────────────────────────

def compare_experiments(exp_api: str, exp_ollama: str) -> None:
    """두 실험의 메트릭 평균을 나란히 출력."""
    print(f"\n=== 비교: {exp_api}  vs  {exp_ollama} ===")
    rows_api = _fetch_experiment_rows(exp_api)
    rows_ollama = _fetch_experiment_rows(exp_ollama)

    if not rows_api and not rows_ollama:
        print("  두 실험 모두 결과 없음")
        return

    def _avg(rows: list[dict], key: str) -> str:
        scores = [r["feedback"][key] for r in rows if key in r["feedback"] and r["feedback"][key] is not None]
        if not scores:
            return "N/A"
        return f"{sum(scores)/len(scores):.3f} (n={len(scores)})"

    all_keys: set[str] = set()
    for r in rows_api + rows_ollama:
        all_keys.update(r["feedback"].keys())

    header = f"{'metric':<28} {'API':>16} {'Ollama':>16} {'delta':>10}"
    print(header)
    print("─" * len(header))

    for key in sorted(all_keys):
        api_str = _avg(rows_api, key)
        ollama_str = _avg(rows_ollama, key)

        try:
            api_val = float(api_str.split()[0]) if api_str != "N/A" else None
            ollama_val = float(ollama_str.split()[0]) if ollama_str != "N/A" else None
            delta = f"{ollama_val - api_val:+.3f}" if api_val is not None and ollama_val is not None else "N/A"
        except Exception:
            delta = "N/A"

        print(f"  {key:<26} {api_str:>16} {ollama_str:>16} {delta:>10}")


# ── 3. 교체 기준 달성 여부 체크 ──────────────────────────────────────────

def check_thresholds(experiment_name: str) -> None:
    """
    스펙의 교체 기준과 비교해 pass/fail 출력.
    agent 이름은 experiment_name 앞부분에서 추론 (intent-, product-, ...).
    """
    agent = experiment_name.split("-")[0]
    thresholds = REPLACEMENT_THRESHOLDS.get(agent)

    print(f"\n=== 교체 기준 체크: {experiment_name} (agent={agent}) ===")
    if not thresholds:
        print(f"  교체 기준 없음 (지원 agent: {list(REPLACEMENT_THRESHOLDS)})")
        return

    rows = _fetch_experiment_rows(experiment_name)
    if not rows:
        print("  결과 없음")
        return

    all_pass = True
    ttft_needs_compare = False

    for metric, threshold in thresholds.items():
        if metric.startswith("_"):
            # 메타 마커 처리
            if metric == "_ttft_vs_api" and threshold:
                ttft_needs_compare = True
                print(f"  ⚠️  ttft_ms               — API baseline 대비 비교 필요 "
                      f"(--compare {experiment_name} {experiment_name.replace('ollama', 'api').replace(experiment_name.split('-')[2], 'baseline')})")
            continue

        scores = [
            r["feedback"][metric]
            for r in rows
            if metric in r["feedback"] and r["feedback"][metric] is not None
        ]
        if not scores:
            print(f"  {'?':2} {metric:<26} N/A (데이터 없음)")
            continue
        avg = sum(scores) / len(scores)
        ok = avg >= threshold
        if not ok:
            all_pass = False
        mark = "✅" if ok else "❌"
        print(f"  {mark} {metric:<26} {avg:.3f} (기준 ≥{threshold:.2f}, n={len(scores)})")

    print()
    if all_pass and not ttft_needs_compare:
        print("→ 교체 기준 달성 ✅ — Ollama 모델로 전환 가능")
    elif all_pass and ttft_needs_compare:
        print("→ 수치 기준 달성 ✅ — TTFT 비교 후 최종 판단 필요 (위 --compare 명령 실행)")
    else:
        print("→ 교체 기준 미달 ❌ — API 유지 또는 2차 실험(prompt 튜닝) 검토")


# ── CLI ─────────────────────────────────────────────────────────────────

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="딸랑구 실험 결과 분석기")
    parser.add_argument("--experiment", "-e", help="분석할 실험명 (Strict/Loose 추출)")
    parser.add_argument("--compare", nargs=2, metavar=("API_EXP", "OLLAMA_EXP"),
                        help="두 실험 비교: --compare intent-api-baseline intent-ollama-qwen2.5:7b")
    parser.add_argument("--check-thresholds", "-t", metavar="EXPERIMENT",
                        help="교체 기준 달성 여부 체크")
    args = parser.parse_args()

    if args.experiment:
        extract_strict_loose_mismatch(args.experiment)
    elif args.compare:
        compare_experiments(args.compare[0], args.compare[1])
    elif args.check_thresholds:
        check_thresholds(args.check_thresholds)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
