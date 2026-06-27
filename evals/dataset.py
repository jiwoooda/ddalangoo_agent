"""
LangSmith Dataset 관리.

테스트 케이스는 evals/data/*.json 파일에서 관리한다.
이 모듈은 JSON 로드 + LangSmith 업로드만 담당한다.

사용법:
  python -m evals.dataset --agent intent          # 단일 에이전트 업로드
  python -m evals.dataset --agent product
  python -m evals.dataset --agent response
  python -m evals.dataset --all                   # 전체 업로드
  python -m evals.dataset --agent intent --dry-run  # 로컬 출력만
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_DATA_DIR = Path(__file__).parent / "data"

# ── 에이전트별 JSON 파일 매핑 ────────────────────────────────────────────
_AGENT_FILES: dict[str, Path] = {
    "intent":   _DATA_DIR / "intent_agent.json",
    "product":  _DATA_DIR / "product_agent.json",
    "response": _DATA_DIR / "response_agent.json",
}

AGENTS = list(_AGENT_FILES.keys())


def _load_json(agent: str) -> dict[str, Any]:
    path = _AGENT_FILES[agent]
    if not path.exists():
        raise FileNotFoundError(f"데이터 파일 없음: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    # 배열 형식(flat list) → 래퍼 형식으로 정규화
    if isinstance(raw, list):
        return {
            "dataset_name": f"ddalangoo-{agent}-v1",
            "description": f"딸랑구 {agent} 평가셋",
            "cases": raw,
        }
    return raw


def _get_output(case: dict) -> dict:
    """'output' 또는 'expected_output' 키를 통일해서 반환."""
    return case.get("output") or case.get("expected_output") or {}


def load_cases(agent: str) -> list[dict]:
    """JSON 파일에서 테스트 케이스 목록만 반환."""
    return _load_json(agent)["cases"]


def get_dataset_name(agent: str) -> str:
    """LangSmith Dataset 이름 반환."""
    return _load_json(agent)["dataset_name"]


def get_dataset_description(agent: str) -> str:
    return _load_json(agent).get("description", "")


# ── LangSmith 업로드 ─────────────────────────────────────────────────────

def upload_dataset(agent: str) -> None:
    from langsmith import Client

    data = _load_json(agent)
    dataset_name = data["dataset_name"]
    description = data.get("description", "")
    cases = data["cases"]

    # input/output 추출 (output 또는 expected_output 키 모두 지원)
    clean_cases = [
        {"input": c["input"], "output": _get_output(c)}
        for c in cases
        if "input" in c and ("output" in c or "expected_output" in c)
    ]

    client = Client()
    existing = [d for d in client.list_datasets() if d.name == dataset_name]
    if existing:
        print(f"[{agent}] 이미 존재: '{dataset_name}' (id={existing[0].id})")
        print(f"  → 덮어쓰려면 LangSmith UI에서 삭제 후 재실행하세요.")
        return

    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description=description,
    )
    client.create_examples(
        inputs=[c["input"] for c in clean_cases],
        outputs=[c["output"] for c in clean_cases],
        dataset_id=dataset.id,
    )
    print(f"[{agent}] 업로드 완료: {len(clean_cases)}개 → '{dataset_name}' (id={dataset.id})")


def upload_all() -> None:
    for agent in AGENTS:
        upload_dataset(agent)


# ── 드라이런 ─────────────────────────────────────────────────────────────

def dry_run(agent: str) -> None:
    from collections import Counter

    data = _load_json(agent)
    cases = [c for c in data["cases"] if "input" in c and ("output" in c or "expected_output" in c)]

    print(f"\n[{agent}] '{data['dataset_name']}' ({len(cases)}개)")
    print(f"  {data.get('description', '')}")

    if agent == "intent":
        intents = [
            _get_output(c).get("intent") or _get_output(c).get("expected_intent", "?")
            for c in cases
        ]
        for intent, count in sorted(Counter(intents).items()):
            print(f"  {intent:<20} {count}개")
        # 카테고리 분포 (새 포맷에 있을 경우)
        if any("category" in c for c in cases):
            cats = Counter(c.get("category", "?") for c in cases)
            print()
            for cat, count in sorted(cats.items()):
                print(f"  [{cat}] {count}개")

    elif agent == "product":
        conditions = [c["input"].get("condition", "?") for c in cases]
        for cond, count in sorted(Counter(conditions).items()):
            print(f"  {cond:<20} {count}개")

    elif agent == "response":
        categories = []
        for c in cases:
            comment = c.get("_comment", "")
            if comment:
                categories.append(comment.strip("─ ()"))
        print(f"  샘플 입력 상품: {cases[0]['input']['product']['product_name']}")

    print()
    print("샘플 케이스:")
    print(json.dumps(cases[0], ensure_ascii=False, indent=2))


# ── CLI ─────────────────────────────────────────────────────────────────

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="딸랑구 LangSmith Dataset 관리")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--agent", choices=AGENTS,
        help="업로드할 에이전트 (intent | product | response)",
    )
    group.add_argument("--all", action="store_true", help="전체 에이전트 업로드")
    parser.add_argument("--dry-run", action="store_true", help="업로드 없이 로컬 출력만")
    args = parser.parse_args()

    if args.dry_run:
        targets = AGENTS if args.all else [args.agent]
        for agent in targets:
            dry_run(agent)
    elif args.all:
        upload_all()
    else:
        upload_dataset(args.agent)


if __name__ == "__main__":
    main()
