"""
End-to-End 통합 테스트 — 실제 LLM 호출 포함.

실행:
  py -3.14 tests/test_e2e.py                   # 전체 시나리오 실행
  py -3.14 tests/test_e2e.py --scenario buy     # 특정 시나리오만
  py -3.14 -m pytest tests/test_e2e.py -v -s   # pytest (로그 보려면 -s 필수)

시나리오:
  1. buy_flow    - 구매 → 수량 → 결제 전체 흐름 (6턴)
  2. reorder     - 재구매 (구매이력 있는 user_001)
  3. next_ask    - 다음 상품 요청 + QA
  4. deny_cancel - 거부 후 취소
"""
import sys
import uuid
import os
from pathlib import Path
from typing import NamedTuple

# Windows 콘솔 UTF-8 출력 설정
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 프로젝트 루트를 sys.path에 추가
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from src.graph.builder import build_graph
from src.state.schema import get_default_shopping_state
from src.utils.agent_logger import agent_logger


# ── 공통 헬퍼 ──────────────────────────────────────────────────────────

class TurnResult(NamedTuple):
    user_input: str
    response: str
    stage: str
    pending_type: str | None
    intent: str | None


def _get_response(graph, config: dict) -> tuple[str, dict]:
    current = graph.get_state(config)
    vals = current.values
    msgs = vals.get("messages", [])
    response = ""
    for msg in reversed(msgs):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            response = msg["content"]
            break
        if getattr(msg, "type", None) == "ai":
            response = msg.content
            break
    return response, vals


def _print_separator(title: str = "") -> None:
    line = "─" * 60
    if title:
        print(f"\n{line}\n  {title}\n{line}")
    else:
        print(line)


def _print_state_summary(vals: dict, turn_num: int) -> None:
    stage = vals.get("stage", "?")
    intent = vals.get("intent", "?")
    pending = (vals.get("pending_action") or {})
    pending_type = pending.get("type", "-")
    keywords = vals.get("keywords") or []
    quantity = vals.get("quantity")
    error = vals.get("error")
    last_agent = vals.get("last_agent", "-")
    cart_count = len(vals.get("cart_items") or [])
    order_id = vals.get("order_id")

    selected = vals.get("selected_product") or {}
    product_name = selected.get("product_name", "-")
    product_price = selected.get("price")

    print(f"  ┌─ STATE [Turn {turn_num}]")
    print(f"  │  stage={stage}  intent={intent}  last_agent={last_agent}")
    print(f"  │  keywords={keywords}  quantity={quantity}  condition={vals.get('condition')}")
    print(f"  │  pending={pending_type}  error={error or '-'}")
    if product_name != "-":
        price_str = f"{product_price:,}원" if product_price else "가격미정"
        print(f"  │  selected_product={product_name} ({price_str})")
    if cart_count:
        cart_total = sum(i.get("total", 0) for i in (vals.get("cart_items") or []))
        print(f"  │  cart={cart_count}개  cart_total={cart_total:,}원")
    if order_id:
        print(f"  │  order_id={order_id}")
    print(f"  └─")


def run_scenario(
    title: str,
    user_id: str,
    turns: list[tuple[str, str | None]],  # (user_input, expected_stage_or_None)
) -> list[TurnResult]:
    """
    단일 시나리오 실행.

    turns: [(user_input, expected_stage), ...]
      - expected_stage: None이면 단순 진행만 (assert 없음)
    """
    _print_separator(f"시나리오: {title}  (user={user_id})")

    thread_id = f"e2e-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}
    agent_logger.start_session(session_id=thread_id, console=True)

    graph = build_graph()
    initial_state = get_default_shopping_state(user_id, thread_id)
    initial_state["conversation_id"] = abs(hash(thread_id)) % 1_000_000
    initial_state["address_text"] = "서울 강남구 테헤란로 1"  # 기본 배송지
    graph.invoke(initial_state, config)

    results: list[TurnResult] = []
    for i, (user_input, expected_stage) in enumerate(turns, 1):
        print(f"\n{'='*60}")
        print(f"  [Turn {i}] 사용자: {user_input}")
        print(f"{'='*60}")

        agent_logger.new_turn(user_input)
        graph.update_state(config, {"messages": [{"role": "user", "content": user_input}]})

        try:
            graph.invoke(None, config)
        except Exception as e:
            print(f"  [그래프 오류] {e}")
            import traceback
            traceback.print_exc()
            break

        response, vals = _get_response(graph, config)
        stage = vals.get("stage", "?")
        pending_type = (vals.get("pending_action") or {}).get("type")
        intent = vals.get("intent")

        print(f"\n  딸랑구: {response}")
        _print_state_summary(vals, i)

        result = TurnResult(user_input, response, stage, pending_type, intent)
        results.append(result)

        if expected_stage is not None:
            assert stage == expected_stage, (
                f"[{title}] Turn {i} stage 불일치: 기대={expected_stage}, 실제={stage}"
            )
            print(f"  ✓ stage={stage} (예상과 일치)")

        if not graph.get_state(config).next or stage in ("completed", "failed"):
            print(f"\n  [시나리오 종료] stage={stage}")
            break

    return results


# ── 시나리오 1: 일반 구매 전체 흐름 ───────────────────────────────────

def test_buy_full_flow():
    """구매 → 수량 → 장바구니 담기 → 장바구니 확인 → 배송지 → 결제수단 → 비밀번호 → 완료."""
    results = run_scenario(
        title="일반 구매 전체 흐름",
        user_id="user_test",
        turns=[
            ("딸기 사고 싶어",     "product_confirming"),   # 1: 상품 추천
            ("응",                 None),                   # 2: confirm → 수량 없어서 재질문
            ("2개요",              None),                   # 3: 수량 → Step 0 담기 (cart_shopping)
            ("결제할게요",          None),                   # 4: Step 1 → cart_review (cart_shopping)
            ("응",                 "payment_processing"),   # 5: Step 1-5 → address_confirm
            ("응",                 "payment_processing"),   # 6: Step 2 → payment_method_confirm
            ("응",                 "payment_processing"),   # 7: Step 3 → payment_password
            ("1234",               "completed"),            # 8: Step 4 → 완료
        ],
    )
    assert any(r.stage == "completed" for r in results), "결제 완료까지 도달하지 못함"
    print("\n  [✓] 일반 구매 전체 흐름 통과")


# ── 시나리오 2: 재구매 흐름 ────────────────────────────────────────────

def test_reorder_flow():
    """user_001 구매이력 기반 재구매 (딸기 - 구매이력 존재)."""
    results = run_scenario(
        title="재구매 흐름",
        user_id="user_001",
        turns=[
            ("저번에 산 딸기 다시 사줘",  None),                  # 1: reorder → 이력 탐색
            ("응",                        None),                  # 2: confirm
            ("1개",                       None),                  # 3: 수량 → Step 0 담기
            ("결제할게요",                 None),                  # 4: Step 1 → cart_review
            ("응",                        "payment_processing"),  # 5: Step 1-5 → address_confirm
            ("응",                        "payment_processing"),  # 6: Step 2 → payment_method_confirm
            ("응",                        "payment_processing"),  # 7: Step 3 → payment_password
            ("1234",                      "completed"),           # 8: Step 4 → 완료
        ],
    )
    assert len(results) >= 1, "한 턴도 실행되지 않음"
    print("\n  [✓] 재구매 흐름 통과")


# ── 시나리오 3: 다음 상품 + QA ─────────────────────────────────────────

def test_next_and_ask():
    """상품 추천 → 다음 거 보여줘 → 이거 유기농이야? (QA)"""
    results = run_scenario(
        title="다음 상품 요청 + QA",
        user_id="user_test",
        turns=[
            ("딸기 사고 싶어",      "product_confirming"),  # 1: 첫 추천
            ("다른 거 보여줘",      "product_confirming"),  # 2: next → 다음 후보
            ("이거 유기농이야?",    "product_confirming"),  # 3: ask → QA 답변
            ("응",                  None),                  # 4: confirm
        ],
    )
    assert len(results) >= 3, "QA까지 도달하지 못함"
    qa_result = results[2]
    assert qa_result.intent == "ask", f"Turn 3 intent 기대=ask, 실제={qa_result.intent}"
    print("\n  [✓] 다음 상품 + QA 통과")


# ── 시나리오 4: 취소 흐름 ──────────────────────────────────────────────

def test_cancel_mid_flow():
    """상품 추천 중 취소."""
    results = run_scenario(
        title="취소 흐름",
        user_id="user_test",
        turns=[
            ("계란 사고 싶어",      "product_confirming"),  # 1: 추천
            ("그만할게요",           "idle"),               # 2: cancel → idle
        ],
    )
    last = results[-1]
    assert last.stage == "idle", f"취소 후 stage 기대=idle, 실제={last.stage}"
    print("\n  [✓] 취소 흐름 통과")


# ── 시나리오 5: 조건 검색 ──────────────────────────────────────────────

def test_condition_search():
    """최저가 조건 검색 + 가성비로 refine."""
    results = run_scenario(
        title="조건 검색",
        user_id="user_test",
        turns=[
            ("우유 최저가로 찾아줘",        "product_confirming"),  # 1: buy + 최저가
            ("가성비 좋은 걸로 다시 찾아줘", "product_confirming"),  # 2: refine
        ],
    )
    assert results[0].stage == "product_confirming", "첫 검색 실패"
    print("\n  [✓] 조건 검색 통과")


# ── 실행 진입점 ────────────────────────────────────────────────────────

_SCENARIOS = {
    "buy":    test_buy_full_flow,
    "reorder": test_reorder_flow,
    "next_ask": test_next_and_ask,
    "cancel": test_cancel_mid_flow,
    "condition": test_condition_search,
}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="딸랑구 E2E 테스트")
    parser.add_argument("--scenario", choices=list(_SCENARIOS.keys()), default=None,
                        help="실행할 시나리오 (없으면 전체 실행)")
    args = parser.parse_args()

    targets = {args.scenario: _SCENARIOS[args.scenario]} if args.scenario else _SCENARIOS

    print(f"\n{'#'*60}")
    print(f"  딸랑구 E2E 테스트  ({len(targets)}개 시나리오)")
    print(f"{'#'*60}")

    passed, failed = [], []
    for name, fn in targets.items():
        try:
            fn()
            passed.append(name)
        except AssertionError as e:
            print(f"\n  [✗] {name} 실패: {e}")
            failed.append(name)
        except Exception as e:
            print(f"\n  [✗] {name} 예외: {e}")
            import traceback
            traceback.print_exc()
            failed.append(name)

    print(f"\n{'#'*60}")
    print(f"  결과: {len(passed)}통과 / {len(failed)}실패")
    if passed:
        print(f"  ✓ 통과: {', '.join(passed)}")
    if failed:
        print(f"  ✗ 실패: {', '.join(failed)}")
    print(f"{'#'*60}\n")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
