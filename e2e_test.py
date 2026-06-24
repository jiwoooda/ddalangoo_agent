"""
E2E 테스트 — 딸랑구 CrewAI 전체 흐름 검증.

시나리오:
  A. 구매 흐름: "딸기 사줘" → 상품 확인 → 수량 입력 → 결제 5단계 → 완료
  B. 재구매 흐름: "저번에 산 계란 다시 줘" → 이력 조회 → 확인 → 결제

실행:
  python e2e_test.py
"""
import sys
import os
import json
import time
import uuid
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, ".")
from dotenv import load_dotenv
load_dotenv()

from src.session.state import SessionState
from src.crews.intent_crew import run_intent_crew
from src.crews.buy_crew import run_buy_crew
from src.crews.reorder_crew import run_reorder_crew
from src.crews.payment_crew import run_payment_crew
from src.router.router import route, get_respond_message
from main import _apply_intent, _apply_crew_result

# ── 로그 설정 ────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
LOG_FILE = f"logs/e2e_{time.strftime('%Y%m%d_%H%M%S')}.log"

def log(msg: str):
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# ── 1턴 실행 헬퍼 ────────────────────────────────────────────────────────
def run_turn(state: SessionState, user_input: str, turn: int) -> str:
    sep = "─" * 60
    log(f"\n{sep}")
    log(f"[TURN {turn}] 사용자: {user_input}")
    log(sep)

    state.add_user_message(user_input)
    state.error = None

    # IntentCrew
    t0 = time.time()
    intent_result = run_intent_crew(state.to_crew_inputs())
    _apply_intent(state, intent_result)
    log(f"  [Intent] {time.time()-t0:.1f}s | intent={state.intent}  keywords={state.keywords}  qty={state.quantity}  confidence={state.confidence:.2f}")

    # Router
    crew_type = route(state, state.intent)
    log(f"  [Router] crew={crew_type}  stage={state.stage}  pending={state.pending_type()}")

    # Crew 실행
    result = {}
    t1 = time.time()

    if crew_type == "buy":
        result = run_buy_crew(state.to_crew_inputs())
        log(f"  [BuyCrew] {time.time()-t1:.1f}s | selected={result.get('selected_product', {}).get('product_name')}  ranked={len(result.get('ranked_products', []))}개")

    elif crew_type == "reorder":
        result = run_reorder_crew(state.to_crew_inputs())
        log(f"  [ReorderCrew] {time.time()-t1:.1f}s | resolution={result.get('resolution_type')}  product={result.get('selected_product', {}).get('product_name')}")
        if result.get("error") == "reorder_no_match":
            log("  [Fallback] 재구매 이력 없음 → BuyCrew 전환")
            result = run_buy_crew(state.to_crew_inputs())

    elif crew_type == "payment":
        result = run_payment_crew(state.to_crew_inputs())
        log(f"  [PaymentCrew] {time.time()-t1:.1f}s | stage={result.get('stage')}  pending={result.get('pending_action', {}).get('type')}")

    elif crew_type == "cancel":
        state.stage = "idle"
        state.keywords = []
        state.quantity = None
        state.selected_product = {}
        state.recommended_products = []
        state.pending_action = {}
        result = {"explanation": "쇼핑을 종료했어요. 또 필요하시면 말씀해 주세요.", "stage": "idle"}
        log(f"  [Cancel] 초기화 완료")

    elif crew_type == "qa":
        from src.agents.response_composer import build_response_composer
        from crewai import Task, Crew, Process
        composer = build_response_composer()
        qa_task = Task(
            description=(
                f"사용자 질문: \"{user_input}\"\n"
                f"현재 추천 상품: {state.selected_product}\n\n"
                "위 상품 정보를 바탕으로 질문에 1~2문장으로 답변하라. "
                "정보가 없으면 '확인되는 정보가 없어요'라고 말한다. "
                "존댓말, 구어체. 텍스트만 반환."
            ),
            expected_output="1~2문장 QA 답변",
            agent=composer,
        )
        qa_crew = Crew(agents=[composer], tasks=[qa_task], process=Process.sequential, verbose=False)
        qa_result = qa_crew.kickoff(inputs=state.to_crew_inputs())
        result = {"stage": state.stage, "explanation": qa_result.raw.strip(), "pending_action": state.pending_action}
        log(f"  [QA] {time.time()-t1:.1f}s")

    else:  # respond
        respond_msg = get_respond_message(state, state.intent)
        result = {"explanation": respond_msg, "stage": state.stage}
        log(f"  [Respond] 즉시 응답")

    _apply_crew_result(state, result)
    response = state.response or "네, 알겠습니다."
    state.add_assistant_message(response)

    log(f"  [State] stage={state.stage}  pending={state.pending_type()}")
    log(f"  [딸랑구] {response}")

    return response


# ── 시나리오 A: 구매 전체 흐름 ─────────────────────────────────────────────
def scenario_buy():
    log("\n" + "=" * 60)
    log("  시나리오 A: 구매 흐름 (딸기 → 결제 완료)")
    log("=" * 60)

    state = SessionState(
        user_id="user_001",
        thread_id=str(uuid.uuid4())[:8],
        address_text="서울 강남구 테헤란로 123",
        conversation_id=1001,
    )

    script = [
        "딸기 사줘",            # buy → BuyCrew → product_confirming
        "2개 주문할게요",       # confirm + quantity → PaymentCrew step0 (장바구니 담기)
        "결제할게요",           # confirm → PaymentCrew step1 (총액 확인)
        "네이버로 할게요",      # confirm → PaymentCrew step2 (배송지)
        "맞아요",              # confirm → PaymentCrew step3 (비밀번호)
        "1234",               # password → PaymentCrew step4 (주문 완료)
    ]

    passed = []
    for i, user_input in enumerate(script, 1):
        try:
            resp = run_turn(state, user_input, i)
            passed.append(True)
            if state.stage in ("completed", "failed"):
                log(f"\n  → 세션 종료: stage={state.stage}")
                break
        except Exception as e:
            import traceback
            log(f"\n  [ERROR] Turn {i} 실패: {e}")
            log(traceback.format_exc())
            passed.append(False)
            break

    success = all(passed) and state.stage == "completed"
    log(f"\n  결과: {'PASS ✓' if success else 'FAIL ✗'}  (turns={len(passed)}, final_stage={state.stage})")
    return success


# ── 시나리오 B: 재구매 흐름 ─────────────────────────────────────────────────
def scenario_reorder():
    log("\n" + "=" * 60)
    log("  시나리오 B: 재구매 흐름 (계란 재구매 → 결제)")
    log("=" * 60)

    state = SessionState(
        user_id="user_001",
        thread_id=str(uuid.uuid4())[:8],
        address_text="서울 강남구 테헤란로 123",
        conversation_id=1002,
    )

    script = [
        "저번에 산 계란 다시 줘",   # reorder → ReorderCrew → product_confirming
        "응 그거",                  # confirm → respond (몇 개 드릴까요?)
        "1개요",                   # quantity_change → PaymentCrew step0
        "결제할게요",               # confirm → PaymentCrew step1
        "네",                      # confirm → PaymentCrew step2
        "맞아요",                  # confirm → PaymentCrew step3
        "1234",                   # password → PaymentCrew step4
    ]

    passed = []
    for i, user_input in enumerate(script, 1):
        try:
            resp = run_turn(state, user_input, i)
            passed.append(True)
            if state.stage in ("completed", "failed"):
                log(f"\n  → 세션 종료: stage={state.stage}")
                break
        except Exception as e:
            import traceback
            log(f"\n  [ERROR] Turn {i} 실패: {e}")
            log(traceback.format_exc())
            passed.append(False)
            break

    success = all(passed) and state.stage == "completed"
    log(f"\n  결과: {'PASS ✓' if success else 'FAIL ✗'}  (turns={len(passed)}, final_stage={state.stage})")
    return success


# ── 시나리오 C: 취소 흐름 ────────────────────────────────────────────────────
def scenario_cancel():
    log("\n" + "=" * 60)
    log("  시나리오 C: 취소 흐름 (구매 중 취소)")
    log("=" * 60)

    state = SessionState(
        user_id="user_test",
        thread_id=str(uuid.uuid4())[:8],
        conversation_id=1003,
    )

    script = [
        "우유 사줘",       # buy → BuyCrew → product_confirming
        "그만할게요",      # cancel → idle
    ]

    passed = []
    for i, user_input in enumerate(script, 1):
        try:
            resp = run_turn(state, user_input, i)
            passed.append(True)
            if state.stage == "idle" and i == 2:
                break
        except Exception as e:
            import traceback
            log(f"\n  [ERROR] Turn {i} 실패: {e}")
            log(traceback.format_exc())
            passed.append(False)
            break

    success = all(passed) and state.stage == "idle"
    log(f"\n  결과: {'PASS ✓' if success else 'FAIL ✗'}  (turns={len(passed)}, final_stage={state.stage})")
    return success


# ── 메인 ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log(f"{'='*60}")
    log(f"  딸랑구 CrewAI E2E 테스트")
    log(f"  시작: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  로그: {LOG_FILE}")
    log(f"{'='*60}")

    results = {}

    results["A_buy"] = scenario_buy()
    results["B_reorder"] = scenario_reorder()
    results["C_cancel"] = scenario_cancel()

    log(f"\n{'='*60}")
    log("  최종 결과 요약")
    log(f"{'='*60}")
    all_pass = True
    for name, ok in results.items():
        status = "PASS ✓" if ok else "FAIL ✗"
        log(f"  {name:<20} {status}")
        if not ok:
            all_pass = False
    log(f"{'='*60}")
    log(f"  전체: {'ALL PASS ✓' if all_pass else 'SOME FAILED ✗'}")
    log(f"  종료: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  로그 파일: {LOG_FILE}")

    sys.exit(0 if all_pass else 1)
