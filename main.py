"""
딸랑구 쇼핑 어시스턴트 — CrewAI 구현 Entry Point.

LangGraph의 interrupt_before를 while 루프 + input()으로 대체.
매 턴: IntentCrew → Router → 해당 Crew 실행.

사용법:
    python main.py
    python main.py --user user_001
    python main.py --demo
"""
import argparse
import sys
import os
import uuid

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

# API 키 경고
_missing = []
if not os.getenv("ANTHROPIC_API_KEY"):
    _missing.append("ANTHROPIC_API_KEY")
if not os.getenv("OPENAI_API_KEY"):
    _missing.append("OPENAI_API_KEY")
if _missing:
    print(f"[경고] 환경변수 미설정: {', '.join(_missing)}")

from src.session.state import SessionState
from src.crews.intent_crew import run_intent_crew
from src.crews.buy_crew import run_buy_crew
from src.crews.reorder_crew import run_reorder_crew
from src.crews.payment_crew import run_payment_crew
from src.router.router import route, get_respond_message


def _apply_intent(state: SessionState, intent_result: dict) -> None:
    """IntentCrew 결과를 SessionState에 반영."""
    state.intent = intent_result.get("intent", "unclear")
    state.confidence = intent_result.get("confidence", 0.0)
    state.needs_clarification = intent_result.get("needs_clarification", False)
    state.condition = intent_result.get("condition") or state.condition

    # 검색 의도일 때만 keywords 교체
    _search_intents = {"buy", "reorder", "refine", "compare_platforms"}
    new_keywords = intent_result.get("keywords") or []
    if state.intent in _search_intents and new_keywords:
        state.keywords = new_keywords
    # 비검색 의도는 기존 keywords 유지

    # 수량: 검색 의도 외에는 기존 수량 유지
    new_qty = intent_result.get("quantity")
    if new_qty is not None:
        state.quantity = new_qty
    elif state.intent not in _search_intents:
        pass  # 기존 수량 유지


def _apply_crew_result(state: SessionState, result: dict) -> None:
    """Crew 실행 결과를 SessionState에 반영."""
    if "stage" in result and result["stage"] is not None:
        state.stage = result["stage"]
    if "selected_product" in result and result["selected_product"]:
        state.selected_product = result["selected_product"]
    if "ranked_products" in result and result["ranked_products"]:
        state.recommended_products = result["ranked_products"]
        state.current_product_index = 0
    if "pending_action" in result and result["pending_action"]:
        state.pending_action = result["pending_action"]
    if "order_id" in result and result["order_id"]:
        state.order_id = result["order_id"]
    if "cart_items" in result and result["cart_items"] is not None:
        state.cart_items = result["cart_items"]
    if "error" in result:
        state.error = result["error"]

    # cart_shopping 진입 시 cart_items가 비어 있으면 selected_product로 보정
    # → to_crew_inputs의 cart_count가 0으로 남아 PaymentCrew가 step0을 반복하는 것을 방지
    if state.stage == "cart_shopping" and not state.cart_items and state.selected_product:
        qty = state.quantity or 1
        price = state.selected_product.get("price", 0)
        state.cart_items = [{**state.selected_product, "quantity": qty, "total": price * qty}]

    # 응답 메시지 결정
    explanation = result.get("explanation", "")
    pending_msg = (result.get("pending_action") or {}).get("message", "")
    state.response = explanation or pending_msg or ""


def _spinner(message: str):
    """진행 중 메시지를 한 줄로 표시하는 컨텍스트 매니저."""
    import threading

    stop_event = threading.Event()

    def spin():
        frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        i = 0
        while not stop_event.is_set():
            print(f"\r  {frames[i % len(frames)]} {message}", end="", flush=True)
            stop_event.wait(0.1)
            i += 1
        print(f"\r  \033[2m{message}\033[0m  ", end="\r", flush=True)

    t = threading.Thread(target=spin, daemon=True)
    t.start()

    class _Ctx:
        def __enter__(self):
            return self
        def __exit__(self, *_):
            stop_event.set()
            t.join()
            print("\r" + " " * 60 + "\r", end="", flush=True)

    return _Ctx()


def run_session(user_id: str = "user_test") -> None:
    thread_id = str(uuid.uuid4())[:8]
    state = SessionState(
        user_id=user_id,
        thread_id=thread_id,
        address_text="서울 강남구 테헤란로 1",
        conversation_id=abs(hash(thread_id)) % 1_000_000,
    )

    print(f"\n{'━'*50}")
    print(f"  딸랑구 쇼핑 어시스턴트")
    print(f"  user: {user_id}  |  종료: exit")
    print(f"{'━'*50}\n")
    print("딸랑구: 안녕하세요! 무엇을 도와드릴까요?\n")

    while True:
        try:
            user_input = input("나: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[종료]")
            break

        if user_input.lower() in ("exit", "quit", "종료", "q"):
            print("딸랑구: 또 필요하시면 불러주세요!")
            break
        if not user_input:
            continue

        state.add_user_message(user_input)
        state.error = None

        # 1. 의도 분석
        with _spinner("생각 중..."):
            intent_result = run_intent_crew(state.to_crew_inputs())
        _apply_intent(state, intent_result)
        intent = state.intent

        # 2. 라우터
        crew_type = route(state, intent)

        # 3. Crew 실행
        result = {}

        if crew_type == "buy":
            with _spinner("상품 검색 중..."):
                result = run_buy_crew(state.to_crew_inputs())

        elif crew_type == "reorder":
            with _spinner("구매 이력 확인 중..."):
                result = run_reorder_crew(state.to_crew_inputs())
            if result.get("error") == "reorder_no_match":
                with _spinner("새 상품 검색 중..."):
                    result = run_buy_crew(state.to_crew_inputs())

        elif crew_type == "payment":
            with _spinner("결제 처리 중..."):
                result = run_payment_crew(state.to_crew_inputs())

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
            with _spinner("답변 생성 중..."):
                qa_result = qa_crew.kickoff(inputs=state.to_crew_inputs())
            result = {
                "stage": state.stage,
                "explanation": qa_result.raw.strip(),
                "pending_action": state.pending_action,
            }

        elif crew_type == "cancel":
            state.stage = "idle"
            state.keywords = []
            state.quantity = None
            state.selected_product = {}
            state.recommended_products = []
            state.pending_action = {}
            state.error = None
            result = {"explanation": "쇼핑을 종료했어요. 또 필요하시면 말씀해 주세요.", "stage": "idle"}

        else:  # respond
            result = {"explanation": get_respond_message(state, intent), "stage": state.stage}

        # 4. 상태 반영 및 출력
        _apply_crew_result(state, result)

        # cart_shopping에서 다른 상품 탐색 의사 표현 후 pending 초기화
        # → 다음 buy 인텐트가 "결제"로 잘못 라우팅되는 것을 방지
        if crew_type == "respond" and state.stage == "cart_shopping" and intent in ("deny", "next"):
            state.pending_action = {}

        response = state.response or "네, 알겠습니다."
        state.add_assistant_message(response)

        print(f"딸랑구: {response}\n")

        if state.stage in ("completed", "failed"):
            print("─" * 50)
            print("  주문이 완료됐어요. 감사합니다!")
            print("─" * 50)
            break


def run_demo() -> None:
    """API 키 없이 라우터 + 도구 동작만 검증."""
    print("\n=== Demo Mode (CrewAI 구조 검증) ===\n")

    from src.session.state import SessionState
    from src.router.router import route

    state = SessionState(user_id="user_001", thread_id="demo")

    cases = [
        ("buy",     "idle"),
        ("reorder", "idle"),
        ("confirm", "product_confirming"),
        ("cancel",  "payment_processing"),
        ("ask",     "product_confirming"),
    ]

    print("─── Router 테스트 ───")
    for intent, stage in cases:
        state.stage = stage
        state.intent = intent
        state.confidence = 0.95
        state.needs_clarification = False
        result = route(state, intent)
        print(f"  intent={intent:<20} stage={stage:<22} → {result}")

    print("\n─── Tool 테스트 ───")
    from src.tools.search_tool import SearchProductsTool
    tool = SearchProductsTool()
    result = tool._run(query="딸기", platforms="naver,coupang,kurly")
    import json
    data = json.loads(result)
    products = data.get("products", [])
    print(f"  '딸기' 검색: {len(products)}건")
    for p in products[:3]:
        print(f"    - {p['product_name']} {p['price']:,}원 ({p['platform']})")

    from src.tools.history_tool import PurchaseHistoryTool
    ht = PurchaseHistoryTool()
    h_result = json.loads(ht._run(user_id="user_001", keywords="딸기"))
    print(f"\n  user_001 딸기 구매이력: {h_result['filtered_count']}건")

    print("\n[Demo 완료]")


def main() -> None:
    parser = argparse.ArgumentParser(description="딸랑구 쇼핑 어시스턴트 — CrewAI")
    parser.add_argument("--user", default="user_test", help="사용자 ID")
    parser.add_argument("--demo", action="store_true", help="LLM 없이 구조 검증")
    args = parser.parse_args()

    if args.demo or (not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY")):
        run_demo()
    else:
        run_session(user_id=args.user)


if __name__ == "__main__":
    main()
