"""
PaymentCrew — 결제 5단계 처리.
pending_type에 따라 PaymentProcessor가 해당 단계를 실행.
"""
import json
from crewai import Task, Crew, Process
from src.agents.payment_processor import build_payment_processor
from src.shared import mock_get_cart


_PAYMENT_TASK = """
현재 결제 단계(pending_type)에 따라 정확히 해당 작업을 수행하라.

현재 상태:
- pending_type: {pending_type}
- 선택 상품: {selected_product_name} ({selected_product_price}원)
- 수량: {quantity}
- 사용자 ID: {user_id}
- 장바구니 수량: {cart_count}개 (총액: {cart_total}원)
- 배송지: {address_text}
- 선택 상품 JSON: {selected_product_json}
- conversation_id: 0

중요: pending_type 값에 따라 정확히 해당 단계만 실행하라. 다른 단계를 실행하지 말 것.

단계별 처리:

[pending_type = "product_confirm" 또는 "quantity_confirm"] → Step 0: 장바구니 담기
  1. add_to_cart 도구 호출: product_json=선택상품JSON, quantity={quantity}, user_id={user_id}
  2. 반환 JSON:
  {{"step": 0, "stage": "cart_shopping", "pending_type": "continue_shopping",
    "message": "<상품명> {quantity}개 담았어요! 결제할까요, 다른 것도 보실래요?",
    "cart_count": <장바구니수>, "cart_total": <총액>}}

[pending_type = null 또는 "continue_shopping"] → Step 1: 총액 + 결제수단 확인
  1. get_cart 도구로 현재 장바구니 확인
  2. 반환 JSON:
  {{"step": 1, "stage": "payment_processing", "pending_type": "payment_method_confirm",
    "message": "<상품 요약>, 총 <총액>원이에요. 네이버로 결제할까요?"}}

[pending_type = "payment_method_confirm"] → Step 2: 배송지 확인
  1. get_address 도구로 배송지 확인
  2. 반환 JSON:
  {{"step": 2, "stage": "payment_processing", "pending_type": "address_confirm",
    "message": "<배송지 앞 3단어>...로 보낼게요. 맞으시죠?"}}

[pending_type = "address_confirm"] → Step 3: 비밀번호 요청
  반환 JSON:
  {{"step": 3, "stage": "payment_processing", "pending_type": "payment_password",
    "message": "비밀번호 입력해주세요!"}}

[pending_type = "payment_password"] → Step 4: 주문 실행
  1. place_order 도구 호출: user_id={user_id}, address_text={address_text}
  2. update_preference 도구 호출: user_id={user_id}, product_json=선택상품JSON
  3. 반환 JSON:
  {{"step": 4, "stage": "completed", "pending_type": "payment_confirm",
    "order_id": "<주문ID>", "message": "완료! 주문이 접수됐어요."}}

JSON만 반환 (설명 없이).
"""


def run_payment_crew(state_inputs: dict) -> dict:
    """
    PaymentCrew 실행.
    반환: stage, pending_action, order_id 등 결제 단계 결과
    """
    processor = build_payment_processor()

    task = Task(
        description=_PAYMENT_TASK,
        expected_output="JSON 형식의 결제 단계 처리 결과",
        agent=processor,
    )

    crew = Crew(
        agents=[processor],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff(inputs=state_inputs)
    raw = result.raw.strip()

    try:
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw.strip())

        stage = parsed.get("stage", "payment_processing")
        pending_type = parsed.get("pending_type", "")
        message = parsed.get("message", "")
        user_id = state_inputs.get("user_id", "")

        # 실제 mock 카트를 읽어 state.cart_items와 동기화
        actual_cart = [] if stage == "completed" else mock_get_cart(user_id)

        return {
            "stage": stage,
            "order_id": parsed.get("order_id"),
            "pending_action": {"type": pending_type, "message": message},
            "explanation": message,
            "error": None,
            "cart_items": actual_cart,
        }
    except Exception:
        return {
            "stage": "payment_processing",
            "pending_action": {
                "type": "payment_method_confirm",
                "message": "결제를 진행할게요. 네이버로 결제할까요?",
            },
            "explanation": "결제를 진행할게요. 네이버로 결제할까요?",
            "error": None,
        }
