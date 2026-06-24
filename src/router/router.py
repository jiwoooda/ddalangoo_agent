"""
Router — intent + stage 기반으로 어떤 Crew를 실행할지 결정.
LangGraph의 조건부 엣지를 Python 함수로 대체.
"""
from src.session.state import SessionState

CrewType = str  # "buy" | "reorder" | "payment" | "respond" | "cancel" | "end"


def route(state: SessionState, intent: str) -> CrewType:
    stage = state.stage
    pending_type = state.pending_type()
    confidence = state.confidence
    needs_clarification = state.needs_clarification

    if needs_clarification or confidence < 0.5 or intent == "unclear":
        return "respond"

    if intent == "cancel":
        return "cancel"

    if stage == "payment_processing":
        return "payment"

    if stage == "cart_shopping":
        if intent in ("buy", "refine", "compare_platforms"):
            # "결제할게요"는 buy로 분류되기도 함 — continue_shopping 대기 중이면 결제로 진행
            if pending_type == "continue_shopping" and intent == "buy":
                return "payment"
            return "buy"
        if intent == "reorder":
            return "reorder"
        if intent == "confirm":
            return "payment"
        if intent in ("deny", "next"):
            # 장바구니 유지하고 새 상품 탐색 시작 → 즉시 응답으로 안내
            return "respond"
        return "respond"

    if stage == "product_confirming":
        if pending_type == "quantity_confirm":
            if intent in ("confirm", "quantity_change") and state.quantity:
                return "payment"
            return "respond"

        if pending_type == "product_select":
            if intent in ("confirm", "option_select"):
                return "reorder"
            return "respond"

        if pending_type == "price_change_confirm":
            if intent in ("confirm", "deny", "cancel", "next"):
                return "payment"
            return "respond"

        # product_confirm: 상품 선택 확인 또는 수량 지정 → 결제로 진행
        if pending_type == "product_confirm":
            if intent in ("confirm", "quantity_change"):
                if not state.quantity:
                    return "respond"  # 수량 먼저 확인
                return "payment"
            # "2개 주문할게요" 등 buy 의도에 수량이 포함된 경우도 결제로 진행
            if intent == "buy" and state.quantity:
                return "payment"

        if intent == "confirm":
            if not state.quantity:
                return "respond"
            return "payment"

        if intent in ("buy", "refine", "compare_platforms"):
            return "buy"
        if intent == "reorder":
            return "reorder"
        if intent in ("deny", "next"):
            return "buy"
        if intent == "ask":
            return "qa"
        return "respond"

    if stage == "searching":
        if intent in ("refine", "compare_platforms"):
            return "buy"
        if intent == "ask":
            return "qa"
        return "respond"

    # idle 또는 기타
    routing_map = {
        "buy": "buy",
        "reorder": "reorder",
        "compare_platforms": "buy",
        "refine": "buy",
        "ask": "qa",
        "next": "buy",
    }
    return routing_map.get(intent, "respond")


def get_respond_message(state: SessionState, intent: str) -> str:
    """Crew 없이 즉시 응답할 때 메시지 생성."""
    pending_msg = (state.pending_action or {}).get("message", "")

    if state.needs_clarification or intent == "unclear":
        return "다시 말씀해 주세요."

    if intent == "cancel":
        return "쇼핑을 종료했어요. 또 필요하시면 말씀해 주세요."

    # 장바구니에 담긴 상태에서 다른 상품 탐색
    if state.stage == "cart_shopping" and intent in ("deny", "next"):
        return "네, 또 필요하신 게 있으세요?"

    # 상품 확인 후 수량 미지정 → 수량 먼저 물어보기 (pending_msg보다 우선)
    if state.stage == "product_confirming" and not state.quantity:
        keywords = state.keywords[0] if state.keywords else "상품"
        return f"{keywords} 몇 개 드릴까요?"

    if pending_msg:
        return pending_msg

    return "무엇을 도와드릴까요?"
