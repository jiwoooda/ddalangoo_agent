"""공통 Graph 노드: wait_for_input, respond, cancel 등."""
from src.state.schema import ShoppingState
from src.utils.agent_logger import agent_logger


def wait_for_input_node(state: ShoppingState) -> dict:
    """
    사용자 입력 대기 노드.
    interrupt_before=["wait_for_input"]로 이 지점에서 항상 멈춘다.
    """
    return {}


def respond_node(state: ShoppingState) -> dict:
    """사용자에게 보낼 메시지 생성."""
    stage = state.get("stage", "idle")
    immediate = state.get("immediate_response")
    explanation = state.get("explanation")
    pending_action = state.get("pending_action") or {}

    if state.get("needs_clarification"):
        msg = immediate or state.get("clarification_reason") or "조금 더 자세히 말씀해 주세요."

    elif pending_action.get("message"):
        msg = pending_action["message"]

    elif stage == "product_confirming":
        msg = explanation or "이 상품으로 주문할까요?"

    elif stage == "payment_processing":
        msg = immediate or "결제를 계속 진행할까요?"

    elif stage == "completed":
        msg = "주문이 완료되었습니다."

    elif stage == "failed":
        msg = state.get("error") or "처리 중 문제가 발생했습니다."

    else:
        msg = immediate or "무엇을 도와드릴까요?"

    agent_logger.log_respond(msg, stage, pending_action)
    return {"messages": [{"role": "assistant", "content": msg}]}


def quantity_check_node(state: ShoppingState) -> dict:
    """수량이 없을 때 수량 질문 설정."""
    keywords = state.get("keywords") or []
    short_name = keywords[0] if keywords else (state.get("selected_product") or {}).get("product_name", "상품")
    return {
        "pending_action": {
            "type": "quantity_confirm",
            "message": f"{short_name} 몇 개 사실래요?",
        }
    }


def ask_what_to_buy_node(state: ShoppingState) -> dict:
    """장바구니 후 추가 쇼핑 의사 표현 시 새 상품 입력 대기."""
    return {
        "pending_action": {
            "type": "what_to_buy",
            "message": "무엇을 구매하실까요?",
        },
        "stage": "cart_shopping",
        "keywords": [],
        "search_results": [],
        "selected_product": None,
        "reorder_resolution": None,
        "error": None,
        "quantity": None,
        "product_url": None,
        "current_product_index": 0,
        "explanation": None,
        "highlight_specs": [],
        "scored_products": [],
        "recommended_products": [],
    }


def cancel_node(state: ShoppingState) -> dict:
    """모든 stage에서의 cancel 처리. state 완전 리셋."""
    cart_items = state.get("cart_items") or []
    if cart_items:
        msg = "알겠어요~ 처음으로 돌아갈게요! 장바구니에 담아둔 건 그대로 있을 거에요 :)"
    else:
        msg = "알겠어요~ 필요하면 언제든 말씀해주세요!"

    return {
        "stage": "idle",
        "intent": None,
        "error": None,
        "pending_action": {"type": "payment_confirm", "message": msg},
        "last_agent": "cancel",
        "keywords": [],
        "search_results": [],
        "scored_products": [],
        "recommended_products": [],
        "selected_product": None,
        "product_url": None,
        "explanation": None,
        "highlight_specs": [],
        "current_product_index": 0,
        "quantity": None,
        "reorder_resolution": None,
    }
