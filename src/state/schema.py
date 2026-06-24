from typing import Annotated, Optional, TypedDict, Literal, Any
from langgraph.graph.message import add_messages

# ══════════════════════════════════════════════
# 공통 타입
# ══════════════════════════════════════════════

Stage = Literal[
    "idle",
    "searching",
    "product_confirming",
    "cart_shopping",
    "payment_processing",
    "payment_password_required",
    "completed",
    "failed",
]

Intent = Literal[
    "buy",
    "reorder",
    "confirm",
    "deny",
    "next",
    "refine",
    "compare_platforms",
    "quantity_change",
    "address_change",
    "option_select",
    "ask",
    "cancel",
    "unclear",
]

Condition = Literal[
    "최저가",
    "가성비",
    "빠른배송",
    "인기순",
    "무료배송",
    "리뷰좋은",
]

PendingActionType = Literal[
    "product_confirm",
    "product_select",
    "clarification",
    "payment_confirm",
    "address_confirm",
    "address_required",
    "price_change_confirm",
    "quantity_confirm",
    "continue_shopping",
    "what_to_buy",
    "no_more_products",
    "payment_method_confirm",
    "payment_password",
]

class PendingAction(TypedDict, total=False):
    type: PendingActionType
    payload: dict[str, Any]
    message: str
    expires_at: Optional[str]

# ══════════════════════════════════════════════
# 1. ShoppingState
# ══════════════════════════════════════════════

class ShoppingState(TypedDict):
    # ── 대화 ──
    messages: Annotated[list, add_messages]

    # ── 플로우 제어 ──
    stage: Stage
    intent: Optional[Intent]
    last_agent: Optional[str]
    error: Optional[str]

    # ── Intent Agent 출력 ──
    confidence: Optional[float]
    immediate_response: Optional[str]
    needs_clarification: bool
    clarification_reason: Optional[str]

    # ── 검색 조건 ──
    keywords: list[str]
    exclude_keywords: list[str]
    negative_constraints: list[str]
    quantity: Optional[int]
    condition: Optional[Condition]

    # ── 플랫폼 ──
    override_platform: Optional[str]
    target_platforms: list[str]
    tried_platforms: list[str]
    selected_platform: Optional[str]

    # ── 상품 탐색 ──
    search_results: list[dict[str, Any]]
    scored_products: list[dict[str, Any]]
    recommended_products: list[dict[str, Any]]
    current_product_index: int
    selected_product: Optional[dict[str, Any]]
    product_url: Optional[str]
    explanation: Optional[str]
    highlight_specs: list[str]

    # ── 확인/대기 액션 ──
    pending_action: Optional[PendingAction]

    # ── 백엔드 결과 ──
    cart: Optional[dict[str, Any]]
    order: Optional[dict[str, Any]]
    payment: Optional[dict[str, Any]]
    checkout_session: Optional[dict[str, Any]]

    # ── 세션 식별자 ──
    session_id: str
    conversation_id: Optional[int]
    user_id: str

    # ── 브라우저 세션 (mock) ──
    storage_state_path: Optional[str]
    cart_items: list[dict[str, Any]]

    # ── Memory Agent context ──
    recommendation_context: Optional[dict[str, Any]]
    reorder_resolution: Optional[dict[str, Any]]

    # ── Intent Agent 슬롯 ──
    current_option_value: Optional[str]
    address_text: Optional[str]
    tool_calls: Optional[list[dict[str, Any]]]
    tool_results: Optional[dict[str, Any]]
    conversation_summary: Optional[str]
    order_id: Optional[str]

# ══════════════════════════════════════════════
# 2. PaymentState
# ══════════════════════════════════════════════

PaymentStage = Literal[
    "idle",
    "validate_input",
    "open_product_page",
    "option_selecting",
    "option_confirming",
    "cart",
    "address_confirming",
    "payment_precheck",
    "payment_password_required",
    "processing",
    "success",
    "failed",
]

PaymentStatus = Literal[
    "pending",
    "pending_user_action",
    "processing",
    "success",
    "failed",
]

class PaymentState(TypedDict):
    user_id: str
    conversation_id: Optional[int]

    selected_product: dict[str, Any]
    product_url: str
    quantity: int
    selected_platform: Optional[str]

    available_options: list[dict[str, Any]]
    current_option_index: int
    current_option_key: Optional[str]
    current_option_value: Optional[str]
    selected_options: dict[str, Any]

    delivery_address: Optional[dict[str, Any]]
    address_confirmed: bool

    playwright_session: Optional[str]
    checkout_session_id: Optional[str]
    order_id: Optional[str]

    payment_stage: PaymentStage
    payment_status: PaymentStatus
    payment_step: Optional[str]
    payment_retry: int
    payment_error: Optional[str]

    pending_action: Optional[dict[str, Any]]

# ══════════════════════════════════════════════
# 3. MemoryState
# ══════════════════════════════════════════════

class MemoryState(TypedDict):
    user_id: str
    user_profile: dict[str, Any]
    purchase_history: list[dict[str, Any]]
    preference_memory: dict[str, Any]
    collective_context: Optional[dict[str, Any]]
    conversation_summary: Optional[str]

# ══════════════════════════════════════════════
# 4. Bridge Functions
# ══════════════════════════════════════════════

def bridge_memory_to_shopping(memory: MemoryState) -> dict:
    return {"last_agent": "memory_agent"}


def bridge_shopping_to_payment(
    state: ShoppingState,
    delivery_address: Optional[dict[str, Any]] = None,
) -> PaymentState:
    selected_product = state.get("selected_product") or {}
    product_url = (
        state.get("product_url")
        or selected_product.get("product_url")
        or selected_product.get("url")
        or ""
    )

    return {
        "user_id": state["user_id"],
        "conversation_id": state.get("conversation_id"),
        "selected_product": selected_product,
        "product_url": product_url,
        "quantity": state.get("quantity") or 1,
        "selected_platform": state.get("selected_platform"),
        "available_options": [],
        "current_option_index": 0,
        "current_option_key": None,
        "current_option_value": None,
        "selected_options": {},
        "delivery_address": delivery_address,
        "address_confirmed": False,
        "playwright_session": None,
        "checkout_session_id": None,
        "order_id": None,
        "payment_stage": "validate_input",
        "payment_status": "pending",
        "payment_step": None,
        "payment_retry": 0,
        "payment_error": None,
        "pending_action": None,
    }


def bridge_payment_to_shopping(payment: PaymentState) -> dict:
    if payment["payment_status"] == "success":
        return {
            "stage": "completed",
            "error": None,
            "last_agent": "payment_agent",
            "pending_action": payment.get("pending_action"),
        }

    if payment["payment_status"] == "failed":
        return {
            "stage": "failed",
            "error": payment.get("payment_error"),
            "last_agent": "payment_agent",
        }

    return {
        "stage": "payment_processing",
        "error": payment.get("payment_error"),
        "last_agent": "payment_agent",
        "pending_action": payment.get("pending_action"),
    }


def get_default_shopping_state(user_id: str, session_id: str) -> dict:
    return {
        "messages": [],
        "stage": "idle",
        "intent": None,
        "last_agent": None,
        "error": None,
        "confidence": None,
        "immediate_response": None,
        "needs_clarification": False,
        "clarification_reason": None,
        "keywords": [],
        "exclude_keywords": [],
        "negative_constraints": [],
        "quantity": None,
        "condition": None,
        "override_platform": None,
        "target_platforms": [],
        "tried_platforms": [],
        "selected_platform": None,
        "search_results": [],
        "scored_products": [],
        "recommended_products": [],
        "current_product_index": 0,
        "selected_product": None,
        "product_url": None,
        "explanation": None,
        "highlight_specs": [],
        "pending_action": None,
        "session_id": session_id,
        "conversation_id": None,
        "user_id": user_id,
        "recommendation_context": None,
        "reorder_resolution": None,
        "storage_state_path": None,
        "cart_items": [],
        "current_option_value": None,
        "address_text": None,
        "tool_calls": None,
        "tool_results": None,
        "conversation_summary": None,
        "order_id": None,
        "cart": None,
        "order": None,
        "payment": None,
        "checkout_session": None,
    }
