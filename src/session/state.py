"""
SessionState — LangGraph의 MemorySaver + ShoppingState 대체.
CrewAI는 내장 상태 머신이 없으므로 Python dataclass로 직접 관리.
"""
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class SessionState:
    user_id: str
    thread_id: str

    # 대화 흐름 상태
    stage: str = "idle"
    intent: Optional[str] = None
    keywords: list[str] = field(default_factory=list)
    exclude_keywords: list[str] = field(default_factory=list)
    quantity: Optional[int] = None
    condition: Optional[str] = None
    confidence: float = 0.95
    needs_clarification: bool = False

    # 상품 상태
    selected_product: dict[str, Any] = field(default_factory=dict)
    recommended_products: list[dict] = field(default_factory=list)
    current_product_index: int = 0
    recommendation_context: dict[str, Any] = field(default_factory=dict)

    # 결제 상태
    cart_items: list[dict] = field(default_factory=list)
    pending_action: dict[str, Any] = field(default_factory=dict)
    order_id: Optional[str] = None
    address_text: Optional[str] = None
    conversation_id: Optional[int] = None

    # 메시지 이력
    messages: list[dict] = field(default_factory=list)

    # 에러 및 응답
    error: Optional[str] = None
    response: str = ""

    def pending_type(self) -> Optional[str]:
        return self.pending_action.get("type") if self.pending_action else None

    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})

    def last_user_input(self) -> str:
        for msg in reversed(self.messages):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return ""

    def to_crew_inputs(self) -> dict[str, Any]:
        """crew.kickoff(inputs=...) 에 전달할 딕셔너리."""
        product_name = (self.selected_product or {}).get("product_name", "없음")
        product_price = (self.selected_product or {}).get("price", 0)
        cart_total = sum(i.get("total", 0) for i in self.cart_items)
        # 총 수량(단위)으로 cart_count 계산 — line item 수가 아닌 실제 수량 합계
        cart_count = sum(i.get("quantity", 1) for i in self.cart_items)

        return {
            "user_input": self.last_user_input(),
            "user_id": self.user_id,
            "stage": self.stage,
            "intent": self.intent or "없음",
            "keywords": ", ".join(self.keywords) if self.keywords else "없음",
            "exclude_keywords": ", ".join(self.exclude_keywords) if self.exclude_keywords else "없음",
            "quantity": str(self.quantity) if self.quantity is not None else "미정",
            "condition": self.condition or "없음",
            "pending_type": self.pending_type() or "없음",
            "selected_product_name": product_name,
            "selected_product_price": str(product_price),
            "selected_product_json": str(self.selected_product) if self.selected_product else "없음",
            "cart_count": str(cart_count),
            "cart_total": str(cart_total),
            "address_text": self.address_text or "없음",
            "recommendation_context": str(self.recommendation_context) if self.recommendation_context else "없음",
            "error": self.error or "없음",
        }
