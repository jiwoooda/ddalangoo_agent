"""PaymentProcessor — 결제 5단계 처리 (Claude Haiku + 장바구니 도구)."""
import os
from crewai import Agent, LLM
from src.tools.cart_tool import AddToCartTool, GetCartTool, PlaceOrderTool, GetAddressTool
from src.tools.preference_tool import UpdatePreferenceTool


def build_payment_processor() -> Agent:
    llm = LLM(
        model="claude-haiku-4-5-20251001",
        temperature=0,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )

    return Agent(
        role="결제 처리 담당자",
        goal=(
            "현재 결제 단계(pending_type)에 따라 정확한 작업을 수행한다. "
            "Step 0: 장바구니 담기. Step 1: 총액+결제수단 확인. "
            "Step 2: 배송지 확인. Step 3: 비밀번호 요청. Step 4: 주문 실행 + 선호도 업데이트."
        ),
        backstory=(
            "온라인 쇼핑 결제 프로세스 전문가로, 각 단계에서 필요한 정보를 확인하고 "
            "사용자가 안심하고 결제를 완료할 수 있도록 안내한다."
        ),
        llm=llm,
        tools=[AddToCartTool(), GetCartTool(), PlaceOrderTool(), GetAddressTool(), UpdatePreferenceTool()],
        verbose=False,
        allow_delegation=False,
    )
