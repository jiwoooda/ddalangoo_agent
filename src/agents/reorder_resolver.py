"""ReorderResolver — 구매이력 기반 재구매 후보 탐색 (Claude Haiku)."""
import os
from crewai import Agent, LLM
from src.tools.history_tool import PurchaseHistoryTool


def build_reorder_resolver() -> Agent:
    llm = LLM(
        model="claude-haiku-4-5-20251001",
        temperature=0,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )

    return Agent(
        role="재구매 이력 분석가",
        goal=(
            "사용자의 구매이력에서 재구매 요청에 맞는 상품을 찾는다. "
            "후보가 1개면 바로 선택하고, 2개 이상이면 목록을 제시해 사용자가 고르게 한다. "
            "이력이 없으면 일반 신규 검색으로 전환해야 함을 알린다."
        ),
        backstory=(
            "고객 재구매 패턴 분석 전문가로, 과거 구매이력에서 사용자가 원하는 상품을 "
            "정확히 찾아내 재구매 과정을 최대한 간단하게 만든다."
        ),
        llm=llm,
        tools=[PurchaseHistoryTool()],
        verbose=False,
        allow_delegation=False,
    )
