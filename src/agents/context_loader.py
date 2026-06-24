"""ContextLoader — 사용자 선호도·구매이력 컨텍스트 로딩 (Claude Haiku)."""
import os
from crewai import Agent, LLM
from src.tools.history_tool import PurchaseHistoryTool, PreferenceMemoryTool


def build_context_loader() -> Agent:
    llm = LLM(
        model="claude-haiku-4-5-20251001",
        temperature=0,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )

    return Agent(
        role="사용자 컨텍스트 로더",
        goal=(
            "사용자의 구매이력과 선호도 메모리를 조회해 상품 추천에 활용할 수 있는 "
            "개인화 컨텍스트를 구성한다. 선호 브랜드, 가격대, 플랫폼, 재구매 패턴을 요약한다."
        ),
        backstory=(
            "개인화 추천 시스템 전문가로, 사용자의 과거 구매 패턴에서 "
            "의미 있는 선호도 신호를 추출해 추천 품질을 높이는 역할을 담당한다."
        ),
        llm=llm,
        tools=[PurchaseHistoryTool(), PreferenceMemoryTool()],
        verbose=False,
        allow_delegation=False,
    )
