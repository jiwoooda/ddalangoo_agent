"""ProductSearcher — 멀티 플랫폼 검색 + 랭킹 (Claude Sonnet)."""
import os
from crewai import Agent, LLM
from src.tools.search_tool import SearchProductsTool


def build_product_searcher() -> Agent:
    llm = LLM(
        model="anthropic/claude-sonnet-4-6",
        temperature=0,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )

    return Agent(
        role="상품 검색 및 랭킹 전문가",
        goal=(
            "네이버·쿠팡·컬리 세 플랫폼을 동시에 검색하고 사용자 선호도를 반영해 "
            "최적의 상품을 1순위로 선정한다. 검색 조건(최저가, 빠른배송 등)을 정확히 적용한다."
        ),
        backstory=(
            "이커머스 플랫폼 비교 분석 전문가로, 가격·배송·리뷰·브랜드 등 다양한 기준을 "
            "종합적으로 평가해 사용자에게 가장 적합한 상품을 찾아낸다."
        ),
        llm=llm,
        tools=[SearchProductsTool()],
        verbose=False,
        allow_delegation=False,
    )
