"""
BuyCrew — 구매/refine/compare_platforms 흐름.
ContextLoader → ProductSearcher → ResponseComposer 순차 실행.
"""
import json
import re
from crewai import Task, Crew, Process
from src.agents.context_loader import build_context_loader
from src.agents.product_searcher import build_product_searcher
from src.agents.response_composer import build_response_composer


_CONTEXT_TASK = """
사용자의 구매이력과 선호도를 조회해 추천 컨텍스트를 구성하라.

사용자 ID: {user_id}
검색 키워드: {keywords}

수행 순서:
1. get_purchase_history 도구로 키워드 관련 구매이력 조회 (keywords 파라미터에 검색 키워드 입력)
2. get_preference_memory 도구로 선호도 메모리 조회
3. 아래 JSON 형식으로 결과 반환

반환 형식 (JSON만 반환):
{{
  "purchase_count": <총 구매 건수>,
  "keyword_history": [<키워드 관련 구매이력>],
  "preference_summary": "<선호 브랜드, 가격대, 플랫폼을 2문장으로 요약>",
  "preferred_platform": "<선호 플랫폼 또는 null>",
  "preferred_brands": [<선호 브랜드 목록>],
  "price_range": {{<카테고리별 avg/min/max>}}
}}
"""

_PRODUCT_TASK = """
아래 정보를 바탕으로 상품을 검색하고 최적 상품을 선정하라.

검색 키워드: {keywords}
검색 조건: {condition}
제외 키워드: {exclude_keywords}

이전 단계(컨텍스트 로더)의 사용자 선호도 정보를 참고해 랭킹에 반영하라.

수행 순서:
1. search_products_tool 도구 호출: query={keywords}, platforms=naver,coupang,kurly, condition={condition}
2. 결과 중 제외 키워드({exclude_keywords})가 포함된 상품 제거
3. 사용자 선호도(선호 브랜드, 가격대, 플랫폼)를 반영해 순위 결정
4. 아래 JSON 형식으로 반환

반환 형식 (JSON만 반환):
{{
  "selected_product": {{
    "product_name": "<상품명>",
    "price": <가격>,
    "platform": "<플랫폼>",
    "delivery": "<배송정보>",
    "rating": <평점 또는 null>,
    "review_count": <리뷰수 또는 null>,
    "product_url": "<URL>",
    "is_sold_out": false
  }},
  "ranked_products": [<전체 랭킹된 상품 목록 최대 9개>],
  "rank_reason": "<1순위 선정 이유 한 문장>"
}}
"""

_RESPONSE_TASK = """
추천 상품에 대한 어르신 친화 설명을 2문장으로 작성하라.

이전 단계에서 선정된 상품 정보와 선호도 컨텍스트를 참고하라.

검색 키워드: {keywords}
검색 조건: {condition}

문장 구조:
- 1문장: 상품명(full) + 가격. 예: "풀무원 달걀 10구, 4,900원이에요."
- 2문장: 추천 이유 1가지 + "주문할까요?"
  예: "평소 즐겨 사시는 브랜드예요. 주문할까요?"

규칙:
- 한 문장 15자 이내 목표
- 금지 단어: 플랫폼, 최저가, 가성비, 혜택, 할인율, 할인가, 프로모션
- 존댓말: ~이에요, ~할까요?
- 구어체, 친근한 어투
- 생성 후 스스로 검토: 문장이 길거나 어려운 단어가 있으면 다시 고쳐 쓴다

텍스트만 반환 (JSON 아님).
"""


def run_buy_crew(state_inputs: dict) -> dict:
    """
    BuyCrew 실행.
    반환: selected_product, ranked_products, explanation
    """
    context_loader = build_context_loader()
    product_searcher = build_product_searcher()
    response_composer = build_response_composer()

    context_task = Task(
        description=_CONTEXT_TASK,
        expected_output="JSON 형식의 사용자 선호도 컨텍스트",
        agent=context_loader,
    )

    product_task = Task(
        description=_PRODUCT_TASK,
        expected_output="JSON 형식의 선정 상품 및 랭킹 목록",
        agent=product_searcher,
        context=[context_task],
    )

    response_task = Task(
        description=_RESPONSE_TASK,
        expected_output="2문장 어르신 친화 상품 설명",
        agent=response_composer,
        context=[product_task],
    )

    crew = Crew(
        agents=[context_loader, product_searcher, response_composer],
        tasks=[context_task, product_task, response_task],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff(inputs=state_inputs)

    # product_task 결과 파싱 — LLM이 설명 텍스트 + ```json 블록을 반환하는 경우 대응
    product_raw = product_task.output.raw.strip() if product_task.output else ""
    selected_product = {}
    ranked_products = []
    try:
        # ```json ... ``` 블록을 우선 추출
        m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', product_raw, re.DOTALL)
        if m:
            parsed = json.loads(m.group(1).strip())
        else:
            parsed = json.loads(product_raw)
        selected_product = parsed.get("selected_product", {})
        ranked_products = parsed.get("ranked_products", [selected_product])
    except Exception:
        pass

    explanation = result.raw.strip()

    return {
        "selected_product": selected_product,
        "ranked_products": ranked_products,
        "explanation": explanation,
        "stage": "product_confirming" if selected_product else "idle",
        "error": None if selected_product else "no_candidates",
        "pending_action": {"type": "product_confirm", "message": explanation} if selected_product else None,
    }
