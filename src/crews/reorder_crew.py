"""
ReorderCrew — 재구매 흐름.
ReorderResolver → ResponseComposer 순차 실행.
"""
import json
from crewai import Task, Crew, Process
from src.agents.reorder_resolver import build_reorder_resolver
from src.agents.response_composer import build_response_composer


_REORDER_TASK = """
사용자의 구매이력에서 재구매 요청에 맞는 상품을 찾아라.

사용자 ID: {user_id}
검색 키워드: {keywords}
사용자 발화: "{user_input}"

수행 순서:
1. get_purchase_history 도구로 키워드 관련 이력 조회
2. 결과 분석:
   - 매칭 없음 → resolution_type: "no_match"
   - 1건 → resolution_type: "resolved", selected에 해당 상품
   - 2건 이상 → resolution_type: "ambiguous", candidates에 목록

반환 형식 (JSON만 반환):
{{
  "resolution_type": "resolved" | "ambiguous" | "no_match",
  "selected": {{
    "product_name": "<상품명>",
    "price_at_purchase": <가격>,
    "platform": "<플랫폼>",
    "product_url": "<URL>",
    "purchased_at": "<구매일>"
  }},
  "candidates": [<후보 목록 (ambiguous일 때)>],
  "question": "<ambiguous일 때 사용자에게 물어볼 질문>"
}}
"""

_REORDER_RESPONSE_TASK = """
재구매 분석 결과를 바탕으로 어르신 친화 응답을 작성하라.

이전 단계(재구매 분석)의 결과를 참고하라.

규칙:
- resolved: "[상품명], [가격]원이에요. 다시 주문할까요?"
- ambiguous: 이전 단계의 question을 그대로 사용
- no_match: "이전에 사신 기록이 없어요. 새로 찾아드릴까요?"
- 존댓말, 구어체, 2문장 이내
- 금지 단어: 플랫폼, 최저가, 프로모션

텍스트만 반환.
"""


def run_reorder_crew(state_inputs: dict) -> dict:
    """
    ReorderCrew 실행.
    반환: resolution_type, selected_product, explanation, stage
    """
    resolver = build_reorder_resolver()
    composer = build_response_composer()

    reorder_task = Task(
        description=_REORDER_TASK,
        expected_output="JSON 형식의 재구매 후보 분석 결과",
        agent=resolver,
    )

    response_task = Task(
        description=_REORDER_RESPONSE_TASK,
        expected_output="어르신 친화 재구매 안내 문장",
        agent=composer,
        context=[reorder_task],
    )

    crew = Crew(
        agents=[resolver, composer],
        tasks=[reorder_task, response_task],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff(inputs=state_inputs)

    # reorder_task 결과 파싱
    reorder_raw = reorder_task.output.raw.strip() if reorder_task.output else ""
    resolution_type = "no_match"
    selected_product = {}
    candidates = []

    try:
        if reorder_raw.startswith("```"):
            reorder_raw = reorder_raw.split("```")[1]
            if reorder_raw.startswith("json"):
                reorder_raw = reorder_raw[4:]
        parsed = json.loads(reorder_raw.strip())
        resolution_type = parsed.get("resolution_type", "no_match")
        selected_product = parsed.get("selected", {})
        candidates = parsed.get("candidates", [])
        # 구매이력 상품은 price_at_purchase를 사용 → price로 정규화
        if selected_product and "price" not in selected_product:
            selected_product["price"] = selected_product.get("price_at_purchase", 0)
        for c in candidates:
            if "price" not in c:
                c["price"] = c.get("price_at_purchase", 0)
    except Exception:
        pass

    explanation = result.raw.strip()

    if resolution_type == "no_match":
        return {
            "resolution_type": "no_match",
            "selected_product": {},
            "explanation": explanation,
            "stage": "searching",
            "error": "reorder_no_match",
        }
    elif resolution_type == "ambiguous":
        return {
            "resolution_type": "ambiguous",
            "selected_product": {},
            "candidates": candidates,
            "explanation": explanation,
            "stage": "product_confirming",
            "pending_action": {
                "type": "product_select",
                "message": explanation,
                "payload": {"candidates": candidates},
            },
            "error": None,
        }
    else:
        return {
            "resolution_type": "resolved",
            "selected_product": selected_product,
            "explanation": explanation,
            "stage": "product_confirming",
            "pending_action": {
                "type": "product_confirm",
                "message": explanation,
            },
            "error": None,
        }
