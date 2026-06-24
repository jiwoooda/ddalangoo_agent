"""
IntentCrew — 매 턴 첫 번째로 실행.
사용자 발화 → intent / keywords / quantity / condition 추출.
"""
import json
from crewai import Task, Crew, Process
from src.agents.intent_analyst import build_intent_analyst


_INTENT_TASK_DESCRIPTION = """
사용자의 쇼핑 발화를 분석해 아래 JSON 형식으로 의도와 슬롯을 추출하라.

현재 대화 상태:
- stage: {stage}
- pending_type: {pending_type}
- 현재 키워드: {keywords}
- 현재 수량: {quantity}

사용자 발화: "{user_input}"

의도 유형 목록:
- buy: 새 상품 구매 요청
- reorder: 이전에 산 상품 재구매
- refine: 검색 조건 변경 ("더 싸게", "빠른 배송으로")
- compare_platforms: 플랫폼 비교 요청
- confirm: 동의/확인 ("응", "네", "맞아요")
- deny: 거부 ("아니", "싫어", "다른 거")
- next: 다음 상품 보기 ("다른 거", "다음 거")
- ask: 상품 질문 ("이거 유기농이야?")
- cancel: 취소 ("그만할게요", "취소")
- quantity_change: 수량 변경
- option_select: 옵션 선택
- unclear: 이해 불가

검색 조건(condition) 목록: 최저가 / 가성비 / 빠른배송 / 인기순 / 무료배송 / 리뷰좋은 / null

규칙:
- buy/reorder/refine/compare_platforms 의도일 때만 keywords를 LLM이 추출한 값으로 설정
- 그 외 의도(ask/confirm/deny/next 등)일 때는 keywords를 현재 키워드({keywords})로 그대로 유지
- 수량은 정수로 변환 (한 개→1, 두 개→2, 세 개→3 등)
- confidence: 0.0~1.0 확신도

반드시 아래 JSON만 반환 (설명 없이):
{{
  "intent": "<의도>",
  "keywords": ["<키워드>"],
  "exclude_keywords": [],
  "quantity": <정수 또는 null>,
  "condition": "<조건 또는 null>",
  "needs_clarification": false,
  "confidence": 0.95,
  "immediate_response": "<한 문장 응답>"
}}
"""


def run_intent_crew(state_inputs: dict) -> dict:
    """
    IntentCrew 실행.
    반환: intent 분석 결과 dict
    """
    analyst = build_intent_analyst()

    task = Task(
        description=_INTENT_TASK_DESCRIPTION,
        expected_output="JSON 형식의 의도 분류 결과",
        agent=analyst,
    )

    crew = Crew(
        agents=[analyst],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff(inputs=state_inputs)
    raw = result.raw.strip()

    try:
        # 코드블록 제거
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception:
        return {
            "intent": "unclear",
            "keywords": state_inputs.get("keywords", "").split(", ") if state_inputs.get("keywords") else [],
            "exclude_keywords": [],
            "quantity": None,
            "condition": None,
            "needs_clarification": True,
            "confidence": 0.0,
            "immediate_response": "다시 말씀해 주세요.",
        }
