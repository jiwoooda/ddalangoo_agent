"""IntentAnalyst — 사용자 발화 의도 분류 (GPT-4o-mini)."""
import os
from crewai import Agent, LLM


def build_intent_analyst() -> Agent:
    llm = LLM(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

    return Agent(
        role="쇼핑 의도 분석가",
        goal=(
            "사용자의 발화에서 쇼핑 의도와 관련 슬롯(키워드, 수량, 조건 등)을 정확하게 추출한다. "
            "현재 대화 stage와 pending_action을 고려해 의도를 판단한다."
        ),
        backstory=(
            "10년 경력의 한국어 자연어처리 전문가로, 고령 사용자의 쇼핑 발화 패턴에 정통하다. "
            "'응', '아니', '그거' 같은 짧은 발화도 문맥을 보고 정확히 해석한다."
        ),
        llm=llm,
        tools=[],
        verbose=False,
        allow_delegation=False,
    )
