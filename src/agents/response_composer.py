"""ResponseComposer — 어르신 친화 응답 생성 + Reflection (Claude Sonnet)."""
import os
from crewai import Agent, LLM


def build_response_composer() -> Agent:
    llm = LLM(
        model="anthropic/claude-sonnet-4-6",
        temperature=0,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )

    return Agent(
        role="어르신 친화 응답 작성가",
        goal=(
            "추천 상품 정보를 70대 어르신이 이해하기 쉬운 2문장 음성 출력용 한국어로 설명한다. "
            "어려운 단어(플랫폼, 할인율, 프로모션 등)를 쓰지 않고, 한 문장은 15자 이내로 작성한다. "
            "생성 후 스스로 검토해 기준 미달이면 더 쉬운 표현으로 고쳐 쓴다."
        ),
        backstory=(
            "고령자 UX 전문가로, 복잡한 쇼핑 정보를 간결하고 친근한 구어체로 변환하는 데 특화되어 있다. "
            "'~이에요', '~할까요?' 같은 존댓말을 사용하며 TTS 출력에 최적화된 문장을 만든다."
        ),
        llm=llm,
        tools=[],
        verbose=False,
        allow_delegation=False,
    )
