"""
Intent Agent Node.

역할: 사용자 발화 → intent + slot 추출.
with_structured_output(Pydantic)으로 스키마를 강제해 누락 방지.
"""
import re
from typing import Literal, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from configs.llm_config import get_llm
from src.state.schema import ShoppingState
from src.prompts.intent_prompt import INTENT_AGENT_PROMPT
from src.utils.agent_logger import agent_logger

IntentType = Literal[
    "buy", "reorder", "confirm", "deny", "next", "refine",
    "compare_platforms", "quantity_change", "address_change",
    "option_select", "ask", "cancel", "unclear",
]

ConditionType = Literal["최저가", "가성비", "빠른배송", "인기순", "무료배송", "리뷰좋은"]

_KR_NUM = {
    "하나": 1, "한": 1, "일": 1,
    "둘": 2, "두": 2,
    "셋": 3, "세": 3,
    "넷": 4, "네": 4,
    "다섯": 5, "오": 5,
    "여섯": 6, "육": 6,
    "일곱": 7, "칠": 7,
    "여덟": 8, "팔": 8,
    "아홉": 9, "구": 9,
    "열": 10, "십": 10,
    "스물": 20, "스무": 20, "이십": 20,
}

def _looks_like_quantity_reply(text: str) -> bool:
    """STT 텍스트가 수량 답변처럼 보이는지 판단."""
    import re
    t = text.strip()
    if re.fullmatch(r'\d+', t):
        return True
    units = r'(개|인분|명|병|통|팩|봉|캔|묶음|박스|세트)'
    kr_nums = r'(하나|한|둘|두|셋|세|넷|네|다섯|여섯|일곱|여덟|아홉|열|\d+)'
    return bool(re.search(kr_nums + r'\s*' + units, t))


def _normalize_keyword_tokens(keywords: list[str]) -> list[str]:
    """중복·공백 제거 및 빈 문자열 필터."""
    seen: set[str] = set()
    result = []
    for kw in keywords:
        kw = kw.strip()
        if kw and kw not in seen:
            seen.add(kw)
            result.append(kw)
    return result


_STOP_WORDS = frozenset({
    "사줘", "사줄래", "사주세요", "사고싶어", "사고 싶어", "구매", "구매해줘", "주문",
    "주문해줘", "찾아줘", "보여줘", "주세요", "해줘", "해줄래", "싶어", "좀", "저",
    "제", "그냥", "그거", "이거", "저거",
})

def _fallback_search_keywords(user_input: str) -> list[str]:
    """LLM이 빈 keywords 반환 시 raw text에서 단순 휴리스틱 추출."""
    import re
    tokens = re.split(r'[\s,]+', user_input.strip())
    result = [t for t in tokens if t and t not in _STOP_WORDS and len(t) >= 2]
    return result[:3]


_BUY_TRIGGERS = frozenset({"사줘", "사줄래", "사주세요", "구매해", "구매해줘", "주문해", "사고싶어", "사 줘"})
_REORDER_SIGNALS = frozenset({"저번에", "지난번에", "재주문", "똑같이 다시", "예전에 산"})

def _should_force_buy_from_freeform(user_input: str, intent: str, stage: str) -> bool:
    """idle 상태에서 buy 트리거가 있는데 LLM이 다른 intent를 뽑았을 때 buy로 교정.
    reorder는 교정 대상에서 제외 — 재구매 신호가 buy 트리거보다 우선."""
    if intent in ("buy", "reorder") or stage != "idle":
        return False
    return any(t in user_input for t in _BUY_TRIGGERS)

def _should_force_reorder(user_input: str, intent: str, stage: str, keywords: list) -> bool:
    """재구매 신호 + 상품명이 있는데 LLM이 buy로 잘못 분류했을 때 reorder로 교정."""
    if intent == "reorder" or stage != "idle":
        return False
    has_reorder_signal = any(t in user_input for t in _REORDER_SIGNALS)
    has_product = bool(keywords)
    return has_reorder_signal and has_product


def _parse_quantity(v) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, int):
        return v
    text = str(v).strip()
    m = re.search(r"(\d+)", text)
    if m:
        return int(m.group(1))
    for kr, num in sorted(_KR_NUM.items(), key=lambda x: -len(x[0])):
        if kr in text:
            return num
    return None


class IntentOutput(BaseModel):
    intent: IntentType = Field(description="사용자 의도")
    keywords: list[str] = Field(default_factory=list, description="검색할 상품명/카테고리/브랜드")
    exclude_keywords: list[str] = Field(default_factory=list, description="제외할 브랜드/플랫폼/상품명")
    negative_constraints: list[str] = Field(default_factory=list, description="자연어 제외 조건")
    quantity: Optional[int] = Field(
        default=None,
        description="명시된 수량만 정수로. 없으면 null.",
        json_schema_extra={"examples": [1, 2, 3, 5, 10]},
    )
    condition: Optional[ConditionType] = Field(default=None, description="검색 조건")
    recipe_dish: Optional[str] = Field(default=None, description="재료 구매 요리명 (예: 된장찌개). 직접 상품 구매면 null")
    recipe_people: Optional[int] = Field(default=None, description="인원수 (예: 4인 가족 → 4). 없으면 null")
    target_platforms: list[str] = Field(default_factory=list, description="비교 대상 플랫폼 목록")
    override_platform: Optional[str] = Field(default=None, description="명시적으로 지정한 단일 플랫폼")
    current_option_value: Optional[str] = Field(default=None, description="명시된 상품 옵션값")
    address_text: Optional[str] = Field(default=None, description="사용자가 말한 배송지 텍스트")
    needs_clarification: bool = Field(default=False, description="추가 정보가 필요하면 true")
    clarification_reason: Optional[str] = Field(default=None, description="needs_clarification=true일 때 이유")
    confidence: float = Field(default=0.95, description="의도 해석 확신도 0.0~1.0")
    immediate_response: str = Field(default="", description="음성 출력용 한 문장 응답")


_llm = None
_structured_llm = None


def _get_llm():
    global _llm, _structured_llm
    if _llm is None:
        _llm = get_llm("intent", temperature=0)
        _structured_llm = _llm.with_structured_output(IntentOutput, method="json_schema")
    return _structured_llm


def _extract_user_input(state: ShoppingState) -> str:
    messages = state.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, dict):
            if msg.get("role") == "user":
                return msg.get("content", "")
        else:
            role = getattr(msg, "type", None) or getattr(msg, "role", None)
            if role == "human":
                return getattr(msg, "content", "")
    return ""


def _is_ambiguous_reorder(user_input: str, keywords: list[str]) -> bool:
    text = user_input.strip().lower()
    if keywords:
        return False
    has_reorder_signal = any(token in text for token in ("다시", "또", "재주문", "똑같이", "저번", "지난번"))
    has_ambiguous_ref = any(token in text for token in ("그거", "그것", "저거", "그 상품", "주문한 거", "산 거", "샀던 거"))
    return has_reorder_signal and has_ambiguous_ref


def intent_agent_node(state: ShoppingState) -> dict:
    user_input = _extract_user_input(state)
    stage = state.get("stage", "idle")
    pending_action = state.get("pending_action")
    pending_type = pending_action.get("type") if isinstance(pending_action, dict) else "null"

    prompt = INTENT_AGENT_PROMPT.format(
        user_input=user_input,
        stage=stage,
        pending_action=pending_type,
        context="",
    )

    llm = _get_llm()
    try:
        # OpenAI는 SystemMessage, Claude는 HumanMessage 필수
        # _llm(기본 모델)로 모델명 확인 — structured_llm 래퍼에는 속성 없음
        base_model_name = getattr(_llm, "model_name", "") or getattr(_llm, "model", "") or ""
        if "gpt" in str(base_model_name).lower():
            messages = [SystemMessage(content=prompt)]
        else:
            messages = [HumanMessage(content=prompt)]
        parsed: IntentOutput = llm.invoke(messages)
    except Exception as e:
        print(f"[intent_agent] structured output error: {e}")
        return {
            "intent": "unclear",
            "keywords": state.get("keywords") or [],
            "exclude_keywords": [],
            "negative_constraints": [],
            "quantity": state.get("quantity"),
            "condition": None,
            "recipe_dish": state.get("recipe_dish"),
            "recipe_people": state.get("recipe_people"),
            "target_platforms": [],
            "override_platform": None,
            "current_option_value": None,
            "address_text": None,
            "needs_clarification": True,
            "clarification_reason": "응답 파싱 오류",
            "confidence": 0.0,
            "immediate_response": "다시 한번 말씀해 주세요.",
            "last_agent": "intent_agent",
            "tool_calls": None,
            "tool_results": None,
        }

    intent = parsed.intent

    # ── reorder 강제 교정: 재구매 신호+상품명 있는데 buy로 잘못 분류된 경우 ──
    kws_for_check = _normalize_keyword_tokens(parsed.keywords or []) or _fallback_search_keywords(user_input)
    if _should_force_reorder(user_input, intent, stage, kws_for_check):
        intent = "reorder"

    # ── buy 강제 교정: idle에서 사줘/구매해 등 트리거 있는데 LLM이 다른 intent ──
    if _should_force_buy_from_freeform(user_input, intent, stage):
        intent = "buy"

    quantity = _parse_quantity(parsed.quantity)
    # quantity_confirm 또는 product_confirm(수량 미입력) 대기 중 수량 답변 → 재파싱 + intent 교정
    if pending_type in ("quantity_confirm", "product_confirm"):
        if _looks_like_quantity_reply(user_input):
            direct = _parse_quantity(user_input)
            if direct is not None:
                quantity = direct
                # product_confirm 상태에서 수량을 말하는 건 구매 의사 확정으로 해석
                if pending_type == "product_confirm" and not state.get("quantity"):
                    intent = "confirm"

    _search_intents = {"buy", "reorder", "refine", "compare_platforms"}
    if quantity is None and intent not in _search_intents:
        quantity = state.get("quantity")

    # 검색과 무관한 intent는 기존 keywords 유지 (ask/confirm/deny/next 등이 keywords를 덮어쓰면 안 됨)
    if intent in _search_intents:
        keywords = _normalize_keyword_tokens(parsed.keywords or []) or state.get("keywords") or []
        # LLM이 빈 keywords 반환 → 휴리스틱 추출
        if not keywords:
            keywords = _fallback_search_keywords(user_input)
    else:
        keywords = state.get("keywords") or _normalize_keyword_tokens(parsed.keywords or [])

    needs_clarification = parsed.needs_clarification
    clarification_reason = parsed.clarification_reason
    confidence = parsed.confidence
    immediate_response = parsed.immediate_response

    # 강제 교정된 reorder: 상품명 있으므로 clarification 불필요, confidence 보정
    if intent == "reorder" and keywords and needs_clarification:
        needs_clarification = False
        clarification_reason = None
        confidence = max(confidence, 0.8)

    # 수량 답변 감지로 quantity가 교정된 경우 clarification 불필요
    if quantity and intent == "confirm" and pending_type in ("quantity_confirm", "product_confirm"):
        needs_clarification = False
        clarification_reason = None
        confidence = max(confidence, 0.85)

    if _is_ambiguous_reorder(user_input, keywords):
        intent = "reorder"
        needs_clarification = True
        clarification_reason = "어떤 상품을 다시 주문할지 알려주세요."
        immediate_response = "어떤 상품을 다시 주문할까요?"

    # recipe 필드는 buy intent일 때만 갱신, 그 외엔 state 값 유지
    recipe_dish = parsed.recipe_dish if intent == "buy" else (parsed.recipe_dish or state.get("recipe_dish"))
    recipe_people = parsed.recipe_people if intent == "buy" else (parsed.recipe_people or state.get("recipe_people"))

    result = {
        "intent": intent,
        "keywords": keywords,
        "exclude_keywords": parsed.exclude_keywords,
        "negative_constraints": parsed.negative_constraints,
        "quantity": quantity,
        "condition": parsed.condition,
        "recipe_dish": recipe_dish,
        "recipe_people": recipe_people,
        "target_platforms": parsed.target_platforms,
        "override_platform": parsed.override_platform,
        "current_option_value": parsed.current_option_value,
        "address_text": parsed.address_text,
        "needs_clarification": needs_clarification,
        "clarification_reason": clarification_reason,
        "confidence": confidence if confidence > 0 else 0.9,
        "immediate_response": immediate_response,
        "last_agent": "intent_agent",
        "tool_calls": None,
        "tool_results": None,
    }
    agent_logger.log_intent(user_input, stage, pending_action, result)
    return result
