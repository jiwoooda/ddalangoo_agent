"""
Response Agent Node.

역할: 랭킹된 상품 → 노인 친화 자연어 설명 생성 / 상품 QA 답변.
(기존 product_agent Phase 2 분리)
"""
import json
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from configs.llm_config import get_llm
from src.state.schema import ShoppingState
from src.prompts.response_prompt import RESPONSE_EXPLAIN_PROMPT, RESPONSE_QA_PROMPT
from src.utils.agent_logger import agent_logger

_llm: BaseChatModel | None = None

# 어르신 친화 검증 — 이 단어가 나오면 Reflection 실패로 재생성
_ELDERLY_FORBIDDEN = ["플랫폼", "최저가", "가성비", "혜택", "할인율", "할인가", "프로모션"]
_MAX_SENTENCE_LEN = 35


def _get_llm() -> BaseChatModel:
    global _llm
    if _llm is None:
        _llm = get_llm("response", temperature=0, max_tokens=300)
    return _llm


def _reflect_elderly(text: str) -> tuple[bool, str]:
    """Reflection: 어르신 친화 출력 검증. (True, "") = 통과."""
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    if len(sentences) > 3:
        return False, f"문장 수 과다({len(sentences)}개)"
    for s in sentences:
        if len(s) > _MAX_SENTENCE_LEN:
            return False, f"긴 문장({len(s)}자): {s[:20]}..."
    found = [w for w in _ELDERLY_FORBIDDEN if w in text]
    if found:
        return False, f"어려운 단어 포함: {', '.join(found)}"
    return True, ""


def _simplify_with_haiku(explanation: str, reason: str) -> str:
    """Reflection 실패 시 context 모델(Haiku)로 재생성. LangSmith 자동 추적."""
    try:
        llm = get_llm("context", temperature=0, max_tokens=150)
        content = (
            f"다음 쇼핑 안내 문장을 70대 어르신이 이해하기 쉽게 고쳐주세요.\n"
            f"문제점: {reason}\n"
            f"원문: {explanation}\n\n"
            f"규칙: 2문장 이내 / 한 문장 15자 이내 / 쉬운 단어만 / 존댓말(~이에요, ~할까요?) / 텍스트만 반환"
        )
        return llm.invoke([HumanMessage(content=content)]).content.strip()
    except Exception:
        return explanation


def _format_preference(preference_context: dict) -> str:
    if not preference_context or not preference_context.get("summary"):
        return "선호 정보 없음 (구매이력 부족)"
    lines = [preference_context["summary"]]
    keyword_summary = preference_context.get("keyword_summary") or ""
    if keyword_summary:
        lines.append(f"키워드 관련 선호: {keyword_summary}")
    return "\n".join(lines)


def _fallback_explanation(product: dict) -> str:
    """LLM 실패 또는 빈 설명 시 상품 필드로 최소 문장 생성."""
    name = product.get("name") or "상품"
    price = product.get("price")
    platform = product.get("platform") or ""
    price_str = f"{price:,}원" if isinstance(price, (int, float)) else (str(price) if price else "")
    parts = [name]
    if price_str:
        parts.append(f"{price_str}이에요.")
    if platform:
        parts.append(f"{platform}에서 판매 중이에요.")
    return " ".join(parts)


def _extract_user_question(state: ShoppingState) -> str | None:
    for msg in reversed(state.get("messages") or []):
        if isinstance(msg, dict):
            if msg.get("role") == "user":
                return msg.get("content")
        else:
            role = getattr(msg, "type", None) or getattr(msg, "role", None)
            if role == "human":
                return getattr(msg, "content", None)
    return None


def response_agent_node(state: ShoppingState) -> dict:
    intent = state.get("intent")
    keywords = state.get("keywords") or []
    condition = state.get("condition")
    recommendation_context = state.get("recommendation_context") or {}
    preference_context = recommendation_context.get("preference_context") or {}
    recommended_products = state.get("recommended_products") or []
    current_idx = state.get("current_product_index") or 0

    # ── QA ──
    if intent == "ask":
        target = state.get("selected_product") or (
            recommended_products[current_idx] if recommended_products else None
        )
        user_question = _extract_user_question(state)

        # 배송지 조회 질문 — 상품 없어도 바로 답변
        _ADDRESS_KEYWORDS = ("배송지", "주소", "배달지", "받는 곳", "배달 주소")
        if user_question and any(k in user_question for k in _ADDRESS_KEYWORDS):
            from src.tools.mock_tools import mock_get_default_address
            user_id = state.get("user_id", "")
            addr = mock_get_default_address(user_id)
            if addr:
                addr_text = " ".join(filter(None, [
                    addr.get("address_line1"), addr.get("address_line2")
                ]))
                msg = f"등록된 배송지는 {addr_text}이에요."
            else:
                msg = "등록된 배송지가 없어요. 배송지를 알려주시면 저장해 드릴게요."
            return {
                "explanation": msg,
                "pending_action": {"type": "address_confirm", "message": msg},
                "stage": state.get("stage", "idle"),
                "last_agent": "response_agent",
                "error": None,
            }

        if not target or not user_question:
            return {
                "stage": state.get("stage", "idle"),
                "needs_clarification": True,
                "pending_action": {
                    "type": "clarification",
                    "message": "어떤 상품에 대해 물어보시는 건지 먼저 알려주세요.",
                    "payload": {},
                },
                "last_agent": "response_agent",
                "error": None,
            }
        answer = _get_llm().invoke([HumanMessage(content=RESPONSE_QA_PROMPT.format(
            product_json=json.dumps(target, ensure_ascii=False),
            question=user_question,
        ))]).content.strip()
        ok, reason = _reflect_elderly(answer)
        haiku_fallback = False
        if not ok:
            agent_logger.log(f"[response_agent] Reflection 실패: {reason} → Haiku 재생성")
            answer = _simplify_with_haiku(answer, reason)
            haiku_fallback = True
        agent_logger.log(f"[response_agent] QA 답변: {answer}")
        return {
            "explanation": answer,
            "reflection_passed": ok,
            "haiku_fallback": haiku_fallback,
            "reflection_reason": reason,
            "stage": "product_confirming",
            "last_agent": "response_agent",
            "error": None,
        }

    # ── 설명 생성 ──
    product = state.get("selected_product")
    if not product:
        return {"stage": "idle", "error": "no_product", "last_agent": "response_agent"}

    try:
        explanation = _get_llm().invoke([HumanMessage(content=RESPONSE_EXPLAIN_PROMPT.format(
            product_json=json.dumps(product, ensure_ascii=False),
            keywords=json.dumps(keywords, ensure_ascii=False),
            condition=condition or "없음",
            preference_context=_format_preference(preference_context),
        ))]).content.strip()
    except Exception as e:
        agent_logger.log(f"[response_agent] 설명 생성 오류: {e} → fallback")
        explanation = ""

    if not explanation:
        explanation = _fallback_explanation(product)
        agent_logger.log(f"[response_agent] fallback 설명: {explanation}")

    # Reflection: 어르신 친화도 검증 → 실패 시 Haiku로 재생성
    ok, reason = _reflect_elderly(explanation)
    haiku_fallback = False
    if not ok:
        agent_logger.log(f"[response_agent] Reflection 실패: {reason} → Haiku 재생성")
        explanation = _simplify_with_haiku(explanation, reason)
        haiku_fallback = True

    agent_logger.log(f"[response_agent] 설명: {explanation}")

    # LLM이 설명 끝에 "주문할까요?" 등 CTA를 붙이는 경우 제거 (pending_msg와 중복 방지)
    _cta_endings = ("주문할까요?", "어떠세요?", "구매할까요?", "사드릴까요?", "주문해드릴까요?")
    explanation_clean = explanation
    for cta in _cta_endings:
        if explanation_clean.endswith(cta):
            explanation_clean = explanation_clean[: -len(cta)].rstrip(" .·\n")
            break

    quantity = state.get("quantity")
    if quantity:
        pending_msg = "주문할까요?"
    else:
        pending_msg = "주문을 원하시면 수량을 말씀해 주세요."

    return {
        "explanation": explanation,
        "reflection_passed": ok,
        "haiku_fallback": haiku_fallback,
        "reflection_reason": reason,
        "pending_action": {"type": "product_confirm", "message": f"{explanation_clean}\n{pending_msg}"},
        "stage": "product_confirming",
        "last_agent": "response_agent",
        "error": None,
    }
