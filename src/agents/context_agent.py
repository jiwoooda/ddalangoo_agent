"""
Context Agent Node.

역할:
  [Context Loading]  buy/refine/compare_platforms → 사용자 프로필/구매이력/선호 조회
  [Post-Payment]     stage == "completed" → 대화 요약 저장
"""
from typing import Any, Optional
from langgraph.store.base import BaseStore
from langchain_core.messages import HumanMessage

from configs.llm_config import get_llm
from src.state.schema import ShoppingState
from src.utils.agent_logger import agent_logger
from src.tools.mock_tools import (
    mock_get_user,
    mock_get_default_address,
    mock_get_preference_memory,
    mock_vector_search_personal,
    mock_vector_search_collective,
    mock_get_purchase_history,
    mock_update_preference_from_purchase,
)

PERSONAL_VECTOR_THRESHOLD = 20

_preference_cache: dict[str, dict[str, Any]] = {}
_context_llm = None


def _get_llm():
    global _context_llm
    if _context_llm is None:
        _context_llm = get_llm("context", temperature=0, max_tokens=400)
    return _context_llm


def _merge_recommendation_results(
    keyword_results: list[dict[str, Any]],
    personal_vector_results: list[dict[str, Any]],
    collective_vector_results: list[dict[str, Any]],
    intent: Optional[str] = None,
) -> list[dict[str, Any]]:
    merged = []
    seen: set = set()
    sources = [
        ("personal_vector", personal_vector_results),
        ("keyword", keyword_results),
        ("collective_vector", collective_vector_results),
    ]
    for source, results in sources:
        for item in results:
            key = item.get("product_url") or item.get("product_name") or item.get("id")
            if not key or key in seen:
                continue
            item = dict(item)
            item["_memory_source"] = source
            merged.append(item)
            seen.add(key)
    return merged[:10]


def _compute_general_preference(histories: list[dict[str, Any]]) -> dict[str, Any]:
    brand_counts: dict[str, int] = {}
    prices: list[int] = []
    platform_counts: dict[str, int] = {}
    product_counts: dict[str, int] = {}

    for h in histories:
        if b := h.get("brand"):
            brand_counts[b] = brand_counts.get(b, 0) + 1
        if p := h.get("price_at_purchase"):
            prices.append(int(p))
        if pl := h.get("platform"):
            platform_counts[pl] = platform_counts.get(pl, 0) + 1
        if n := h.get("product_name"):
            product_counts[n] = product_counts.get(n, 0) + 1

    preferred_brands = sorted(
        [{"brand": b, "count": c} for b, c in brand_counts.items()],
        key=lambda x: -x["count"],
    )[:5]

    price_range: dict[str, int] = {}
    if prices:
        price_range = {"avg": sum(prices) // len(prices), "min": min(prices), "max": max(prices)}

    repurchase_patterns = [
        name for name, count in sorted(product_counts.items(), key=lambda x: -x[1]) if count >= 2
    ][:5]

    preferred_platform = max(platform_counts, key=platform_counts.get) if platform_counts else None

    parts: list[str] = []
    if preferred_brands:
        parts.append("자주 구매한 브랜드: " + ", ".join(b["brand"] for b in preferred_brands[:3]))
    if price_range:
        parts.append(f"평균 구매가: {price_range['avg']:,}원")
    if repurchase_patterns:
        parts.append("재구매 상품: " + ", ".join(repurchase_patterns[:3]))
    if preferred_platform:
        parts.append(f"주로 이용 플랫폼: {preferred_platform}")

    return {
        "preferred_brands": preferred_brands,
        "price_range": price_range,
        "repurchase_patterns": repurchase_patterns,
        "preferred_platform": preferred_platform,
        "summary": ". ".join(parts) if parts else "",
    }


def _generate_llm_summary(preference: dict[str, Any], keyword_history: Optional[list] = None) -> str:
    """LangSmith 자동 추적: raw anthropic 대신 LangChain ChatModel 사용."""
    try:
        brands = ", ".join(b["brand"] for b in (preference.get("preferred_brands") or [])[:3])
        pr = preference.get("price_range") or {}
        repurchase = ", ".join((preference.get("repurchase_patterns") or [])[:3])
        platform = preference.get("preferred_platform") or ""

        prompt = (
            "다음은 쇼핑 앱 사용자의 구매이력 통계입니다. "
            "이 데이터를 바탕으로 상품 추천에 활용할 수 있는 간결한 선호도 요약을 2~3문장으로 작성하세요. "
            "자연스러운 한국어로 작성하고, 추천 기준이 될 핵심 특성(브랜드 성향, 가격대, 재구매 패턴, 선호 플랫폼)을 포함하세요.\n\n"
            f"- 선호 브랜드: {brands or '없음'}\n"
            f"- 평균 구매가: {pr.get('avg', 0):,}원 (범위: {pr.get('min', 0):,}~{pr.get('max', 0):,}원)\n"
            f"- 재구매 상품: {repurchase or '없음'}\n"
            f"- 주 이용 플랫폼: {platform or '없음'}"
        )

        if keyword_history:
            kw_names = ", ".join(h["product_name"] for h in keyword_history if h.get("product_name"))
            if kw_names:
                prompt += f"\n- 관련 키워드 구매이력: {kw_names}"
            prompt += "\n\n위 키워드 관련 구매이력도 포함해 요약하세요."

        return _get_llm().invoke([HumanMessage(content=prompt)]).content.strip()
    except Exception:
        return preference.get("summary", "")


def build_preference_context(user_id: str, keywords: list[str]) -> dict[str, Any]:
    histories = mock_get_purchase_history(user_id)
    if not histories:
        return {}

    cache_hit = user_id in _preference_cache
    if cache_hit:
        general_pref = _preference_cache[user_id]
        agent_logger.log(f"[context_agent] 일반 선호도 캐시 HIT (user_id={user_id})")
    else:
        general_pref = _compute_general_preference(histories)
        general_pref["summary"] = _generate_llm_summary(general_pref)
        _preference_cache[user_id] = general_pref
        agent_logger.log(f"[context_agent] 일반 선호도 캐시 저장 완료")

    keyword_history: list = []
    keyword_summary: str = ""
    if keywords:
        keyword_history = [
            {
                "product_name": h.get("product_name"),
                "brand": h.get("brand"),
                "price": h.get("price_at_purchase"),
                "platform": h.get("platform"),
            }
            for h in histories
            if any(
                kw.lower() in (h.get("product_name") or "").lower()
                or kw.lower() in (h.get("keyword") or "").lower()
                or kw.lower() in (h.get("category") or "").lower()
                for kw in keywords
            )
        ][:5]

        if keyword_history:
            keyword_summary = _generate_llm_summary(general_pref, keyword_history)

    return {
        **general_pref,
        "keyword_history": keyword_history,
        "keyword_summary": keyword_summary,
        "_cache_hit": cache_hit,
    }


def get_recommendation_context(
    user_id: str,
    keywords: list[str],
    intent: Optional[str] = None,
) -> dict[str, Any]:
    user_profile = mock_get_user(user_id) or {}
    default_address = mock_get_default_address(user_id)
    preference_memory = mock_get_preference_memory(user_id)

    histories = mock_get_purchase_history(user_id)
    purchase_count = len(histories)

    query = " ".join(keywords)
    keyword_results = [
        h for h in histories
        if any(
            k.lower() in (h.get("product_name") or "").lower()
            or k.lower() in (h.get("keyword") or "").lower()
            or k.lower() in (h.get("category") or "").lower()
            for k in keywords
        )
    ][:5]

    use_personal_vector = purchase_count >= PERSONAL_VECTOR_THRESHOLD
    personal_vector_results = mock_vector_search_personal(user_id, query) if use_personal_vector else []

    age_group = user_profile.get("age_group")
    collective_vector_results = mock_vector_search_collective(query=query, age_group=age_group)

    if use_personal_vector:
        retrieval_mode = "hybrid_personal_collective"
    else:
        retrieval_mode = "keyword_collective"

    merged_context = _merge_recommendation_results(
        keyword_results=keyword_results,
        personal_vector_results=personal_vector_results,
        collective_vector_results=collective_vector_results,
        intent=intent,
    )

    preference_context = build_preference_context(user_id, keywords)

    return {
        "user_profile": {
            "user_id": user_id,
            "name": user_profile.get("name"),
            "age_group": user_profile.get("age_group"),
            "default_address": default_address,
        },
        "preference_memory": preference_memory,
        "purchase_count": purchase_count,
        "retrieval_mode": retrieval_mode,
        "keyword_results": keyword_results,
        "personal_vector_results": personal_vector_results,
        "collective_vector_results": collective_vector_results,
        "merged_context": merged_context,
        "preference_context": preference_context,
    }


def _summarize_messages(messages: list) -> str:
    recent = messages[-5:] if len(messages) > 5 else messages
    parts = []
    for msg in recent:
        if isinstance(msg, dict):
            role, content = msg.get("role", ""), str(msg.get("content", ""))[:50]
        else:
            role = getattr(msg, "type", "") or getattr(msg, "role", "")
            content = str(getattr(msg, "content", ""))[:50]
        parts.append(f"[{role}] {content}")
    return " | ".join(parts)


def context_agent_node(state: ShoppingState, store: Optional[BaseStore] = None) -> dict:
    """
    Context Agent.

    [결제 완료 후] stage == "completed": 대화 요약 저장 후 종료.
    [Context Loading] buy/refine/compare_platforms: 프로필+선호도 컨텍스트 생성.
    """
    stage = state.get("stage")
    intent = state.get("intent")
    user_id = state.get("user_id", "")
    keywords = state.get("keywords") or []

    agent_logger.log(
        f"\n{'─'*40}\n[context_agent] 진입 | stage={stage}  intent={intent}  "
        f"user_id={user_id}  keywords={keywords}\n{'─'*40}"
    )

    updates: dict[str, Any] = {"last_agent": "context_agent"}

    if stage == "completed":
        messages = state.get("messages") or []
        if len(messages) > 10:
            updates["conversation_summary"] = _summarize_messages(messages)

        # Learning & Adaptation: 구매한 상품 기반으로 선호도 즉시 업데이트
        selected_product = state.get("selected_product") or {}
        keywords = state.get("keywords") or []
        if selected_product:
            mock_update_preference_from_purchase(user_id, selected_product, keywords)
            _preference_cache.pop(user_id, None)  # 캐시 무효화 → 다음 추천부터 반영
            agent_logger.log(f"[context_agent] 선호도 업데이트 완료 → 캐시 무효화 (user={user_id})")

        agent_logger.log("[context_agent] 결제 완료 → 대화 요약 저장")
        return updates

    recommendation_context = get_recommendation_context(
        user_id=user_id,
        keywords=keywords,
        intent=intent,
    )

    pref_ctx = recommendation_context.get("preference_context") or {}
    agent_logger.log_context_agent(
        {
            "stage": stage,
            "intent": intent,
            "user_id": user_id,
            "keywords": keywords,
            "purchase_count": recommendation_context.get("purchase_count", 0),
            "retrieval_mode": recommendation_context.get("retrieval_mode"),
            "cache_hit": pref_ctx.get("_cache_hit", False),
        },
        {"preference_context": pref_ctx},
    )

    if store is not None:
        store.put(("recommendation_context", user_id), "latest", recommendation_context)

    return {
        **updates,
        "recommendation_context": recommendation_context,
    }


def get_recommendation_context_from_store(
    user_id: str,
    store: Optional[BaseStore] = None,
) -> dict[str, Any]:
    if store is None:
        return {}
    item = store.get(("recommendation_context", user_id), "latest")
    if item is None:
        return {}
    return item.value if hasattr(item, "value") else {}
