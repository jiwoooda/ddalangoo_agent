"""
Product Agent Node.

역할: 멀티플랫폼 검색(tool) + 랭킹.
- naver/coupang/kurly 동시 검색 → 후보군 수집
- rank_products tool 호출 → 순위화
- 설명 생성은 Response Agent에서 담당

next/deny 재호출 시 검색 스킵 — 기존 recommended_products 재사용.
"""
import json
from typing import Any
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from configs.llm_config import get_llm
from langchain_core.tools import tool as lc_tool

from src.state.schema import ShoppingState
from src.tools.mock_search import search_products
from src.prompts.product_prompt import PRODUCT_RANK_PROMPT
from src.utils.agent_logger import agent_logger

ALL_PLATFORMS = ["naver", "coupang", "kurly"]


def _no_results_message(keywords: list[str]) -> str:
    label = keywords[0] if keywords else None
    if label:
        return f"{label}를 찾지 못했어요. 다른 상품을 말씀해 주세요."
    return "찾으시는 상품이 없어요. 다른 상품을 말씀해 주세요."


def _missing_product_message(keywords: list[str]) -> str:
    return _no_results_message(keywords)


def _no_more_products_message(keywords: list[str]) -> str:
    label = keywords[0] if keywords else None
    if label:
        return f"{label}로는 더 추천할 상품이 없어요. 다른 상품을 찾아볼까요?"
    return "더 추천할 상품이 없어요. 다른 상품을 찾아볼까요?"

CONDITION_MAP = {
    "최저가": "price_asc",
    "가성비": "value",
    "빠른배송": "delivery_fast",
    "인기순": "popularity",
    "리뷰좋은": "review_score",
    "무료배송": "free_shipping",
}

_llm: BaseChatModel | None = None


def _get_llm() -> BaseChatModel:
    global _llm
    if _llm is None:
        _llm = get_llm("product", temperature=0, max_tokens=800)
    return _llm


def _format_products(products: list[dict[str, Any]]) -> str:
    if not products:
        return "후보 상품 없음"
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    lines = []
    for i, p in enumerate(products):
        label = labels[i] if i < len(labels) else str(i)
        name = p.get("product_name", "이름 없음")
        price = p.get("price")
        price_str = f"{price:,}원" if price else "가격 미확인"
        delivery = p.get("delivery") or ""
        rating = p.get("rating")
        review = p.get("review_count")
        platform = p.get("platform", "")

        parts = [f"[{label}] {name}", price_str]
        if delivery:
            parts.append(str(delivery))
        if rating:
            parts.append(f"⭐{rating}")
        if review:
            parts.append(f"리뷰 {review:,}개")
        if platform:
            parts.append(f"({platform})")
        lines.append("  ".join(parts))
    return "\n".join(lines)


def _format_preference(preference_context: dict[str, Any]) -> str:
    if not preference_context:
        return "선호 정보 없음 (구매이력 부족)"
    lines = []
    summary = preference_context.get("summary") or ""
    if summary:
        lines.append(summary)

    preferred_brands = preference_context.get("preferred_brands") or []
    if preferred_brands:
        brand_names = [
            str(item.get("brand"))
            for item in preferred_brands
            if isinstance(item, dict) and item.get("brand")
        ]
        if brand_names:
            lines.append("선호 브랜드: " + ", ".join(brand_names[:5]))

    price_range = preference_context.get("price_range") or {}
    if price_range:
        avg = price_range.get("avg")
        min_price = price_range.get("min")
        max_price = price_range.get("max")
        if avg:
            if min_price and max_price:
                lines.append(f"선호 가격대: 평균 {avg:,}원 (범위 {min_price:,}~{max_price:,}원)")
            else:
                lines.append(f"선호 가격대: 평균 {avg:,}원")

    repurchase_patterns = preference_context.get("repurchase_patterns") or []
    if repurchase_patterns:
        lines.append("재구매 패턴: " + ", ".join(map(str, repurchase_patterns[:5])))

    preferred_platform = preference_context.get("preferred_platform")
    if preferred_platform:
        lines.append(f"선호 쇼핑몰: {preferred_platform}")

    keyword_summary = preference_context.get("keyword_summary") or ""
    if keyword_summary:
        lines.append(f"키워드 관련 선호: {keyword_summary}")
    return "\n".join(lines) if lines else "선호 정보 없음 (구매이력 부족)"


def _make_rank_tool(candidates: list[dict[str, Any]]):
    label_map = {chr(ord("A") + i): p for i, p in enumerate(candidates[:26])}

    @lc_tool
    def rank_products(ranked_labels: list[str], filtered_out_labels: list[str] = []) -> str:
        """
        후보 상품을 순위대로 정렬한다.
        ranked_labels: 1위부터 순서대로 레이블 리스트 (예: ["C", "A", "B"])
        filtered_out_labels: 키워드와 상품군이 달라 제외할 레이블 (예: ["D", "E"])
        """
        ranked = [
            label_map[lbl.upper()]
            for lbl in ranked_labels
            if lbl.upper() in label_map
        ]
        return json.dumps(ranked, ensure_ascii=False)

    return rank_products


def _filter_results(
    products: list[dict[str, Any]],
    exclude_keywords: list[str],
) -> list[dict[str, Any]]:
    filtered = []
    for p in products:
        if p.get("is_sold_out"):
            continue
        if not p.get("product_url"):
            continue
        if p.get("price") is None:
            continue
        name = p.get("product_name", "").lower()
        if any(ex.lower() in name for ex in exclude_keywords):
            continue
        filtered.append(p)
    return filtered


def _rank(
    candidates: list[dict[str, Any]],
    keywords: list[str],
    condition: str | None,
    preference_context: dict[str, Any],
) -> list[dict[str, Any]]:
    return _rank_with_metadata(candidates, keywords, condition, preference_context)["ranked_products"]


def _rank_with_metadata(
    candidates: list[dict[str, Any]],
    keywords: list[str],
    condition: str | None,
    preference_context: dict[str, Any],
) -> dict[str, Any]:
    rank_tool = _make_rank_tool(candidates)
    rank_response = _get_llm().bind_tools([rank_tool]).invoke([
        HumanMessage(content=PRODUCT_RANK_PROMPT.format(
            formatted_products=_format_products(candidates),
            preference_context=_format_preference(preference_context),
            keywords=json.dumps(keywords, ensure_ascii=False),
            condition=condition or "없음",
        ))
    ])
    if rank_response.tool_calls:
        tc = rank_response.tool_calls[0]
        try:
            result = json.loads(rank_tool.invoke(tc["args"]))
            if result:
                return {
                    "ranked_products": result,
                    "tool_call_success": True,
                    "tool_call_error": None,
                }
        except Exception as e:
            agent_logger.log(f"[product_agent] rank_tool 실패: {e}")
            return {
                "ranked_products": candidates,
                "tool_call_success": False,
                "tool_call_error": str(e),
            }
    return {
        "ranked_products": candidates,
        "tool_call_success": False,
        "tool_call_error": "no_tool_call",
    }


def product_agent_node(state: ShoppingState) -> dict:
    intent = state.get("intent")
    keywords = state.get("keywords") or []
    exclude_keywords = state.get("exclude_keywords") or []
    condition = state.get("condition")
    current_idx = state.get("current_product_index") or 0
    existing_ranked = state.get("recommended_products") or []
    recommendation_context = state.get("recommendation_context") or {}
    preference_context = recommendation_context.get("preference_context") or {}

    if not keywords or all(k in ["그거", "저번에", "그것", "저것"] for k in keywords):
        return {
            "search_results": [],
            "stage": "idle",
            "error": "invalid_keywords",
            "last_agent": "product_agent",
            "pending_action": {"type": "clarification", "message": _no_results_message([])},
        }

    # ── next/deny: 재검색 없이 다음 후보 ──
    if intent in ("next", "deny") and existing_ranked:
        if condition:
            reranked = _rank(existing_ranked, keywords, condition, preference_context)
            top_product = reranked[0] if reranked else None
            if not top_product:
                return {
                    "stage": "idle",
                    "error": "no_relevant_products",
                    "last_agent": "product_agent",
                    "pending_action": {"type": "clarification", "message": _missing_product_message(keywords)},
                }
            agent_logger.log_product_agent(
                {"intent": intent, "rerank": True, "condition": condition},
                {"selected_product": top_product},
            )
            return {
                "selected_product": top_product,
                "product_url": top_product.get("product_url"),
                "recommended_products": reranked,
                "current_product_index": 0,
                "stage": "searching",
                "quantity": None,
                "last_agent": "product_agent",
                "error": None,
            }

        next_idx = current_idx + 1
        if next_idx >= len(existing_ranked):
            return {
                "stage": "searching",
                "error": "no_more_products",
                "last_agent": "product_agent",
                "pending_action": {
                    "type": "no_more_products",
                    "message": _no_more_products_message(keywords),
                },
            }
        next_product = existing_ranked[next_idx]
        agent_logger.log_product_agent(
            {"intent": intent, "next_idx": next_idx},
            {"selected_product": next_product},
        )
        return {
            "selected_product": next_product,
            "product_url": next_product.get("product_url"),
            "recommended_products": existing_ranked,
            "current_product_index": next_idx,
            "stage": "searching",
            "quantity": None,
            "last_agent": "product_agent",
            "error": None,
        }

    # ── 전체 플랫폼 동시 검색 ──
    query = " ".join(keywords)
    sort = CONDITION_MAP.get(condition, "relevance") if condition else "relevance"

    agent_logger.log(f"[product_agent] 검색 | query={query}  platforms={ALL_PLATFORMS}")
    raw_results = search_products(query=query, platforms=ALL_PLATFORMS, condition=sort)
    candidates = _filter_results(raw_results, exclude_keywords)

    if not candidates:
        return {
            "stage": "idle",
            "error": "no_candidates",
            "last_agent": "product_agent",
            "pending_action": {"type": "clarification", "message": _missing_product_message(keywords)},
        }

    agent_logger.log(f"[product_agent] 랭킹 | 후보 {len(candidates)}개")
    ranked_products = _rank(candidates, keywords, condition, preference_context)

    top_product = ranked_products[0] if ranked_products else None
    if not top_product:
        return {
            "stage": "idle",
            "error": "no_relevant_products",
            "last_agent": "product_agent",
            "pending_action": {"type": "clarification", "message": _missing_product_message(keywords)},
        }

    agent_logger.log_product_agent(
        {"intent": intent, "candidates": len(candidates), "ranked": len(ranked_products)},
        {"selected_product": top_product},
    )
    return {
        "search_results": candidates,
        "selected_product": top_product,
        "product_url": top_product.get("product_url"),
        "recommended_products": ranked_products,
        "current_product_index": 0,
        "stage": "searching",
        "quantity": None,
        "last_agent": "product_agent",
        "error": None,
    }
