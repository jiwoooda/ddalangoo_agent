"""
Reorder Agent Node.

역할:
  1. 구매이력에서 재구매 후보 탐색
  2. URL 검증 + 후보 선택 (기존 reorder_node 흡수)
  3. product_confirming 또는 product_agent(no_match)로 직접 반환

기존 memory_agent(reorder 분기) + reorder_node 통합.
"""
from typing import Any
from src.state.schema import ShoppingState
from src.tools.mock_tools import mock_get_purchase_history, mock_validate_product_url
from src.utils.agent_logger import agent_logger


def _get_latest_user_text(state: ShoppingState) -> str:
    for msg in reversed(state.get("messages") or []):
        if isinstance(msg, dict):
            content = msg.get("content")
            role = msg.get("role") or msg.get("type")
            if content and role in (None, "user", "human"):
                return str(content)
        else:
            content = getattr(msg, "content", None)
            msg_type = getattr(msg, "type", None) or getattr(msg, "role", None)
            if content and msg_type in (None, "human", "user"):
                return str(content)
    return ""


def _resolve_reorder_candidates(
    user_id: str,
    keywords: list[str],
    query: str,
    top_k: int = 5,
) -> dict[str, Any]:
    search_terms = keywords or [query]
    all_histories = mock_get_purchase_history(user_id)
    history = [
        h for h in all_histories
        if any(
            k.lower() in (h.get("product_name") or "").lower()
            or k.lower() in (h.get("keyword") or "").lower()
            or k.lower() in (h.get("category") or "").lower()
            for k in search_terms
        )
    ][:top_k]

    candidates = []
    for item in history:
        candidates.append({
            "purchase_history_id": item.get("id"),
            "product_id": item.get("product_id"),
            "product_option_id": item.get("product_option_id"),
            "product_name": item.get("product_name"),
            "product_url": item.get("product_url"),
            "option_text": item.get("option_text"),
            "selected_options": item.get("selected_options") or {},
            "price_at_purchase": item.get("price_at_purchase", 0),
            "platform": item.get("platform"),
            "purchased_at": item.get("purchased_at"),
            "score": 0.8,
        })

    if not candidates:
        return {"resolution_type": "no_match", "candidates": []}

    candidates.sort(key=lambda c: c.get("purchased_at") or "", reverse=True)

    seen_names: set[str] = set()
    distinct: list[dict] = []
    for c in candidates:
        name = (c.get("product_name") or "").strip()
        if name not in seen_names:
            seen_names.add(name)
            distinct.append(c)

    if len(distinct) >= 2:
        names = ", ".join(
            f"{i+1}. {c.get('product_name', '상품')}"
            for i, c in enumerate(distinct[:3])
        )
        return {
            "resolution_type": "ambiguous",
            "candidates": distinct[:3],
            "question": f"사신 적 있는 상품이 여러 개예요. 어떤 걸로 할까요? {names}",
        }

    return {"resolution_type": "resolved", "candidates": candidates, "selected": candidates[0]}


def _confirm_candidate(candidate: dict) -> dict:
    raw_url = candidate.get("product_url", "")
    product_url = raw_url if mock_validate_product_url(raw_url) else ""
    price = candidate.get("price_at_purchase", candidate.get("price", 0))
    product_name = candidate.get("product_name", "상품")
    selected_product = {
        "product_id": candidate.get("product_id"),
        "product_option_id": candidate.get("product_option_id"),
        "purchase_history_id": candidate.get("purchase_history_id"),
        "product_name": product_name,
        "brand": candidate.get("brand"),
        "category": candidate.get("category"),
        "price": price,
        "platform": candidate.get("platform", ""),
        "option_text": candidate.get("option_text"),
        "selected_options": candidate.get("selected_options") or {},
        "product_url": product_url,
    }
    return {
        "selected_product": selected_product,
        "product_url": product_url,
        "pending_action": {
            "type": "product_confirm",
            "message": f"{product_name}, {price:,}원이에요. 다시 주문할까요?",
            "payload": {
                "purchase_history_id": candidate.get("purchase_history_id"),
                "product_url": product_url,
                "selected_options": candidate.get("selected_options") or {},
                "option_text": candidate.get("option_text"),
            },
        },
        "stage": "product_confirming",
        "error": None,
        "last_agent": "reorder_agent",
    }


def _selection_terms(text: str, keywords: list[str]) -> list[str]:
    base = f"{text} {' '.join(keywords or [])}".lower().replace(" ", "")
    terms = [base] if base else []
    aliases = {
        "딸기": ["strawberry"],
        "신선": ["fresh"],
        "냉동": ["frozen"],
        "설향": ["설향"],
    }
    for key, values in aliases.items():
        if key in base:
            terms.extend(values)
    seen = set()
    return [t for t in terms if t and not (t in seen or seen.add(t))]


def _candidate_text(candidate: dict) -> str:
    return " ".join(
        str(candidate.get(field) or "")
        for field in ("product_name", "option_text", "brand", "category", "keyword")
    ).lower().replace(" ", "")


def _select_from_pending(state: ShoppingState) -> dict | None:
    pending = state.get("pending_action") or {}
    candidates = (pending.get("payload") or {}).get("candidates") or []
    if not candidates:
        return None

    user_text = _get_latest_user_text(state)
    digits = [ch for ch in user_text if ch.isdigit()]
    if digits:
        index = int(digits[0]) - 1
        if 0 <= index < len(candidates):
            return candidates[index]

    terms = _selection_terms(user_text, state.get("keywords") or [])
    scored = sorted(
        [(sum(1 for t in terms if t in _candidate_text(c)), c) for c in candidates],
        key=lambda x: x[0],
        reverse=True,
    )
    if scored and scored[0][0] > 0 and (len(scored) == 1 or scored[0][0] > scored[1][0]):
        return scored[0][1]
    return None


def reorder_agent_node(state: ShoppingState) -> dict:
    """
    Reorder Agent.

    [product_select pending]  사용자가 모호한 후보 중 선택 → 확인 단계로
    [새 reorder 요청]          구매이력 탐색 → resolved/ambiguous/no_match 처리
    """
    pending = state.get("pending_action") or {}
    user_id = state.get("user_id", "")
    keywords = state.get("keywords") or []

    agent_logger.log(
        f"\n{'─'*40}\n[reorder_agent] 진입 | pending_type={pending.get('type')}  "
        f"user_id={user_id}  keywords={keywords}\n{'─'*40}"
    )

    # 사용자가 모호한 후보 목록에서 선택하는 경우
    if pending.get("type") == "product_select":
        candidate = _select_from_pending(state)
        if candidate:
            agent_logger.log(f"[reorder_agent] product_select → 후보 선택됨: {candidate.get('product_name')}")
            output = _confirm_candidate(candidate)
            agent_logger.log_reorder_agent(
                {"pending_type": "product_select", "user_id": user_id, "keywords": keywords},
                {"resolution_type": "resolved", "selected_candidate": candidate,
                 "candidates": (pending.get("payload") or {}).get("candidates") or [],
                 "stage": output.get("stage"), "pending_action": output.get("pending_action")},
            )
            return output
        output = {
            "pending_action": {
                **pending,
                "message": "어떤 상품인지 모르겠어요. 번호로 다시 말씀해 주세요.",
            },
            "stage": "product_confirming",
            "error": None,
            "last_agent": "reorder_agent",
        }
        agent_logger.log_reorder_agent(
            {"pending_type": "product_select", "user_id": user_id, "keywords": keywords},
            {"resolution_type": "select_failed", "candidates": [],
             "stage": output.get("stage"), "pending_action": output.get("pending_action")},
        )
        return output

    # 새 reorder 요청: 구매이력 탐색
    query = _get_latest_user_text(state) or " ".join(keywords)
    result = _resolve_reorder_candidates(user_id, keywords, query)
    resolution_type = result["resolution_type"]

    if resolution_type == "no_match":
        output = {
            "stage": "searching",
            "error": "reorder_no_match",
            "search_results": [],
            "last_agent": "reorder_agent",
        }
        agent_logger.log_reorder_agent(
            {"pending_type": pending.get("type", "-"), "user_id": user_id, "keywords": keywords},
            {**result, "stage": output.get("stage"), "pending_action": None},
        )
        return output

    if resolution_type == "ambiguous":
        candidates = result["candidates"]
        output = {
            "pending_action": {
                "type": "product_select",
                "message": result.get("question", "이전에 사신 상품이 여러 개예요. 어떤 걸로 할까요?"),
                "payload": {"candidates": candidates},
            },
            "stage": "product_confirming",
            "error": None,
            "search_results": [],
            "last_agent": "reorder_agent",
        }
        agent_logger.log_reorder_agent(
            {"pending_type": pending.get("type", "-"), "user_id": user_id, "keywords": keywords},
            {**result, "stage": output.get("stage"), "pending_action": output.get("pending_action")},
        )
        return output

    # resolved
    output = _confirm_candidate(result["selected"])
    agent_logger.log_reorder_agent(
        {"pending_type": pending.get("type", "-"), "user_id": user_id, "keywords": keywords},
        {**result, "selected_candidate": result["selected"],
         "stage": output.get("stage"), "pending_action": output.get("pending_action")},
    )
    return output
