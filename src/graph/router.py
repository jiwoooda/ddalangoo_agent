from typing import Literal
from src.state.schema import ShoppingState
from src.utils.agent_logger import agent_logger, _ptype

RouteName = Literal[
    "context_agent",
    "reorder_agent",
    "product_agent",
    "response_agent",
    "payment_agent",
    "quantity_check",
    "ask_what_to_buy",
    "respond",
    "cancel",
    "end",
]


def route(state: ShoppingState) -> RouteName:
    """
    Intent + Stage 기반 라우팅.
    1. clarification 우선
    2. cancel 우선
    3. payment_processing이면 Payment로 위임
    4. product_confirming에서 intent별 분기
    5. idle/searching에서 intent 기반 분기
    """
    intent = state.get("intent")
    stage = state.get("stage", "idle")
    confidence = state.get("confidence") or 0.0
    needs_clarification = state.get("needs_clarification", False)
    pending_type = _ptype(state.get("pending_action"))

    def _decide(dest: RouteName) -> RouteName:
        agent_logger.log_router("intent_agent", dest, intent or "-", stage, pending_type)
        return dest

    if needs_clarification or confidence < 0.5 or intent == "unclear":
        return _decide("respond")

    if intent == "cancel":
        return _decide("cancel")

    if stage == "payment_processing":
        return _decide("payment_agent")

    if stage == "cart_shopping":
        if intent in ("buy", "reorder", "refine", "compare_platforms"):
            if intent == "reorder":
                return _decide("reorder_agent")
            return _decide("product_agent")
        if pending_type == "what_to_buy":
            if intent == "reorder":
                return _decide("reorder_agent")
            if intent in ("buy", "refine", "compare_platforms"):
                return _decide("product_agent")
            if intent == "confirm":
                return _decide("payment_agent")
            return _decide("respond")
        if intent == "confirm":
            return _decide("payment_agent")
        if intent in ("deny", "next"):
            return _decide("ask_what_to_buy")
        return _decide("respond")

    if stage == "product_confirming":
        pa_type = (state.get("pending_action") or {}).get("type")

        if pa_type == "quantity_confirm":
            if state.get("quantity"):
                return _decide("payment_agent")
            if intent in ("confirm", "quantity_change"):
                return _decide("quantity_check")

        if pa_type == "product_select":
            if intent in ("confirm", "option_select"):
                return _decide("reorder_agent")
            return _decide("respond")

        if pa_type == "price_change_confirm":
            if intent in ("confirm", "deny", "cancel", "next"):
                return _decide("payment_agent")
            return _decide("respond")

        if intent == "confirm":
            if not state.get("quantity"):
                return _decide("quantity_check")
            return _decide("payment_agent")

        if intent in ("buy", "reorder"):
            if intent == "reorder":
                return _decide("reorder_agent")
            return _decide("product_agent")

        if intent in ("deny", "next"):
            return _decide("product_agent")

        if intent == "ask":
            return _decide("response_agent")

        if intent in ("refine", "compare_platforms"):
            return _decide("product_agent")

        return _decide("respond")

    if stage == "searching":
        if intent in ("refine", "compare_platforms"):
            return _decide("product_agent")
        if intent == "ask":
            return _decide("response_agent")
        return _decide("respond")

    routing_map: dict[str, RouteName] = {
        "buy": "context_agent",
        "reorder": "reorder_agent",
        "compare_platforms": "product_agent",
        "refine": "product_agent",
        "ask": "response_agent",
        "next": "product_agent",
        "confirm": "respond",
        "deny": "respond",
        "option_select": "respond",
        "quantity_change": "respond",
        "address_change": "respond",
    }
    return _decide(routing_map.get(intent, "respond"))


def after_product_agent(state: ShoppingState) -> Literal["response_agent", "respond"]:
    if state.get("error") in ("invalid_keywords", "no_candidates", "no_relevant_products", "no_more_products"):
        return "respond"
    return "response_agent"


def after_response_agent(state: ShoppingState) -> Literal["respond"]:
    return "respond"


def after_reorder_agent(state: ShoppingState) -> Literal["respond", "product_agent"]:
    if state.get("error") == "reorder_no_match":
        return "product_agent"
    return "respond"


def after_context_agent(state: ShoppingState) -> Literal["product_agent", "respond"]:
    stage = state.get("stage")
    if stage == "completed":
        return "respond"
    intent = state.get("intent")
    pending_type = _ptype(state.get("pending_action"))
    dest = "product_agent" if intent in ("buy", "refine", "compare_platforms") else "respond"
    agent_logger.log_router("context_agent", dest, intent or "-", stage or "-", pending_type)
    return dest


def after_payment_agent(state: ShoppingState) -> Literal["context_agent", "respond"]:
    stage = state.get("stage", "idle")
    pending_type = _ptype(state.get("pending_action"))
    intent = state.get("intent") or "-"

    def _decide(dest):
        agent_logger.log_router("payment_agent", dest, intent, stage, pending_type)
        return dest

    if stage == "completed":
        return _decide("context_agent")
    return _decide("respond")


def after_respond(state: ShoppingState) -> Literal["wait_for_input", "end"]:
    stage = state.get("stage", "idle")
    error = state.get("error")

    if stage in ("completed", "failed"):
        return "end"

    if error and "fatal" in error.lower():
        return "end"

    return "wait_for_input"
