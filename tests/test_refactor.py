"""
리팩토링 후 핵심 플로우 검증 테스트.
LLM 호출 없이 라우터/검색/그래프/결제 mock만 확인한다.
"""
import pytest
from src.state.schema import get_default_shopping_state
from src.graph.router import (
    route,
    after_product_agent,
    after_response_agent,
    after_reorder_agent,
    after_context_agent,
    after_payment_agent,
    after_respond,
)
from src.tools.mock_search import search_products
from src.agents.nodes import cancel_node


# ── 라우터 ────────────────────────────────────────────────

def _state(**kwargs):
    s = get_default_shopping_state("user_test", "sess")
    s.update(kwargs)
    return s


def test_route_buy_goes_to_context():
    assert route(_state(intent="buy", confidence=0.95)) == "context_agent"

def test_route_reorder_goes_to_reorder():
    assert route(_state(intent="reorder", confidence=0.95)) == "reorder_agent"

def test_route_cancel():
    assert route(_state(intent="cancel", confidence=0.95)) == "cancel"

def test_route_unclear_goes_to_respond():
    assert route(_state(intent="unclear", confidence=0.3, needs_clarification=True)) == "respond"

def test_route_ask_goes_to_response_agent():
    assert route(_state(intent="ask", confidence=0.95)) == "response_agent"

def test_route_compare_platforms_goes_to_product_agent():
    assert route(_state(intent="compare_platforms", confidence=0.95)) == "product_agent"

def test_route_payment_processing():
    s = _state(intent="confirm", stage="payment_processing", confidence=0.95)
    assert route(s) == "payment_agent"

def test_route_product_confirming_confirm_with_quantity():
    s = _state(
        intent="confirm", stage="product_confirming", confidence=0.95,
        quantity=2, pending_action={"type": "product_confirm"},
    )
    assert route(s) == "payment_agent"

def test_route_product_confirming_next_goes_to_product():
    s = _state(intent="next", stage="product_confirming", confidence=0.95)
    assert route(s) == "product_agent"

def test_route_product_confirming_deny_goes_to_product():
    s = _state(intent="deny", stage="product_confirming", confidence=0.95)
    assert route(s) == "product_agent"


# ── platform_agent / memory_agent / reorder_node 참조 없음 확인 ──

def test_no_old_nodes_in_route():
    """route()가 삭제된 노드를 반환하지 않는다."""
    cases = [
        _state(intent="buy", confidence=0.95),
        _state(intent="reorder", confidence=0.95),
        _state(intent="refine", confidence=0.95),
        _state(intent="compare_platforms", confidence=0.95),
        _state(intent="next", stage="product_confirming", confidence=0.95),
        _state(intent="deny", stage="product_confirming", confidence=0.95),
    ]
    for s in cases:
        result = route(s)
        assert result not in ("platform_agent", "memory_agent", "reorder_node"), \
            f"삭제된 노드 '{result}' 반환됨"


def test_route_ask_at_product_confirming_goes_to_response_agent():
    """product_confirming에서 ask는 재검색 없이 response_agent로 직행해야 한다."""
    s = _state(intent="ask", stage="product_confirming", confidence=0.95)
    assert route(s) == "response_agent"


def test_route_deny_next_at_product_confirming_goes_to_product_agent():
    """deny/next는 product_agent로 가야 한다."""
    for intent in ("deny", "next"):
        s = _state(intent=intent, stage="product_confirming", confidence=0.95)
        assert route(s) == "product_agent"


def test_route_product_select_pending_goes_to_reorder_agent():
    """product_select pending에서 confirm은 reorder_agent로 가야 한다."""
    s = _state(
        intent="confirm",
        stage="product_confirming",
        confidence=0.95,
        pending_action={"type": "product_select", "payload": {"candidates": []}},
    )
    assert route(s) == "reorder_agent"


# ── after_* 라우터 ─────────────────────────────────────────

def test_after_product_agent_normal_goes_to_response():
    s = _state(stage="searching")
    assert after_product_agent(s) == "response_agent"

def test_after_product_agent_error_goes_to_respond():
    for err in ("invalid_keywords", "no_candidates", "no_relevant_products", "no_more_products"):
        s = _state(error=err)
        assert after_product_agent(s) == "respond"

def test_after_response_agent_always_respond():
    assert after_response_agent(_state()) == "respond"

def test_after_reorder_agent_no_match_goes_to_product():
    s = _state(error="reorder_no_match")
    assert after_reorder_agent(s) == "product_agent"

def test_after_reorder_agent_success_goes_to_respond():
    assert after_reorder_agent(_state()) == "respond"

def test_after_context_agent_buy_goes_to_product():
    assert after_context_agent(_state(intent="buy")) == "product_agent"

def test_after_context_agent_completed_goes_to_respond():
    assert after_context_agent(_state(stage="completed")) == "respond"

def test_after_payment_completed_goes_to_context():
    assert after_payment_agent(_state(stage="completed")) == "context_agent"

def test_after_respond_completed_ends():
    assert after_respond(_state(stage="completed")) == "end"

def test_after_respond_idle_continues():
    assert after_respond(_state(stage="idle")) == "wait_for_input"


# ── 멀티플랫폼 검색 ────────────────────────────────────────

def test_search_all_platforms_returns_multi_platform():
    results = search_products("딸기", ["naver", "coupang", "kurly"])
    platforms = {r["platform"] for r in results}
    assert len(platforms) > 1, f"단일 플랫폼만 반환됨: {platforms}"

def test_search_limit_per_platform():
    results = search_products("딸기", ["naver", "coupang", "kurly"], limit_per_platform=3)
    from collections import Counter
    counts = Counter(r["platform"] for r in results)
    for plat, count in counts.items():
        assert count <= 3, f"{plat} 결과가 {count}개로 3개 초과"

def test_search_no_platform_falls_back_to_default():
    results = search_products("딸기", [])
    assert len(results) > 0

def test_search_unknown_query_returns_default_products():
    results = search_products("존재하지않는상품xyz", ["naver", "coupang", "kurly"])
    assert len(results) > 0  # DEFAULT_PRODUCTS fallback


# ── 그래프 빌드 ────────────────────────────────────────────

def test_graph_builds_without_error():
    from src.graph.builder import build_graph
    graph = build_graph()
    assert graph is not None

def test_graph_has_no_old_nodes():
    from src.graph.builder import build_graph
    graph = build_graph()
    node_names = set(graph.get_graph().nodes.keys())
    for removed in ("platform_agent", "memory_agent", "reorder_node"):
        assert removed not in node_names, f"삭제된 노드 '{removed}'가 그래프에 존재함"

def test_graph_has_context_agent_node():
    from src.graph.builder import build_graph
    graph = build_graph()
    assert "context_agent" in set(graph.get_graph().nodes.keys())

def test_graph_has_reorder_agent_node():
    from src.graph.builder import build_graph
    graph = build_graph()
    assert "reorder_agent" in set(graph.get_graph().nodes.keys())

def test_graph_has_response_agent_node():
    from src.graph.builder import build_graph
    graph = build_graph()
    assert "response_agent" in set(graph.get_graph().nodes.keys())

def test_graph_has_product_agent_node():
    from src.graph.builder import build_graph
    graph = build_graph()
    assert "product_agent" in set(graph.get_graph().nodes.keys())


# ── cancel_node ────────────────────────────────────────────

def test_cancel_node_resets_stage():
    s = _state(stage="payment_processing", intent="cancel")
    result = cancel_node(s)
    assert result["stage"] == "idle"

def test_cancel_node_with_cart_items_keeps_cart_msg():
    s = _state(cart_items=[{"product_name": "딸기", "price": 9900, "quantity": 1, "total": 9900}])
    result = cancel_node(s)
    assert "장바구니" in result["pending_action"]["message"]


# ── mock 장바구니 / 결제 ───────────────────────────────────

def test_mock_add_to_cart_and_get_cart():
    from src.tools.mock_tools import mock_add_to_cart, mock_get_cart, mock_clear_cart
    mock_clear_cart("test_cart_user")
    product = {"product_name": "딸기 500g", "price": 9900, "platform": "kurly", "product_url": "https://mock.kurly.com/1"}
    mock_add_to_cart("test_cart_user", product, 2, ["딸기"])
    cart = mock_get_cart("test_cart_user")
    assert len(cart) == 1
    assert cart[0]["quantity"] == 2
    assert cart[0]["total"] == 19800
    mock_clear_cart("test_cart_user")

def test_mock_add_multiple_items():
    from src.tools.mock_tools import mock_add_to_cart, mock_get_cart, mock_clear_cart
    mock_clear_cart("test_multi_user")
    p1 = {"product_name": "딸기", "price": 9900, "platform": "kurly", "product_url": "https://mock.kurly.com/1"}
    p2 = {"product_name": "계란", "price": 4900, "platform": "coupang", "product_url": "https://mock.coupang.com/2"}
    mock_add_to_cart("test_multi_user", p1, 1)
    mock_add_to_cart("test_multi_user", p2, 2)
    cart = mock_get_cart("test_multi_user")
    assert len(cart) == 2
    assert sum(i["total"] for i in cart) == 9900 + 9800
    mock_clear_cart("test_multi_user")

def test_mock_place_order_clears_cart_and_saves_history():
    from src.tools.mock_tools import mock_add_to_cart, mock_get_cart, mock_place_order, mock_clear_cart, mock_get_purchase_history
    uid = "test_order_user"
    mock_clear_cart(uid)
    product = {"product_name": "사과 1.5kg", "price": 9900, "platform": "naver", "product_url": "https://mock.naver.com/1"}
    mock_add_to_cart(uid, product, 1, ["사과"])
    order = mock_place_order(uid, {"address_line1": "서울 강남구 테헤란로 1"})
    assert order["order_id"].startswith("ORDER-")
    assert order["total"] == 9900
    assert mock_get_cart(uid) == []
    history = mock_get_purchase_history(uid)
    assert any(h["order_id"] == order["order_id"] for h in history)

def test_payment_agent_add_to_cart_step():
    from src.payment.subgraph import payment_agent_node
    from src.tools.mock_tools import mock_clear_cart, mock_get_cart
    uid = "test_payment_user"
    mock_clear_cart(uid)
    product = {"product_name": "딸기 500g", "price": 9900, "platform": "kurly", "product_url": "https://mock.kurly.com/1", "delivery": "새벽배송"}
    s = _state(
        stage="product_confirming",
        intent="confirm",
        user_id=uid,
        selected_product=product,
        quantity=2,
        keywords=["딸기"],
        pending_action={"type": "product_confirm"},
    )
    result = payment_agent_node(s)
    assert result["stage"] == "cart_shopping"
    assert "담았어요" in result["pending_action"]["message"]
    cart = mock_get_cart(uid)
    assert len(cart) == 1
    assert cart[0]["quantity"] == 2
    mock_clear_cart(uid)

def test_payment_agent_quantity_change_intent():
    """intent=quantity_change도 장바구니 담기를 트리거해야 한다."""
    from src.payment.subgraph import payment_agent_node
    from src.tools.mock_tools import mock_clear_cart, mock_get_cart
    uid = "test_qty_change_user"
    mock_clear_cart(uid)
    product = {"product_name": "딸기", "price": 9900, "platform": "kurly", "product_url": "https://mock.kurly.com/1", "delivery": "새벽배송"}
    s = _state(
        stage="product_confirming",
        intent="quantity_change",
        user_id=uid,
        selected_product=product,
        quantity=3,
        keywords=["딸기"],
        pending_action={"type": "quantity_confirm"},
    )
    result = payment_agent_node(s)
    assert result["stage"] == "cart_shopping"
    assert mock_get_cart(uid)[0]["quantity"] == 3
    mock_clear_cart(uid)


def test_payment_agent_full_flow():
    from src.payment.subgraph import payment_agent_node
    from src.tools.mock_tools import mock_clear_cart, mock_add_to_cart
    uid = "test_full_payment_user"
    mock_clear_cart(uid)
    product = {"product_name": "계란 10구", "price": 4900, "platform": "coupang", "product_url": "https://mock.coupang.com/1", "delivery": "로켓배송"}
    mock_add_to_cart(uid, product, 2, ["계란"])

    base = dict(_state(user_id=uid, selected_product=product, quantity=2, keywords=["계란"], address_text="서울 강남구 테헤란로 1"))

    # Step 1
    s1 = {**base, "stage": "cart_shopping", "pending_action": None, "intent": "confirm"}
    r1 = payment_agent_node(s1)
    assert r1["stage"] == "payment_processing"
    assert r1["pending_action"]["type"] == "payment_method_confirm"

    # Step 2
    s2 = {**base, "stage": "payment_processing", "pending_action": {"type": "payment_method_confirm"}, "intent": "confirm"}
    r2 = payment_agent_node(s2)
    assert r2["pending_action"]["type"] == "address_confirm"

    # Step 3
    s3 = {**base, "stage": "payment_processing", "pending_action": {"type": "address_confirm"}, "intent": "confirm"}
    r3 = payment_agent_node(s3)
    assert r3["pending_action"]["type"] == "payment_password"

    # Step 4
    mock_clear_cart(uid)
    mock_add_to_cart(uid, product, 2, ["계란"])
    s4 = {**base, "stage": "payment_processing", "pending_action": {"type": "payment_password"}, "intent": "confirm"}
    r4 = payment_agent_node(s4)
    assert r4["stage"] == "completed"
    assert r4["order_id"].startswith("ORDER-")
    assert r4["cart_items"] == []


# ── reorder_agent 단독 테스트 ──────────────────────────────

def test_reorder_agent_no_match():
    from src.agents.reorder_agent import reorder_agent_node
    s = _state(intent="reorder", keywords=["없는상품xyz"], user_id="reorder_test_user")
    result = reorder_agent_node(s)
    assert result["error"] == "reorder_no_match"
    assert result["stage"] == "searching"

def test_reorder_agent_product_select_no_candidates():
    from src.agents.reorder_agent import reorder_agent_node
    s = _state(
        intent="confirm",
        stage="product_confirming",
        pending_action={"type": "product_select", "payload": {"candidates": []}},
    )
    result = reorder_agent_node(s)
    assert result["stage"] == "product_confirming"
    assert "번호" in result["pending_action"]["message"]


# ── context_agent 단독 테스트 ──────────────────────────────

def test_context_agent_completed_saves_summary():
    from src.agents.context_agent import context_agent_node
    from langchain_core.messages import HumanMessage
    msgs = [HumanMessage(content=f"msg{i}") for i in range(12)]
    s = _state(stage="completed", user_id="ctx_test_user", messages=msgs)
    result = context_agent_node(s)
    assert result["last_agent"] == "context_agent"
    assert "conversation_summary" in result

def test_context_agent_buy_loads_context():
    from src.agents.context_agent import context_agent_node
    s = _state(intent="buy", keywords=["딸기"], user_id="ctx_test_user2")
    result = context_agent_node(s)
    assert result["last_agent"] == "context_agent"
    assert "recommendation_context" in result
