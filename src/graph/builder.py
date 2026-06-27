"""
Orchestrator Graph Builder.

LangGraph StateGraph 구성:
- wait_for_input → intent_agent → route() → {agents} → respond → after_respond()
- interrupt_before=["wait_for_input"] (human-in-the-loop)
- MemorySaver / InMemoryStore (standalone 기본; production: PostgresSaver/PostgresStore)
"""
from typing import Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore

from src.state.schema import ShoppingState
from src.graph.router import (
    route,
    after_respond,
    after_context_agent,
    after_reorder_agent,
    after_product_agent,
    after_response_agent,
    after_recipe_agent,
    after_payment_agent,
)
from src.agents.intent_agent import intent_agent_node
from src.agents.context_agent import context_agent_node
from src.agents.reorder_agent import reorder_agent_node
from src.agents.product_agent import product_agent_node
from src.agents.response_agent import response_agent_node
from src.agents.nodes import wait_for_input_node, respond_node, cancel_node, ask_what_to_buy_node
from src.agents.recipe_agent import recipe_agent_node
from src.payment.subgraph import payment_agent_node


def build_graph(checkpointer=None, store: Optional[BaseStore] = None):
    """
    LangGraph Orchestrator Graph 구성.

    Parameters
    ----------
    checkpointer : 체크포인터 (기본: MemorySaver — in-memory)
    store : KV 스토어 (기본: InMemoryStore)
    """
    if checkpointer is None:
        checkpointer = MemorySaver()
    if store is None:
        store = InMemoryStore()

    def _context_agent_node(state: ShoppingState) -> dict:
        return context_agent_node(state, store=store)

    builder = StateGraph(ShoppingState)

    builder.add_node("wait_for_input", wait_for_input_node)
    builder.add_node("intent_agent", intent_agent_node)
    builder.add_node("context_agent", _context_agent_node)
    builder.add_node("reorder_agent", reorder_agent_node)
    builder.add_node("product_agent", product_agent_node)
    builder.add_node("response_agent", response_agent_node)
    builder.add_node("recipe_agent", recipe_agent_node)
    builder.add_node("payment_agent", payment_agent_node)
    builder.add_node("respond", respond_node)
    builder.add_node("ask_what_to_buy", ask_what_to_buy_node)
    builder.add_node("cancel", cancel_node)

    builder.set_entry_point("wait_for_input")
    builder.add_edge("wait_for_input", "intent_agent")

    builder.add_conditional_edges(
        "intent_agent",
        route,
        {
            "context_agent": "context_agent",
            "reorder_agent": "reorder_agent",
            "product_agent": "product_agent",
            "response_agent": "response_agent",
            "recipe_agent": "recipe_agent",
            "payment_agent": "payment_agent",
            "ask_what_to_buy": "ask_what_to_buy",
            "respond": "respond",
            "cancel": "cancel",
            "end": END,
        },
    )

    builder.add_conditional_edges(
        "context_agent",
        after_context_agent,
        {"product_agent": "product_agent", "respond": "respond"},
    )

    builder.add_conditional_edges(
        "reorder_agent",
        after_reorder_agent,
        {"respond": "respond", "product_agent": "product_agent"},
    )

    builder.add_edge("ask_what_to_buy", "respond")

    builder.add_conditional_edges(
        "product_agent",
        after_product_agent,
        {"response_agent": "response_agent", "respond": "respond"},
    )
    builder.add_conditional_edges(
        "response_agent",
        after_response_agent,
        {"respond": "respond"},
    )

    builder.add_conditional_edges(
        "recipe_agent",
        after_recipe_agent,
        {"context_agent": "context_agent", "respond": "respond"},
    )

    builder.add_conditional_edges(
        "payment_agent",
        after_payment_agent,
        {"context_agent": "context_agent", "recipe_agent": "recipe_agent", "respond": "respond"},
    )
    builder.add_edge("cancel", "respond")

    builder.add_conditional_edges(
        "respond",
        after_respond,
        {"wait_for_input": "wait_for_input", "end": END},
    )

    return builder.compile(
        checkpointer=checkpointer,
        store=store,
        interrupt_before=["wait_for_input"],
    )


def create_default_graph():
    return build_graph()
