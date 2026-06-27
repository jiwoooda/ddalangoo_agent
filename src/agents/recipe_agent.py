"""
Recipe Agent Node.

역할:
  Mode 1 - 재료 목록 생성: 요리명+인원수 → LLM structured output → ingredient_confirm
  Mode 2 - 재료 제거 편집: 사용자가 특정 재료 제외 요청 → 목록 업데이트 후 재확인
  Mode 3 - 쇼핑 시작: 현재 재료 keywords/quantity 세팅 → context_agent 진입
  Mode 4 - 다음 재료 안내: 장바구니 담기 후 자동 진입 → 다음 재료 안내 or 전체 완료
"""
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage

from configs.llm_config import get_llm
from src.state.schema import ShoppingState
from src.prompts.recipe_prompt import RECIPE_GENERATE_PROMPT
from src.utils.agent_logger import agent_logger


class RecipeItem(BaseModel):
    name: str = Field(description="재료명 (예: 된장, 두부, 애호박)")
    quantity: int = Field(description="필요 수량 (양의 정수)")
    unit: str = Field(description="단위 (개, 통, 모, 단, g 등)")


class RecipeOutput(BaseModel):
    items: list[RecipeItem] = Field(description="주요 재료 목록 5~7가지")


_llm = None
_structured_llm = None


def _get_llm():
    global _llm, _structured_llm
    if _llm is None:
        _llm = get_llm("recipe", temperature=0.2)
        _structured_llm = _llm.with_structured_output(RecipeOutput)
    return _structured_llm


def _extract_user_input(state: ShoppingState) -> str:
    for msg in reversed(state.get("messages") or []):
        if isinstance(msg, dict):
            if msg.get("role") == "user":
                return msg.get("content", "")
        else:
            role = getattr(msg, "type", None) or getattr(msg, "role", None)
            if role == "human":
                return getattr(msg, "content", "")
    return ""


def _generate_items(dish: str, people: int) -> list[dict]:
    prompt = RECIPE_GENERATE_PROMPT.format(dish=dish, people=people)
    try:
        result: RecipeOutput = _get_llm().invoke([SystemMessage(content=prompt)])
        return [item.model_dump() for item in result.items]
    except Exception as e:
        agent_logger.log(f"[recipe_agent] 재료 생성 오류: {e}")
        return []


def _remove_ingredients(user_message: str, items: list[dict]) -> list[dict]:
    return [item for item in items if item["name"] not in user_message]


def _format_list_message(dish: str, people: Optional[int], items: list[dict]) -> str:
    people_str = f"{people}인 기준 " if people else ""
    lines = [f"{people_str}{dish} 재료예요:"]
    for item in items:
        lines.append(f"  {item['name']} {item['quantity']}{item['unit']}")
    lines.append("빼실 게 있으면 말씀해 주세요.")
    return "\n".join(lines)


def recipe_agent_node(state: ShoppingState) -> dict:
    stage = state.get("stage")
    intent = state.get("intent")
    recipe_dish = state.get("recipe_dish") or ""
    recipe_people = state.get("recipe_people") or 4
    recipe_items = list(state.get("recipe_items") or [])
    current_idx = state.get("current_recipe_item_index") or 0
    user_message = _extract_user_input(state)

    agent_logger.log(
        f"[recipe_agent] 진입 | stage={stage} intent={intent} "
        f"dish={recipe_dish} idx={current_idx}/{len(recipe_items)}"
    )

    # ── Mode 4: 다음 재료 안내 (payment_agent 장바구니 담기 후 자동 진입) ──
    if stage == "cart_shopping" and recipe_items:
        next_idx = current_idx + 1
        if next_idx >= len(recipe_items):
            output = {
                "current_recipe_item_index": next_idx,
                "stage": "cart_shopping",
                "keywords": [],
                "quantity": None,
                "pending_action": {
                    "type": "continue_shopping",
                    "message": "모든 재료를 담았어요! 결제하실까요?",
                },
                "last_agent": "recipe_agent",
                "error": None,
            }
        else:
            current_item = recipe_items[current_idx]
            next_item = recipe_items[next_idx]
            output = {
                "current_recipe_item_index": next_idx,
                "stage": "recipe_planning",
                "keywords": [],
                "quantity": None,
                "pending_action": {
                    "type": "ingredient_confirm",
                    "message": (
                        f"{current_item['name']} 담았어요! "
                        f"다음은 {next_item['name']} {next_item['quantity']}{next_item['unit']}이에요. "
                        f"찾아볼까요?"
                    ),
                },
                "last_agent": "recipe_agent",
                "error": None,
            }
        agent_logger.log(f"[recipe_agent] Mode 4 | {output['pending_action']['message']}")
        return output

    # ── Mode 3: 재료 확정 → 현재 재료 쇼핑 시작 ──
    if stage == "recipe_planning" and intent == "confirm" and recipe_items:
        item = recipe_items[current_idx]
        output = {
            "intent": "buy",
            "keywords": [item["name"]],
            "quantity": item.get("quantity"),
            "stage": "idle",
            "last_agent": "recipe_agent",
            "error": None,
        }
        agent_logger.log(f"[recipe_agent] Mode 3 | {item['name']} 쇼핑 시작")
        return output

    # ── Mode 2: 재료 제거 편집 ──
    if stage == "recipe_planning" and recipe_items:
        updated = _remove_ingredients(user_message, recipe_items)
        msg = _format_list_message(recipe_dish, recipe_people, updated if updated != recipe_items else recipe_items)
        output = {
            "recipe_items": updated,
            "stage": "recipe_planning",
            "pending_action": {"type": "ingredient_confirm", "message": msg},
            "last_agent": "recipe_agent",
            "error": None,
        }
        agent_logger.log(f"[recipe_agent] Mode 2 | {len(recipe_items)}→{len(updated)}개")
        return output

    # ── Mode 1: 재료 목록 생성 ──
    if not recipe_dish:
        return {
            "stage": "idle",
            "needs_clarification": True,
            "clarification_reason": "어떤 요리의 재료를 찾으시나요?",
            "last_agent": "recipe_agent",
            "error": "missing_recipe_dish",
        }

    items = _generate_items(recipe_dish, recipe_people)
    if not items:
        return {
            "stage": "idle",
            "error": "recipe_generation_failed",
            "last_agent": "recipe_agent",
        }

    msg = _format_list_message(recipe_dish, recipe_people, items)
    output = {
        "recipe_items": items,
        "current_recipe_item_index": 0,
        "stage": "recipe_planning",
        "pending_action": {"type": "ingredient_confirm", "message": msg},
        "last_agent": "recipe_agent",
        "error": None,
    }
    agent_logger.log(f"[recipe_agent] Mode 1 | {recipe_dish} 재료 {len(items)}개 생성")
    return output
