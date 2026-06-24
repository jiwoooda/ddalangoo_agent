"""구매이력 조회 Tool — mock_get_purchase_history 래핑."""
import json
from typing import Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from src.shared import mock_get_purchase_history, mock_get_preference_memory


class HistoryInput(BaseModel):
    user_id: str = Field(description="사용자 ID")
    keywords: str = Field(default="", description="필터할 키워드 콤마 구분 (비우면 전체 조회)")


class PurchaseHistoryTool(BaseTool):
    name: str = "get_purchase_history"
    description: str = (
        "사용자의 과거 구매이력을 조회합니다. "
        "keywords를 지정하면 해당 키워드가 포함된 구매이력만 반환합니다. "
        "재구매 후보 탐색 및 선호도 파악에 활용합니다."
    )
    args_schema: Type[BaseModel] = HistoryInput

    def _run(self, user_id: str, keywords: str = "") -> str:
        all_history = mock_get_purchase_history(user_id)
        if keywords:
            kw_list = [k.strip().lower() for k in keywords.split(",") if k.strip()]
            filtered = [
                h for h in all_history
                if any(
                    k in (h.get("product_name") or "").lower()
                    or k in (h.get("keyword") or "").lower()
                    or k in (h.get("category") or "").lower()
                    for k in kw_list
                )
            ]
        else:
            filtered = all_history

        return json.dumps({
            "user_id": user_id,
            "total_count": len(all_history),
            "filtered_count": len(filtered),
            "history": filtered[:5],
        }, ensure_ascii=False, indent=2)


class PreferenceInput(BaseModel):
    user_id: str = Field(description="사용자 ID")


class PreferenceMemoryTool(BaseTool):
    name: str = "get_preference_memory"
    description: str = (
        "사용자의 선호도 메모리를 조회합니다. "
        "선호 플랫폼, 가격대, 브랜드, 최근 키워드, 선호 배송 방식이 포함됩니다."
    )
    args_schema: Type[BaseModel] = PreferenceInput

    def _run(self, user_id: str) -> str:
        pref = mock_get_preference_memory(user_id)
        return json.dumps(pref, ensure_ascii=False, indent=2)
