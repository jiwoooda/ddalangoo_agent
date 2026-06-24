"""선호도 업데이트 Tool — mock_update_preference_from_purchase 래핑."""
import json
from typing import Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from src.shared import mock_update_preference_from_purchase


class UpdatePreferenceInput(BaseModel):
    user_id: str = Field(description="사용자 ID")
    product_json: str = Field(description="구매 완료된 상품 정보 JSON")
    keywords: str = Field(default="", description="검색 키워드 콤마 구분")


class UpdatePreferenceTool(BaseTool):
    name: str = "update_preference"
    description: str = (
        "구매 완료 후 사용자 선호도 메모리를 업데이트합니다. "
        "플랫폼 패턴, 가격대, 브랜드, 최근 키워드, 배송 선호가 갱신됩니다."
    )
    args_schema: Type[BaseModel] = UpdatePreferenceInput

    def _run(self, user_id: str, product_json: str, keywords: str = "") -> str:
        try:
            product = json.loads(product_json)
        except Exception:
            return json.dumps({"error": "product_json 파싱 실패"}, ensure_ascii=False)
        kw_list = [k.strip() for k in keywords.split(",") if k.strip()]
        mock_update_preference_from_purchase(user_id, product, kw_list)
        return json.dumps({"success": True, "message": f"선호도 업데이트 완료 (user={user_id})"}, ensure_ascii=False)
