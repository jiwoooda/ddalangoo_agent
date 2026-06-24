"""상품 검색 Tool — mock_search.search_products 래핑."""
import json
from typing import Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from src.shared import search_products


class SearchInput(BaseModel):
    query: str = Field(description="검색어 (예: 딸기, 계란 500g)")
    platforms: str = Field(description="검색 플랫폼 콤마 구분 (예: naver,coupang,kurly)")
    condition: str = Field(default="relevance", description="정렬 조건: price_asc / review_score / delivery_fast / free_shipping / relevance")


class SearchProductsTool(BaseTool):
    name: str = "search_products_tool"
    description: str = (
        "네이버·쿠팡·컬리 쇼핑 플랫폼에서 상품을 검색합니다. "
        "검색 결과는 상품명, 가격, 배송정보, 평점, 리뷰수, 플랫폼이 포함된 JSON 리스트입니다."
    )
    args_schema: Type[BaseModel] = SearchInput

    def _run(self, query: str, platforms: str = "naver,coupang,kurly", condition: str = "relevance") -> str:
        platform_list = [p.strip() for p in platforms.split(",") if p.strip()]
        results = search_products(query=query, platforms=platform_list, condition=condition)
        if not results:
            return json.dumps({"error": "검색 결과 없음", "products": []}, ensure_ascii=False)
        return json.dumps({"products": results[:9]}, ensure_ascii=False, indent=2)
