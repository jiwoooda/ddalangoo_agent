"""장바구니 및 결제 Tool — mock_add_to_cart / mock_place_order 래핑."""
import json
from typing import Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from src.shared import (
    mock_add_to_cart,
    mock_get_cart,
    mock_place_order,
    mock_get_default_address,
)


class AddToCartInput(BaseModel):
    user_id: str = Field(description="사용자 ID")
    product_json: str = Field(description="담을 상품 정보 JSON 문자열")
    quantity: int = Field(default=1, description="수량 (미지정 시 1)")
    keyword: str = Field(default="", description="검색 키워드")


class AddToCartTool(BaseTool):
    name: str = "add_to_cart"
    description: str = "선택한 상품을 장바구니에 담습니다."
    args_schema: Type[BaseModel] = AddToCartInput

    def _run(self, user_id: str, product_json: str, quantity: int, keyword: str = "") -> str:
        try:
            product = json.loads(product_json)
        except Exception:
            return json.dumps({"error": "product_json 파싱 실패"}, ensure_ascii=False)
        keywords = [keyword] if keyword else []
        mock_add_to_cart(user_id, product, quantity, keywords)
        cart = mock_get_cart(user_id)
        cart_total = sum(i.get("total", 0) for i in cart)
        return json.dumps({
            "success": True,
            "cart_count": len(cart),
            "cart_total": cart_total,
            "cart": cart,
        }, ensure_ascii=False, indent=2)


class GetCartInput(BaseModel):
    user_id: str = Field(description="사용자 ID")


class GetCartTool(BaseTool):
    name: str = "get_cart"
    description: str = "현재 장바구니 내용과 총액을 조회합니다."
    args_schema: Type[BaseModel] = GetCartInput

    def _run(self, user_id: str) -> str:
        cart = mock_get_cart(user_id)
        cart_total = sum(i.get("total", 0) for i in cart)
        return json.dumps({
            "cart_count": len(cart),
            "cart_total": cart_total,
            "cart": cart,
        }, ensure_ascii=False, indent=2)


class PlaceOrderInput(BaseModel):
    user_id: str = Field(description="사용자 ID")
    address_text: str = Field(default="", description="배송지 주소 텍스트")
    conversation_id: int = Field(default=0, description="대화 ID")


class PlaceOrderTool(BaseTool):
    name: str = "place_order"
    description: str = "최종 주문을 실행합니다. 비밀번호 확인 후 호출합니다."
    args_schema: Type[BaseModel] = PlaceOrderInput

    def _run(self, user_id: str, address_text: str = "", conversation_id: int = 0) -> str:
        if address_text:
            delivery_address = {
                "address_line1": address_text,
                "address_line2": "",
                "recipient_name": "고객",
                "recipient_phone": "",
                "zip_code": "",
            }
        else:
            delivery_address = mock_get_default_address(user_id) or {}

        order = mock_place_order(
            user_id=user_id,
            delivery_address=delivery_address,
            payment_method="naver_pay",
            conversation_id=conversation_id,
        )
        return json.dumps({
            "success": True,
            "order_id": order["order_id"],
            "total": order["total"],
            "message": "주문이 완료됐어요!",
        }, ensure_ascii=False, indent=2)


class GetAddressInput(BaseModel):
    user_id: str = Field(description="사용자 ID")


class GetAddressTool(BaseTool):
    name: str = "get_address"
    description: str = "사용자의 기본 배송지를 조회합니다."
    args_schema: Type[BaseModel] = GetAddressInput

    def _run(self, user_id: str) -> str:
        address = mock_get_default_address(user_id) or {}
        addr_line = f"{address.get('address_line1', '')} {address.get('address_line2', '')}".strip()
        return json.dumps({
            "address": addr_line or "배송지 없음",
            "raw": address,
        }, ensure_ascii=False)
