"""
Payment Agent Node.

결제 단계 (모두 mock):
  1. 상품 확인 → mock_add_to_cart → 장바구니 안내
  2. 결제 시작 → mock_get_cart → 총액 + 결제수단 확인
  3. 결제수단 확인 → 배송지 확인
  4. 배송지 확인 → 비밀번호 요청
  5. 비밀번호 입력 → mock_place_order → 완료
"""
import re
from src.state.schema import ShoppingState
from src.utils.agent_logger import agent_logger, _ptype
from src.tools.mock_tools import (
    mock_add_to_cart,
    mock_get_cart,
    mock_place_order,
    mock_get_default_address,
)

_KR_NUMBERS = {
    "하나": 1, "한": 1, "일": 1,
    "둘": 2, "두": 2,
    "셋": 3, "세": 3,
    "넷": 4, "네": 4,
    "다섯": 5, "오": 5,
    "여섯": 6, "육": 6,
    "일곱": 7, "칠": 7,
    "여덟": 8, "팔": 8,
    "아홉": 9, "구": 9,
    "열": 10, "십": 10,
}


def _coerce_positive_int(value, default=None):
    if value is None:
        return default
    if isinstance(value, int):
        return value if value > 0 else default
    text = str(value).strip()
    m = re.search(r"\d+", text.replace(",", ""))
    if m:
        parsed = int(m.group())
        return parsed if parsed > 0 else default
    for token, number in sorted(_KR_NUMBERS.items(), key=lambda x: -len(x[0])):
        if token in text:
            return number
    return default


def _build_delivery_address(state: ShoppingState) -> dict:
    address_text = state.get("address_text")
    if address_text:
        return {"address_line1": address_text, "address_line2": "", "recipient_name": "고객", "recipient_phone": "", "zip_code": ""}
    return mock_get_default_address(state.get("user_id", "")) or {}


def _format_address(address: dict) -> str:
    addr1 = address.get("address_line1", "")
    addr2 = address.get("address_line2", "")
    return f"{addr1} {addr2}".strip() if addr2 else addr1


def _short_address(addr: str) -> str:
    parts = addr.split()
    return " ".join(parts[:3]) + "..." if len(parts) > 3 else addr


def _delivery_msg(delivery_info: str) -> str:
    if "샛별" in delivery_info:
        return " 내일 아침 7시 전 도착이에요."
    if "로켓" in delivery_info:
        return " 내일 도착이에요."
    if "당일" in delivery_info:
        return " 오늘 도착이에요."
    return ""


def payment_agent_node(state: ShoppingState) -> dict:
    stage = state.get("stage")
    pending_type = (state.get("pending_action") or {}).get("type")
    user_id = state.get("user_id", "")

    selected_product = state.get("selected_product") or {}
    keywords = state.get("keywords") or []
    short_name = keywords[0] if keywords else selected_product.get("product_name", "상품")
    price = _coerce_positive_int(selected_product.get("price"), default=0) or 0
    quantity = _coerce_positive_int(state.get("quantity"), default=None)
    delivery_info = selected_product.get("delivery", "")

    total = price * (quantity or 1)

    intent = state.get("intent")
    _log_in = {"intent": intent, "pending_type": pending_type, "quantity": quantity, "stage": stage}

    # ── Step 0: 상품 확인 → 장바구니 담기 ──
    if (
        stage == "product_confirming"
        and pending_type in ("product_confirm", "quantity_confirm")
        and intent in ("confirm", "quantity_change")
    ):
        mock_add_to_cart(user_id, selected_product, quantity, keywords)
        cart = mock_get_cart(user_id)
        if len(cart) > 1:
            cart_msg = f"{short_name}도 담았어요! 총 {len(cart)}가지예요. 결제할까요, 더 담을까요?"
        else:
            cart_msg = f"{short_name} {quantity}개 담았어요! 결제할까요, 다른 것도 보실래요?"
        output = {
            "stage": "cart_shopping",
            "selected_product": selected_product,
            "cart_items": cart,
            "error": None,
            "last_agent": "payment_agent",
            "pending_action": {"type": "continue_shopping", "message": cart_msg},
        }
        agent_logger.log_payment_agent(_log_in, output)
        agent_logger.log(f"[payment_agent] Step 0 완료 | 장바구니 {len(cart)}개  총액 {sum(i['total'] for i in cart):,}원")
        return output

    # ── Step 1: 장바구니 최종 점검 (cart_review) ──
    if (stage == "cart_shopping" or pending_type is None) and pending_type != "cart_review":
        cart = mock_get_cart(user_id)
        if not cart and not selected_product:
            output = {
                "stage": "cart_shopping",
                "error": "payment_precheck_missing_product",
                "last_agent": "payment_agent",
                "pending_action": {"type": "what_to_buy", "message": "상품을 아직 찾지 못했어요. 무엇을 구매하실까요?"},
            }
            agent_logger.log_payment_agent(_log_in, output)
            return output

        cart = cart or []
        if cart:
            cart_total = sum(item["total"] for item in cart)
            items_summary = ", ".join(
                f"{(item.get('keywords') or [item.get('product_name', '상품')])[0]} {item.get('quantity', 1)}개"
                for item in cart
            )
            review_msg = f"총 {cart_total:,}원이에요. 수량 바꾸거나 빼실 게 있으면 말씀해 주세요."
        else:
            review_msg = f"{short_name} {quantity or 1}개, {total:,}원이에요. 수량 바꾸거나 빼실 게 있으면 말씀해 주세요."

        output = {
            "stage": "cart_shopping",
            "error": None,
            "last_agent": "payment_agent",
            "pending_action": {"type": "cart_review", "message": review_msg},
        }
        agent_logger.log_payment_agent(_log_in, output)
        agent_logger.log(f"[payment_agent] Step 1 완료 | 장바구니 점검 대기")
        return output

    # ── Step 1-5: cart_review 확인 → 결제수단 선택 ──
    if pending_type == "cart_review":
        if intent == "quantity_change":
            new_qty = _coerce_positive_int(state.get("quantity"), default=quantity)
            if new_qty and selected_product:
                from src.tools.mock_tools import mock_clear_cart
                mock_clear_cart(user_id)
                mock_add_to_cart(user_id, selected_product, new_qty, keywords)
            cart = mock_get_cart(user_id)
            cart_total = sum(item["total"] for item in cart) if cart else total
            items_summary = ", ".join(
                f"{(item.get('keywords') or [item.get('product_name', '상품')])[0]} {item.get('quantity', 1)}개"
                for item in cart
            ) if cart else f"{short_name} {new_qty}개"
            review_msg = f"총 {cart_total:,}원이에요. 수량 바꾸거나 빼실 게 있으면 말씀해 주세요."
            output = {
                "stage": "cart_shopping",
                "quantity": new_qty,
                "cart_items": cart,
                "error": None,
                "last_agent": "payment_agent",
                "pending_action": {"type": "cart_review", "message": review_msg},
            }
            agent_logger.log_payment_agent(_log_in, output)
            return output

        address = _build_delivery_address(state)
        addr_display = _format_address(address)
        if not addr_display:
            output = {
                "stage": "cart_shopping",
                "error": "address_required",
                "last_agent": "payment_agent",
                "pending_action": {"type": "address_required", "message": "배송지가 아직 없어요. 먼저 배송지를 등록해 주세요.", "payload": {"subType": "address_required"}},
            }
            agent_logger.log_payment_agent(_log_in, output)
            return output
        output = {
            "stage": "payment_processing",
            "error": None,
            "last_agent": "payment_agent",
            "pending_action": {"type": "address_confirm", "message": f"{_short_address(addr_display)}로 보낼게요. 맞으시죠?"},
        }
        agent_logger.log_payment_agent(_log_in, output)
        agent_logger.log(f"[payment_agent] Step 1-5 완료 | 배송지={addr_display}")
        return output

    # ── Step 2: 배송지 확인 → 결제수단 선택 ──
    if pending_type == "address_confirm":
        cart = mock_get_cart(user_id)
        if cart:
            cart_total = sum(item["total"] for item in cart)
            items_summary = ", ".join(
                f"{(item.get('keywords') or [item.get('product_name', '상품')])[0]} {item.get('quantity', 1)}개"
                for item in cart
            )
            payment_msg = f"{items_summary}, 총 {cart_total:,}원이에요. 네이버로 결제할까요?"
        else:
            payment_msg = f"{short_name} {quantity or 1}개, {total:,}원이에요. 네이버로 결제할까요?"
        output = {
            "stage": "payment_processing",
            "error": None,
            "last_agent": "payment_agent",
            "pending_action": {"type": "payment_method_confirm", "message": payment_msg},
        }
        agent_logger.log_payment_agent(_log_in, output)
        agent_logger.log(f"[payment_agent] Step 2 완료 | 결제수단 선택 대기")
        return output

    # ── Step 3: 결제수단 확인 → 비밀번호 요청 ──
    if pending_type == "payment_method_confirm":
        output = {
            "stage": "payment_processing",
            "error": None,
            "last_agent": "payment_agent",
            "pending_action": {"type": "payment_password", "message": "비밀번호 입력해주세요!"},
        }
        agent_logger.log_payment_agent(_log_in, output)
        agent_logger.log("[payment_agent] Step 3 완료 | 비밀번호 요청")
        return output

    # ── Step 4: 비밀번호 → mock 결제 실행 ──
    if pending_type == "payment_password":
        delivery_address = _build_delivery_address(state)
        order = mock_place_order(
            user_id=user_id,
            delivery_address=delivery_address,
            payment_method="naver_pay",
            conversation_id=state.get("conversation_id"),
        )
        arrival = _delivery_msg(delivery_info)
        completion_msg = f"완료!{arrival}" if arrival else "완료! 주문이 접수됐어요."
        output = {
            "stage": "completed",
            "order_id": order["order_id"],
            "cart_items": [],
            "error": None,
            "last_agent": "payment_agent",
            "pending_action": {"type": "payment_confirm", "message": completion_msg},
            "storage_state_path": None,
        }
        agent_logger.log_payment_agent(_log_in, output)
        agent_logger.log(
            f"[payment_agent] Step 4 완료 | 주문 접수  "
            f"order_id={order['order_id']}  total={order['total']:,}원"
        )
        return output

    # fallback
    output = {
        "stage": "payment_processing",
        "error": None,
        "last_agent": "payment_agent",
        "pending_action": {"type": "payment_method_confirm", "message": f"{short_name} {quantity}개, {total:,}원이에요. 네이버로 결제할까요?"},
    }
    agent_logger.log_payment_agent(_log_in, output)
    return output
