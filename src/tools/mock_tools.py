"""
외부 서비스 Mock 구현.

인터페이스/contract는 실제와 동일하게 유지.
Playwright, 결제 SDK, Meta-MCP, DB, VectorDB 모두 mock.
"""
import uuid
from typing import Any, Optional

# ══════════════════════════════════════════════
# Mock 상품 데이터
# ══════════════════════════════════════════════

MOCK_PRODUCTS: dict[str, list[dict[str, Any]]] = {
    "딸기": [
        {
            "product_name": "설향 딸기 500g",
            "price": 12900,
            "rating": 4.8,
            "review_count": 1523,
            "delivery": "새벽배송",
            "delivery_fee": 0,
            "platform": "kurly",
            "image_url": "https://mock.kurly.com/strawberry.jpg",
            "product_url": "https://mock.kurly.com/products/strawberry-500g",
            "is_sold_out": False,
            "raw": {},
        },
        {
            "product_name": "컬리 딸기 1kg 산지직송",
            "price": 19800,
            "rating": 4.7,
            "review_count": 643,
            "delivery": "새벽배송",
            "delivery_fee": 0,
            "platform": "kurly",
            "image_url": "https://mock.kurly.com/strawberry-1kg.jpg",
            "product_url": "https://mock.kurly.com/products/strawberry-1kg",
            "is_sold_out": False,
            "raw": {},
        },
        {
            "product_name": "죽향 딸기 1kg",
            "price": 22000,
            "rating": 4.6,
            "review_count": 892,
            "delivery": "로켓배송",
            "delivery_fee": 0,
            "platform": "coupang",
            "image_url": "https://mock.coupang.com/strawberry.jpg",
            "product_url": "https://mock.coupang.com/products/strawberry-1kg",
            "is_sold_out": False,
            "raw": {},
        },
        {
            "product_name": "논산 딸기 500g 특품",
            "price": 11500,
            "rating": 4.5,
            "review_count": 2104,
            "delivery": "로켓배송",
            "delivery_fee": 0,
            "platform": "coupang",
            "image_url": "https://mock.coupang.com/strawberry-sp.jpg",
            "product_url": "https://mock.coupang.com/products/strawberry-sp-500g",
            "is_sold_out": False,
            "raw": {},
        },
        {
            "product_name": "딸기 500g 당일수확",
            "price": 10900,
            "rating": 4.4,
            "review_count": 780,
            "delivery": "내일 도착",
            "delivery_fee": 3000,
            "platform": "naver",
            "image_url": "https://mock.naver.com/strawberry.jpg",
            "product_url": "https://mock.naver.com/products/strawberry-500g",
            "is_sold_out": False,
            "raw": {},
        },
        {
            "product_name": "프리미엄 설향 딸기 1kg",
            "price": 24000,
            "rating": 4.9,
            "review_count": 310,
            "delivery": "2일 후 도착",
            "delivery_fee": 0,
            "platform": "naver",
            "image_url": "https://mock.naver.com/strawberry-1kg.jpg",
            "product_url": "https://mock.naver.com/products/premium-strawberry-1kg",
            "is_sold_out": False,
            "raw": {},
        },
    ],
    "운동화": [
        {
            "product_name": "아디다스 슈퍼스타",
            "price": 89000,
            "rating": 4.5,
            "review_count": 3201,
            "delivery": "내일 도착",
            "delivery_fee": 0,
            "platform": "naver",
            "image_url": "https://mock.naver.com/shoes.jpg",
            "product_url": "https://mock.naver.com/products/adidas-superstar",
            "is_sold_out": False,
            "raw": {},
        },
        {
            "product_name": "나이키 에어맥스",
            "price": 139000,
            "rating": 4.7,
            "review_count": 5104,
            "delivery": "2일 후 도착",
            "delivery_fee": 0,
            "platform": "naver",
            "image_url": "https://mock.naver.com/nike.jpg",
            "product_url": "https://mock.naver.com/products/nike-airmax",
            "is_sold_out": False,
            "raw": {},
        },
        {
            "product_name": "뉴발란스 993 운동화",
            "price": 119000,
            "rating": 4.6,
            "review_count": 2870,
            "delivery": "로켓배송",
            "delivery_fee": 0,
            "platform": "coupang",
            "image_url": "https://mock.coupang.com/nb993.jpg",
            "product_url": "https://mock.coupang.com/products/nb-993",
            "is_sold_out": False,
            "raw": {},
        },
        {
            "product_name": "아식스 젤 카야노 운동화",
            "price": 159000,
            "rating": 4.8,
            "review_count": 1430,
            "delivery": "로켓배송",
            "delivery_fee": 0,
            "platform": "coupang",
            "image_url": "https://mock.coupang.com/asics.jpg",
            "product_url": "https://mock.coupang.com/products/asics-kayano",
            "is_sold_out": False,
            "raw": {},
        },
    ],
    "참기름": [
        {
            "product_name": "오뚜기 참기름 500ml",
            "price": 15900,
            "rating": 4.7,
            "review_count": 892,
            "delivery": "로켓배송",
            "delivery_fee": 0,
            "platform": "coupang",
            "image_url": "https://mock.coupang.com/sesame.jpg",
            "product_url": "https://mock.coupang.com/products/sesame-oil",
            "is_sold_out": False,
            "raw": {},
        },
        {
            "product_name": "CJ 백설 참기름 320ml",
            "price": 11900,
            "rating": 4.5,
            "review_count": 1203,
            "delivery": "로켓배송",
            "delivery_fee": 0,
            "platform": "coupang",
            "image_url": "https://mock.coupang.com/sesame-cj.jpg",
            "product_url": "https://mock.coupang.com/products/cj-sesame-oil",
            "is_sold_out": False,
            "raw": {},
        },
        {
            "product_name": "참들이향 참기름 500ml",
            "price": 17500,
            "rating": 4.8,
            "review_count": 540,
            "delivery": "내일 도착",
            "delivery_fee": 0,
            "platform": "naver",
            "image_url": "https://mock.naver.com/sesame.jpg",
            "product_url": "https://mock.naver.com/products/sesame-oil-premium",
            "is_sold_out": False,
            "raw": {},
        },
        {
            "product_name": "컬리 국산 참기름 200ml",
            "price": 13800,
            "rating": 4.7,
            "review_count": 329,
            "delivery": "새벽배송",
            "delivery_fee": 0,
            "platform": "kurly",
            "image_url": "https://mock.kurly.com/sesame.jpg",
            "product_url": "https://mock.kurly.com/products/sesame-oil-200ml",
            "is_sold_out": False,
            "raw": {},
        },
    ],
    "계란": [
        {
            "product_name": "풀무원 유정란 10구",
            "price": 4900,
            "rating": 4.6,
            "review_count": 3200,
            "delivery": "로켓배송",
            "delivery_fee": 0,
            "platform": "coupang",
            "image_url": "https://mock.coupang.com/egg.jpg",
            "product_url": "https://mock.coupang.com/products/egg-10",
            "is_sold_out": False,
            "raw": {},
        },
        {
            "product_name": "동물복지 계란 15구",
            "price": 7200,
            "rating": 4.8,
            "review_count": 1900,
            "delivery": "새벽배송",
            "delivery_fee": 0,
            "platform": "kurly",
            "image_url": "https://mock.kurly.com/egg.jpg",
            "product_url": "https://mock.kurly.com/products/egg-15",
            "is_sold_out": False,
            "raw": {},
        },
        {
            "product_name": "무항생제 계란 30구",
            "price": 9500,
            "rating": 4.7,
            "review_count": 2450,
            "delivery": "내일 도착",
            "delivery_fee": 0,
            "platform": "naver",
            "image_url": "https://mock.naver.com/egg-30.jpg",
            "product_url": "https://mock.naver.com/products/egg-30",
            "is_sold_out": False,
            "raw": {},
        },
    ],
    "우유": [
        {
            "product_name": "서울우유 1L",
            "price": 2800,
            "rating": 4.5,
            "review_count": 5100,
            "delivery": "로켓배송",
            "delivery_fee": 0,
            "platform": "coupang",
            "image_url": "https://mock.coupang.com/milk.jpg",
            "product_url": "https://mock.coupang.com/products/milk-1l",
            "is_sold_out": False,
            "raw": {},
        },
        {
            "product_name": "매일 ESL 흰우유 900ml",
            "price": 3200,
            "rating": 4.6,
            "review_count": 2800,
            "delivery": "로켓배송",
            "delivery_fee": 0,
            "platform": "coupang",
            "image_url": "https://mock.coupang.com/milk-maeil.jpg",
            "product_url": "https://mock.coupang.com/products/maeil-milk-900ml",
            "is_sold_out": False,
            "raw": {},
        },
        {
            "product_name": "연세 저지방 우유 1L",
            "price": 3500,
            "rating": 4.7,
            "review_count": 980,
            "delivery": "새벽배송",
            "delivery_fee": 0,
            "platform": "kurly",
            "image_url": "https://mock.kurly.com/milk-yonsei.jpg",
            "product_url": "https://mock.kurly.com/products/yonsei-lowfat-milk",
            "is_sold_out": False,
            "raw": {},
        },
        {
            "product_name": "남양 맛있는우유 GT 1L",
            "price": 2600,
            "rating": 4.4,
            "review_count": 3100,
            "delivery": "내일 도착",
            "delivery_fee": 0,
            "platform": "naver",
            "image_url": "https://mock.naver.com/milk-namyang.jpg",
            "product_url": "https://mock.naver.com/products/namyang-milk-1l",
            "is_sold_out": False,
            "raw": {},
        },
    ],
    "사과": [
        {
            "product_name": "GAP 인증 홍로 사과 1.5kg",
            "price": 9900,
            "rating": 4.7,
            "review_count": 2300,
            "delivery": "내일 도착",
            "delivery_fee": 0,
            "platform": "naver",
            "image_url": "https://mock.naver.com/apple.jpg",
            "product_url": "https://mock.naver.com/products/apple-15kg",
            "is_sold_out": False,
            "raw": {},
        },
        {
            "product_name": "경북 부사 사과 3kg",
            "price": 18900,
            "rating": 4.8,
            "review_count": 1560,
            "delivery": "로켓배송",
            "delivery_fee": 0,
            "platform": "coupang",
            "image_url": "https://mock.coupang.com/apple-3kg.jpg",
            "product_url": "https://mock.coupang.com/products/apple-3kg",
            "is_sold_out": False,
            "raw": {},
        },
        {
            "product_name": "산지직송 꿀사과 2kg",
            "price": 14500,
            "rating": 4.6,
            "review_count": 870,
            "delivery": "새벽배송",
            "delivery_fee": 0,
            "platform": "kurly",
            "image_url": "https://mock.kurly.com/apple-2kg.jpg",
            "product_url": "https://mock.kurly.com/products/honey-apple-2kg",
            "is_sold_out": False,
            "raw": {},
        },
    ],
}

DEFAULT_PRODUCTS = [
    {
        "product_name": "상품 A",
        "price": 15000,
        "rating": 4.3,
        "review_count": 200,
        "delivery": "내일 도착",
        "delivery_fee": 0,
        "platform": "naver",
        "image_url": "https://mock.naver.com/product-a.jpg",
        "product_url": "https://mock.naver.com/products/a",
        "is_sold_out": False,
        "raw": {},
    },
]

# ══════════════════════════════════════════════
# Mock search_product Tool
# ══════════════════════════════════════════════

def mock_search_product(
    query: str,
    platforms: list[str],
    condition: str = "relevance",
    budget_max: Optional[int] = None,
) -> list[dict[str, Any]]:
    results = []

    for keyword, products in MOCK_PRODUCTS.items():
        if keyword in query.lower() or any(keyword in q.lower() for q in query.split()):
            for p in products:
                if not platforms or p["platform"] in platforms:
                    results.append(p)

    if not results:
        results = list(DEFAULT_PRODUCTS)

    if budget_max:
        results = [p for p in results if p["price"] <= budget_max]

    if condition == "price_asc":
        results = sorted(results, key=lambda x: x["price"])
    elif condition == "review_score":
        results = sorted(results, key=lambda x: (x.get("rating") or 0, x.get("review_count") or 0), reverse=True)
    elif condition == "delivery_fast":
        results = sorted(results, key=lambda x: 0 if "로켓" in (x.get("delivery") or "") or "내일" in (x.get("delivery") or "") else 1)
    elif condition == "free_shipping":
        results = sorted(results, key=lambda x: 0 if x.get("delivery_fee") == 0 else 1)

    return results

# ══════════════════════════════════════════════
# Mock Playwright Browser Session
# ══════════════════════════════════════════════

class MockPageState:
    def __init__(self, url: str):
        self.url = url
        self.failed = False
        self.page = self


class MockBrowserSession:
    def __init__(self, session_id: str):
        self.id = session_id

    def open_product_page(self, product_url: str) -> MockPageState:
        return MockPageState(product_url)


class MockOptionResult:
    def __init__(self, failed: bool = False):
        self.failed = failed


class MockCartResult:
    def __init__(self, failed: bool = False):
        self.failed = failed


class MockAddressResult:
    def __init__(self, failed: bool = False):
        self.failed = failed


class MockValidation:
    def __init__(self, available: bool = True, price_changed: bool = False, current_price: Optional[int] = None):
        self.available = available
        self.price_changed = price_changed
        self.current_price = current_price


class MockPaymentPage:
    def __init__(self, failed: bool = False):
        self.failed = failed
        self.url = "https://mock.naverpay.com/payment/checkout"


_playwright_sessions: dict[str, MockBrowserSession] = {}


def get_or_create_playwright_session(user_id: str, session_key: Optional[str] = None) -> MockBrowserSession:
    if session_key and session_key in _playwright_sessions:
        return _playwright_sessions[session_key]
    session_id = str(uuid.uuid4())
    session = MockBrowserSession(session_id)
    _playwright_sessions[session_id] = session
    return session


def extract_available_options(page: MockPageState) -> list[dict[str, Any]]:
    if "strawberry" in page.url or "딸기" in page.url:
        return [{"key": "용량", "values": ["500g", "1kg"]}]
    return []


def build_option_question(available_options: list[dict[str, Any]], current_index: int = 0) -> str:
    if not available_options or current_index >= len(available_options):
        return "옵션을 선택해 주세요."
    option = available_options[current_index]
    values = ", ".join(option.get("values", []))
    return f"{option['key']}을 선택해 주세요: {values}"


def apply_options_to_page(page: MockPageState, selected_options: dict[str, Any]) -> MockOptionResult:
    return MockOptionResult(failed=False)


def add_to_cart_or_buy_now(page: MockPageState, quantity: int = 1) -> MockCartResult:
    return MockCartResult(failed=False)


def fill_delivery_address(page: MockPageState, delivery_address: dict[str, Any]) -> MockAddressResult:
    return MockAddressResult(failed=False)


def format_address_confirm_message(delivery_address: dict[str, Any]) -> str:
    addr1 = delivery_address.get("address_line1", "")
    addr2 = delivery_address.get("address_line2", "")
    recipient = delivery_address.get("recipient_name", "")
    addr = f"{addr1} {addr2}".strip()
    return f"{recipient}님, {addr}로 배송할까요?"


def revalidate_product_on_page(page: MockPageState) -> MockValidation:
    return MockValidation(available=True, price_changed=False)


def proceed_to_naverpay(page: MockPageState) -> MockPaymentPage:
    return MockPaymentPage(failed=False)


def open_webview(webview_type: str, url: str) -> dict[str, Any]:
    return {"type": "open_webview", "webview_type": webview_type, "url": url}


# ══════════════════════════════════════════════
# Mock Checkout Session
# ══════════════════════════════════════════════

class MockCheckoutSession:
    def __init__(self, session_id, user_id, product, product_url, quantity, selected_platform,
                 delivery_address=None, conversation_id=None):
        self.id = session_id
        self.user_id = user_id
        self.product = product
        self.product_url = product_url
        self.quantity = quantity
        self.selected_platform = selected_platform or "naver"
        self.delivery_address = delivery_address
        self.price = product.get("price", 0)
        self.payment_url = f"https://mock.naverpay.com/checkout/{session_id}"
        self.conversation_id = conversation_id
        self.cart_id = str(uuid.uuid4())
        self.status = "active"
        self.total_expected_amount = self.price * quantity
        self.address_confirmed = False


_checkout_sessions: dict[str, MockCheckoutSession] = {}


def get_checkout_session(checkout_session_id: str) -> Optional[MockCheckoutSession]:
    return _checkout_sessions.get(checkout_session_id)


def create_checkout_session(user_id, conversation_id, product, product_url, quantity, selected_platform) -> MockCheckoutSession:
    session_id = str(uuid.uuid4())
    session = MockCheckoutSession(
        session_id=session_id, user_id=user_id, product=product, product_url=product_url,
        quantity=quantity, selected_platform=selected_platform, conversation_id=conversation_id,
    )
    _checkout_sessions[session_id] = session
    return session


def update_checkout_price(checkout_session_id: str, new_price: int) -> None:
    session = _checkout_sessions.get(checkout_session_id)
    if session:
        session.price = new_price


# ══════════════════════════════════════════════
# Mock Order / Payment Record
# ══════════════════════════════════════════════

class MockOrder:
    def __init__(self, order_id, checkout_id, payment_url, user_id="",
                 conversation_id=None, status="pending_confirmation", total_payment_amount=0, platform="naver"):
        self.id = order_id
        self.checkout_id = checkout_id
        self.payment_url = payment_url
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.status = status
        self.total_payment_amount = total_payment_amount
        self.platform = platform


class MockPaymentRecord:
    def __init__(self, payment_id, order_id, payment_amount=0, payment_provider="naver_pay",
                 payment_status="pending", payment_url=None):
        self.id = payment_id
        self.order_id = order_id
        self.payment_amount = payment_amount
        self.payment_provider = payment_provider
        self.payment_status = payment_status
        self.payment_url = payment_url or f"https://mock.naverpay.com/payment/{payment_id}"


_orders: dict[str, MockOrder] = {}
_order_ids: dict[str, MockOrder] = {}


def find_order_by_idempotency_key(checkout_id: str) -> Optional[MockOrder]:
    return _orders.get(checkout_id)


def create_order_from_checkout(checkout_id: str, validation: MockValidation) -> MockOrder:
    checkout = _checkout_sessions.get(checkout_id)
    total = checkout.total_expected_amount if checkout else 0
    order_id = str(uuid.uuid4())
    order = MockOrder(
        order_id=order_id, checkout_id=checkout_id,
        payment_url=f"https://mock.naverpay.com/payment/{order_id}",
        user_id=checkout.user_id if checkout else "",
        conversation_id=checkout.conversation_id if checkout else None,
        status="pending_confirmation",
        total_payment_amount=total,
        platform=checkout.selected_platform if checkout else "naver",
    )
    _orders[checkout_id] = order
    _order_ids[order_id] = order
    return order


def create_order_item(order_id: str, checkout_id: str, validation: MockValidation) -> None:
    pass


def create_payment_record(order_id: str) -> MockPaymentRecord:
    order = _order_ids.get(order_id)
    payment_id = str(uuid.uuid4())
    return MockPaymentRecord(
        payment_id=payment_id, order_id=order_id,
        payment_amount=order.total_payment_amount if order else 0,
        payment_provider="naver_pay", payment_status="pending",
    )


def mark_order_payment_failed(order_id: str) -> None:
    pass


# ══════════════════════════════════════════════
# Mock transaction context manager
# ══════════════════════════════════════════════

class transaction:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


# ══════════════════════════════════════════════
# Mock DB — Users, Addresses, Purchase History
# ══════════════════════════════════════════════

MOCK_USERS: dict[str, dict[str, Any]] = {
    "1": {"name": "김영희", "age_group": "70s"},
    "user_001": {"name": "김영희", "age_group": "60s"},
    "user_test": {"name": "테스트유저", "age_group": "30s"},
    "demo": {"name": "데모유저", "age_group": "40s"},
}

MOCK_ADDRESSES: dict[str, dict[str, Any]] = {
    "1": {
        "id": "addr_001",
        "address_label": "집",
        "recipient_name": "김영희",
        "recipient_phone": "010-1234-5678",
        "address_line1": "서울특별시 강남구 테헤란로 1길 10",
        "address_line2": "101호",
        "zip_code": "06000",
        "delivery_request": "문 앞에 놓아주세요",
        "is_default": True,
    },
    "user_001": {
        "id": "addr_001",
        "address_label": "집",
        "recipient_name": "김영희",
        "recipient_phone": "010-1234-5678",
        "address_line1": "서울특별시 강남구 테헤란로 123",
        "address_line2": "101호",
        "zip_code": "06234",
        "delivery_request": "문 앞에 놓아주세요",
        "is_default": True,
    },
    "user_test": {
        "id": "addr_test",
        "address_label": "집",
        "recipient_name": "테스트유저",
        "recipient_phone": "010-0000-0000",
        "address_line1": "서울특별시 마포구 합정동 100",
        "address_line2": "",
        "zip_code": "04040",
        "delivery_request": "",
        "is_default": True,
    },
    "demo": {
        "id": "addr_demo",
        "address_label": "집",
        "recipient_name": "데모유저",
        "recipient_phone": "010-9999-0000",
        "address_line1": "서울특별시 송파구 올림픽로 300",
        "address_line2": "201호",
        "zip_code": "05544",
        "delivery_request": "",
        "is_default": True,
    },
}

MOCK_PURCHASE_HISTORY: dict[str, list[dict[str, Any]]] = {
    "user_001": [
        {
            "id": "ph_001",
            "order_id": "order_001",
            "product_name": "설향 딸기 500g",
            "brand": None,
            "category": "과일",
            "option_text": "500g",
            "platform": "kurly",
            "price_at_purchase": 12900,
            "quantity": 1,
            "total_price": 12900,
            "selected_options": {"용량": "500g"},
            "product_url": "https://mock.kurly.com/products/strawberry-500g",
            "purchased_at": "2025-01-10T10:00:00",
            "satisfaction_score": None,
            "keyword": "딸기",
        },
        {
            "id": "ph_002",
            "order_id": "order_002",
            "product_name": "풀무원 유정란 10구",
            "brand": "풀무원",
            "category": "계란",
            "option_text": None,
            "platform": "coupang",
            "price_at_purchase": 4900,
            "quantity": 2,
            "total_price": 9800,
            "selected_options": {},
            "product_url": "https://mock.coupang.com/products/egg-10",
            "purchased_at": "2025-02-05T14:00:00",
            "satisfaction_score": 5,
            "keyword": "계란",
        },
    ],
    "user_test": [],
    "demo": [
        {
            "id": "ph_demo_001",
            "order_id": "order_demo_001",
            "product_name": "오뚜기 참기름 500ml",
            "brand": "오뚜기",
            "category": "조미료",
            "option_text": None,
            "platform": "coupang",
            "price_at_purchase": 15900,
            "quantity": 1,
            "total_price": 15900,
            "selected_options": {},
            "product_url": "https://mock.coupang.com/products/sesame-oil",
            "purchased_at": "2025-03-01T09:00:00",
            "satisfaction_score": None,
            "keyword": "참기름",
        },
    ],
}

MOCK_PREFERENCE_MEMORY: dict[str, dict[str, Any]] = {
    "user_001": {
        "platform_pattern": {"과일": "kurly", "딸기": "kurly", "계란": "coupang"},
        "price_range": {
            "과일": {"samples": [12900], "min": 12900, "max": 12900, "avg": 12900},
            "계란": {"samples": [4900], "min": 4900, "max": 4900, "avg": 4900},
        },
        "preferred_delivery": "새벽배송",
        "excluded_brands": [],
        "preferred_brands": ["풀무원"],
        "recent_keywords": ["딸기", "계란", "과일"],
        "updated_at": "2025-02-05T14:00:00",
    },
    "user_test": {
        "platform_pattern": {},
        "price_range": {},
        "preferred_delivery": None,
        "excluded_brands": [],
        "preferred_brands": [],
        "recent_keywords": [],
    },
    "demo": {
        "platform_pattern": {"조미료": "coupang", "참기름": "coupang"},
        "price_range": {},
        "preferred_delivery": "로켓배송",
        "excluded_brands": [],
        "preferred_brands": ["오뚜기"],
        "recent_keywords": ["참기름"],
        "updated_at": "2025-03-01T09:00:00",
    },
}


def mock_get_user(user_id: str) -> Optional[dict[str, Any]]:
    return MOCK_USERS.get(user_id)


def mock_get_default_address(user_id: str) -> Optional[dict[str, Any]]:
    return MOCK_ADDRESSES.get(user_id)


def mock_get_purchase_history(user_id: str) -> list[dict[str, Any]]:
    return MOCK_PURCHASE_HISTORY.get(user_id, [])


def mock_keyword_search_history(user_id: str, keywords: list[str], limit: int = 5) -> list[dict[str, Any]]:
    history = mock_get_purchase_history(user_id)
    results = []
    for item in history:
        name = item.get("product_name", "").lower()
        cat = item.get("category", "").lower()
        keyword = item.get("keyword", "").lower()
        if any(k.lower() in name or k.lower() in cat or k.lower() in keyword for k in keywords):
            results.append(item)
    return results[:limit]


def mock_get_preference_memory(user_id: str) -> dict[str, Any]:
    return MOCK_PREFERENCE_MEMORY.get(user_id, {})


def mock_update_preference_from_purchase(
    user_id: str,
    product: dict[str, Any],
    keywords: list[str],
) -> None:
    """구매 완료 후 선호도 메모리 자동 업데이트 (Learning & Adaptation)."""
    pref = MOCK_PREFERENCE_MEMORY.setdefault(user_id, {
        "platform_pattern": {},
        "price_range": {},
        "preferred_delivery": None,
        "excluded_brands": [],
        "preferred_brands": [],
        "recent_keywords": [],
    })

    platform = product.get("platform") or ""
    category = product.get("category") or (keywords[0] if keywords else None)
    price = int(product.get("price") or 0)
    brand = product.get("brand") or ""
    delivery = product.get("delivery") or ""

    # 플랫폼 패턴: 카테고리/키워드 → 선호 플랫폼
    if platform and category:
        pref.setdefault("platform_pattern", {})[category] = platform
    for kw in keywords:
        if kw and platform:
            pref.setdefault("platform_pattern", {})[kw] = platform

    # 가격대: 카테고리별 샘플 누적 → min/max/avg 갱신
    if price and category:
        pr = pref.setdefault("price_range", {}).setdefault(
            category, {"samples": [], "min": 0, "max": 0, "avg": 0}
        )
        pr["samples"] = (pr.get("samples") or []) + [price]
        pr["min"] = min(pr["samples"])
        pr["max"] = max(pr["samples"])
        pr["avg"] = sum(pr["samples"]) // len(pr["samples"])

    # 브랜드 선호
    if brand and brand not in (pref.get("preferred_brands") or []):
        pref.setdefault("preferred_brands", []).append(brand)

    # 최근 키워드 (최대 10개)
    recent = pref.get("recent_keywords") or []
    for kw in keywords:
        if kw and kw not in recent:
            recent.insert(0, kw)
    pref["recent_keywords"] = recent[:10]

    # 배송 선호: 새벽배송 > 로켓배송 > 당일배송
    if "새벽" in delivery:
        pref["preferred_delivery"] = "새벽배송"
    elif "로켓" in delivery and pref.get("preferred_delivery") != "새벽배송":
        pref["preferred_delivery"] = "로켓배송"
    elif "당일" in delivery and not pref.get("preferred_delivery"):
        pref["preferred_delivery"] = "당일배송"

    from datetime import datetime
    pref["updated_at"] = datetime.now().isoformat()
    pass  # 선호도 업데이트 완료


def mock_count_purchases(user_id: str) -> int:
    return len(mock_get_purchase_history(user_id))


def mock_save_purchase_history(
    user_id: str,
    conversation_id: Optional[int],
    order_id: str,
    product_name: str,
    price_at_purchase: int,
    quantity: int,
    platform: str = "",
    product_url: str = "",
    option_text: Optional[str] = None,
    selected_options: Optional[dict[str, Any]] = None,
    brand: Optional[str] = None,
    category: Optional[str] = None,
    keyword: Optional[str] = None,
) -> str:
    from datetime import datetime
    history_id = f"ph_{str(uuid.uuid4())[:8]}"
    entry = {
        "id": history_id,
        "user_id": user_id,
        "conversation_id": conversation_id,
        "order_id": order_id,
        "product_name": product_name,
        "brand": brand,
        "category": category,
        "option_text": option_text,
        "platform": platform,
        "price_at_purchase": price_at_purchase,
        "quantity": quantity,
        "total_price": price_at_purchase * quantity,
        "selected_options": selected_options or {},
        "product_url": product_url,
        "purchased_at": datetime.now().isoformat(),
        "satisfaction_score": None,
        "keyword": keyword,
    }
    MOCK_PURCHASE_HISTORY.setdefault(user_id, []).append(entry)
    pass  # 구매이력 저장 완료
    return history_id


# ══════════════════════════════════════════════
# Mock Cart
# ══════════════════════════════════════════════

_mock_carts: dict[str, list[dict[str, Any]]] = {}


def mock_add_to_cart(
    user_id: str,
    product: dict[str, Any],
    quantity: int,
    keywords: Optional[list[str]] = None,
) -> dict[str, Any]:
    item = {
        "product_name": product.get("product_name", "상품"),
        "price": product.get("price", 0),
        "quantity": quantity,
        "total": (product.get("price", 0) or 0) * quantity,
        "platform": product.get("platform", ""),
        "product_url": product.get("product_url", ""),
        "product": product,
        "keywords": keywords or [],
    }
    _mock_carts.setdefault(user_id, []).append(item)
    pass  # 장바구니 담기 완료
    return item


def mock_get_cart(user_id: str) -> list[dict[str, Any]]:
    return list(_mock_carts.get(user_id, []))


def mock_clear_cart(user_id: str) -> None:
    _mock_carts[user_id] = []


def mock_place_order(
    user_id: str,
    delivery_address: dict[str, Any],
    payment_method: str = "naver_pay",
    conversation_id: Optional[int] = None,
) -> dict[str, Any]:
    from datetime import datetime
    cart = mock_get_cart(user_id)
    order_id = f"ORDER-{str(uuid.uuid4())[:8].upper()}"
    total = sum(item["total"] for item in cart)

    for item in cart:
        mock_save_purchase_history(
            user_id=user_id,
            conversation_id=conversation_id,
            order_id=order_id,
            product_name=item["product_name"],
            price_at_purchase=item["price"],
            quantity=item["quantity"],
            platform=item["platform"],
            product_url=item["product_url"],
            keyword=(item["keywords"][0] if item.get("keywords") else None),
        )

    mock_clear_cart(user_id)
    pass  # 주문 완료
    return {
        "order_id": order_id,
        "total": total,
        "item_count": len(cart),
        "payment_method": payment_method,
        "delivery_address": delivery_address,
        "status": "confirmed",
    }


# ══════════════════════════════════════════════
# Mock Vector DB
# ══════════════════════════════════════════════

def mock_vector_search_personal(user_id: str, query: str, limit: int = 5) -> list[dict[str, Any]]:
    history = mock_get_purchase_history(user_id)
    return history[:limit]


def mock_vector_search_collective(query: str, age_group: Optional[str] = None, limit: int = 5) -> list[dict[str, Any]]:
    all_products = []
    for products in MOCK_PRODUCTS.values():
        all_products.extend(products)
    return all_products[:limit]


# ══════════════════════════════════════════════
# Mock URL Validator
# ══════════════════════════════════════════════

_BLOCKED_URLS: set[str] = set()


def mock_validate_product_url(url: str) -> bool:
    if not url:
        return False
    if url in _BLOCKED_URLS:
        return False
    valid_domains = (
        "mock.kurly.com", "mock.coupang.com", "mock.naver.com", "example.com",
        "www.kurly.com", "www.coupang.com",
    )
    return any(domain in url for domain in valid_domains)


def block_product_url(url: str) -> None:
    _BLOCKED_URLS.add(url)


def unblock_product_url(url: str) -> None:
    _BLOCKED_URLS.discard(url)
