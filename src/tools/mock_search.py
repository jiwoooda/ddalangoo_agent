"""
Mock Search Client — Meta MCP 대체.

검색 우선순위:
1. META_MCP_SERVER_URL 환경변수가 설정된 경우 → 원격 SSE 엔드포인트 호출
2. NAVER_CLIENT_ID / NAVER_CLIENT_SECRET이 설정된 경우 → 네이버 쇼핑 API 직접 호출
3. 위 두 가지 모두 없는 경우 → 내장 mock 데이터 반환 (항상 동작 보장)

플랫폼 에이전트에서 `search_products(query, platforms, condition)` 형태로 호출됨.
"""
import json
import os
import concurrent.futures
from typing import Any
from urllib.parse import quote, urlencode, urljoin
from urllib.request import Request, urlopen

from src.tools.mock_tools import mock_search_product

SORT_MAP = {
    "price_asc": "price_low",
    "price_desc": "price_high",
    "relevance": "sim",
    "popularity": "sim",
    "review_score": "sim",
    "delivery_fast": "sim",
    "free_shipping": "sim",
    "value": "sim",
}

NAVER_API_SORT_MAP = {
    "price_low": "sim",
    "price_high": "sim",
    "sim": "sim",
    "date": "date",
}

KURLY_SHOP_KEYWORDS = ("컬리", "마켓컬리", "kurly", "컬리n마트", "컬리 n마트")
KURLY_BASE_URL = "https://www.kurly.com"


def _strip_html(value: str) -> str:
    return value.replace("<b>", "").replace("</b>", "")


def _is_kurly_item(item: dict[str, Any]) -> bool:
    mall_name = str(item.get("mallName") or "").lower()
    title = str(item.get("title") or "").lower()
    return any(keyword in mall_name or keyword in title for keyword in KURLY_SHOP_KEYWORDS)


def _kurly_search_url(query: str) -> str:
    return f"{KURLY_BASE_URL}/search?sword={quote(query)}"


def _normalize(products: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for p in products:
        delivery_info = p.get("delivery_info") or p.get("delivery") or "일반배송"
        result.append({
            "product_name": p.get("name") or p.get("product_name", ""),
            "price": p.get("price", 0),
            "rating": p.get("rating"),
            "review_count": p.get("review_count"),
            "delivery": delivery_info,
            "delivery_fee": 0 if any(k in delivery_info for k in ("로켓", "무료")) else None,
            "platform": p.get("platform", ""),
            "image_url": p.get("image_url"),
            "product_url": p.get("product_url") or p.get("url", ""),
            "is_sold_out": p.get("is_sold_out", False),
            "raw": p,
        })
    return result


def _call_remote_mcp(
    query: str,
    platforms: list[str],
    sort: str,
    limit_per_platform: int = 3,
) -> list[dict[str, Any]]:
    """META_MCP_SERVER_URL SSE 엔드포인트 호출."""
    base_url = os.getenv("META_MCP_SERVER_URL", "").strip()
    if not base_url:
        return []

    params = urlencode({
        "query": query,
        "platforms": ",".join(platforms),
        "sort": sort,
        "limit_per_platform": limit_per_platform,
        "limit": len(platforms) * limit_per_platform,
    })
    endpoint = f"{urljoin(base_url.rstrip('/') + '/', 'sse')}?{params}"

    try:
        pass  # [mock_search] remote MCP SSE
        with urlopen(endpoint, timeout=20) as response:
            body = response.read().decode("utf-8")

        # SSE 파싱: search_result 이벤트의 data JSON 추출
        current_event = "message"
        data_lines: list[str] = []
        for line in body.splitlines():
            if line.startswith("event:"):
                current_event = line.removeprefix("event:").strip()
                data_lines = []
            elif line.startswith("data:"):
                data_lines.append(line.removeprefix("data:").strip())
            elif not line.strip() and current_event == "search_result" and data_lines:
                data = json.loads("\n".join(data_lines))
                products = data.get("products", [])
                print(f"[mock_search] remote MCP products={len(products)}")
                return _normalize(products)
        if current_event == "search_result" and data_lines:
            data = json.loads("\n".join(data_lines))
            products = data.get("products", [])
            return _normalize(products)
    except Exception as e:
        print(f"[mock_search] remote MCP failed: {e}")
    return []


def _fetch_naver_platform(
    platform: str,
    query: str,
    naver_sort: str,
    limit_per_platform: int,
    client_id: str,
    client_secret: str,
) -> list[dict[str, Any]]:
    """단일 플랫폼 네이버 API 호출 — ThreadPoolExecutor에서 병렬 실행."""
    search_query = f"{query} 컬리N마트" if platform == "kurly" and "컬리" not in query else query
    display = limit_per_platform * 3 if platform == "kurly" else limit_per_platform
    url = (
        "https://openapi.naver.com/v1/search/shop.json"
        f"?query={quote(search_query)}&display={display}&sort={naver_sort}"
    )
    req = Request(url, headers={
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret,
    })
    try:
        with urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"[mock_search] naver API failed platform={platform}: {e}")
        return []

    items = data.get("items") or []
    if platform == "kurly":
        items = [item for item in items if _is_kurly_item(item)]

    results = []
    for item in items[:limit_per_platform]:
        raw_url = item.get("link") or ""
        product_url = raw_url
        if platform == "kurly" and "kurly.com" not in raw_url:
            product_url = _kurly_search_url(query)
        results.append({
            "product_name": _strip_html(item.get("title") or ""),
            "price": int(item.get("lprice") or 0),
            "rating": None,
            "review_count": None,
            "delivery": "",
            "delivery_fee": None,
            "platform": platform,
            "image_url": item.get("image"),
            "product_url": product_url,
            "is_sold_out": False,
            "raw": item,
        })
    return results


def _call_naver_api(
    query: str,
    platforms: list[str],
    sort: str,
    limit_per_platform: int = 3,
) -> list[dict[str, Any]]:
    """네이버 쇼핑 API 병렬 호출 (Parallelization — naver/kurly 동시 요청)."""
    client_id = os.getenv("NAVER_CLIENT_ID")
    client_secret = os.getenv("NAVER_CLIENT_SECRET")
    if not client_id or not client_secret:
        return []

    naver_sort = NAVER_API_SORT_MAP.get(sort, "sim")
    valid_platforms = [p for p in platforms if p in ("naver", "kurly")]
    if not valid_platforms:
        return []

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(valid_platforms)) as executor:
        futures = {
            executor.submit(
                _fetch_naver_platform, p, query, naver_sort, limit_per_platform, client_id, client_secret
            ): p
            for p in valid_platforms
        }
        products: list[dict[str, Any]] = []
        for future in concurrent.futures.as_completed(futures):
            products.extend(future.result())

    if sort == "price_low":
        products.sort(key=lambda p: p.get("price") or 0)
    elif sort == "price_high":
        products.sort(key=lambda p: p.get("price") or 0, reverse=True)

    pass  # [mock_search] naver API done
    return products


def search_products(
    query: str,
    platforms: list[str],
    condition: str = "relevance",
    budget_max: int | None = None,
    limit_per_platform: int = 3,
) -> list[dict[str, Any]]:
    """
    platform_agent에서 호출하는 단일 진입점.

    우선순위:
    1. META_MCP_SERVER_URL → 원격 MCP SSE
    2. NAVER_CLIENT_ID    → 네이버 쇼핑 API
    3. 내장 mock 데이터   (항상 동작)
    """
    valid_platforms = [p for p in platforms if p in ("naver", "coupang", "kurly")]
    if not valid_platforms:
        valid_platforms = ["naver", "coupang"]

    sort = SORT_MAP.get(condition, "sim")

    # 1. 원격 MCP
    results = _call_remote_mcp(
        query=query, platforms=valid_platforms, sort=sort,
        limit_per_platform=limit_per_platform,
    )
    if results:
        if budget_max:
            results = [r for r in results if r.get("price", 0) <= budget_max]
        return results

    # 2. 네이버 API
    results = _call_naver_api(
        query=query, platforms=valid_platforms, sort=sort,
        limit_per_platform=limit_per_platform,
    )
    if results:
        if budget_max:
            results = [r for r in results if r.get("price", 0) <= budget_max]
        return results

    # 3. Mock 데이터 (fallback)
    pass  # [mock_search] using built-in mock data
    results = mock_search_product(
        query=query,
        platforms=valid_platforms,
        condition=condition,
        budget_max=budget_max,
    )
    # mock은 플랫폼별로 슬라이싱
    per_platform: dict[str, list] = {}
    for p in results:
        plat = p.get("platform", "")
        per_platform.setdefault(plat, []).append(p)
    return [p for items in per_platform.values() for p in items[:limit_per_platform]]
