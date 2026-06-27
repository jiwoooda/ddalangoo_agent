"""
LangSmith 평가 함수 모음.

evaluator 시그니처: (outputs: dict, reference_outputs: dict) -> dict
  outputs          : predict 함수의 반환값
  reference_outputs: JSON 파일 case의 output 필드

반환:
  {"key": "metric_name", "score": 0.0~1.0, "comment": "..."}
"""
from __future__ import annotations

# ── 어르신 친화도 기준 (response_agent와 동일) ──────────────────────────
_ELDERLY_FORBIDDEN = ["플랫폼", "최저가", "가성비", "혜택", "할인율", "할인가", "프로모션"]
_MAX_SENTENCE_LEN = 35


def _check_elderly_friendly(text: str) -> tuple[bool, str]:
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    if len(sentences) > 3:
        return False, f"문장 수 과다({len(sentences)}개)"
    for s in sentences:
        if len(s) > _MAX_SENTENCE_LEN:
            return False, f"긴 문장({len(s)}자): {s[:20]}..."
    found = [w for w in _ELDERLY_FORBIDDEN if w in text]
    if found:
        return False, f"어려운 단어: {', '.join(found)}"
    return True, ""


# ── intent_agent 평가기 ──────────────────────────────────────────────────

def intent_accuracy_evaluator(outputs: dict, reference_outputs: dict) -> dict:
    """intent 필드 정확 일치 여부 (0 or 1). 구포맷(expected_intent)/신포맷(intent) 모두 지원."""
    predicted = (outputs or {}).get("intent", "")
    ref = reference_outputs or {}
    expected = ref.get("expected_intent") or ref.get("intent", "")
    score = 1.0 if predicted == expected else 0.0
    return {
        "key": "intent_accuracy",
        "score": score,
        "comment": f"predicted={predicted!r}, expected={expected!r}",
    }


def keyword_overlap_evaluator(outputs: dict, reference_outputs: dict) -> dict:
    """keywords 필드 overlap F1. 구포맷(expected_keywords)/신포맷(keywords) 모두 지원."""
    ref = reference_outputs or {}
    expected_kws = ref.get("expected_keywords") or ref.get("keywords")
    if not expected_kws:
        return {"key": "keyword_overlap", "score": None}

    predicted_kws = set(k.lower() for k in (outputs.get("keywords") or []))
    expected_set = set(k.lower() for k in expected_kws)

    tp = len(predicted_kws & expected_set)
    precision = tp / len(predicted_kws) if predicted_kws else 0.0
    recall = tp / len(expected_set) if expected_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "key": "keyword_overlap",
        "score": round(f1, 3),
        "comment": f"predicted={sorted(predicted_kws)}, expected={sorted(expected_set)}",
    }


def needs_clarification_evaluator(outputs: dict, reference_outputs: dict) -> dict:
    """needs_clarification 필드 정확도 (신포맷 전용)."""
    ref = reference_outputs or {}
    expected = ref.get("needs_clarification")
    if expected is None:
        return {"key": "needs_clarification_accuracy", "score": None}
    predicted = bool((outputs or {}).get("needs_clarification", False))
    score = 1.0 if predicted == expected else 0.0
    return {
        "key": "needs_clarification_accuracy",
        "score": score,
        "comment": f"predicted={predicted}, expected={expected}",
    }


# ── product_agent 평가기 ─────────────────────────────────────────────────

def product_ranking_evaluator(outputs: dict, reference_outputs: dict) -> dict:
    """
    조건별 랭킹 정확도. outputs['top_product']가 expected 기준을 충족하는지 검사.

    검사 우선순위:
    1. expected_top_product  → 이름 완전 일치
    2. expected_top_price_max  → top 가격 ≤ 기준값
    3. expected_top_review_min → top 리뷰 수 ≥ 기준값
    4. expected_top_rating_min → top 평점 ≥ 기준값
    5. expected_top_delivery   → top delivery 필드에 기준 문자열 포함
    """
    top = (outputs or {}).get("top_product")
    if not top:
        return {"key": "ranking_accuracy", "score": 0.0, "comment": "top_product 없음"}

    ref = reference_outputs or {}

    # 1. 이름 일치
    if expected_name := ref.get("expected_top_product"):
        match = top.get("product_name", "") == expected_name
        return {
            "key": "ranking_accuracy",
            "score": 1.0 if match else 0.0,
            "comment": f"top={top.get('product_name')!r}, expected={expected_name!r}",
        }

    # 2. 최저가 — 가격 상한
    if (price_max := ref.get("expected_top_price_max")) is not None:
        top_price = top.get("price", 999_999_999)
        ok = top_price <= price_max
        return {
            "key": "ranking_accuracy",
            "score": 1.0 if ok else 0.0,
            "comment": f"top_price={top_price:,}원, max={price_max:,}원",
        }

    # 3. 인기순 — 리뷰 수 하한
    if (review_min := ref.get("expected_top_review_min")) is not None:
        top_reviews = top.get("review_count", 0)
        ok = top_reviews >= review_min
        return {
            "key": "ranking_accuracy",
            "score": 1.0 if ok else 0.0,
            "comment": f"top_reviews={top_reviews:,}, min={review_min:,}",
        }

    # 4. 리뷰좋은 — 평점 하한
    if (rating_min := ref.get("expected_top_rating_min")) is not None:
        top_rating = top.get("rating", 0.0)
        ok = top_rating >= rating_min
        return {
            "key": "ranking_accuracy",
            "score": 1.0 if ok else 0.0,
            "comment": f"top_rating={top_rating}, min={rating_min}",
        }

    # 5. 무료배송/빠른배송 — delivery 문자열 포함
    if expected_delivery := ref.get("expected_top_delivery"):
        top_delivery = top.get("delivery", "")
        ok = expected_delivery in top_delivery
        return {
            "key": "ranking_accuracy",
            "score": 1.0 if ok else 0.0,
            "comment": f"top_delivery={top_delivery!r}, expected_contains={expected_delivery!r}",
        }

    # 기준 없음 → 상품 존재 여부만 검사
    return {"key": "ranking_accuracy", "score": 1.0, "comment": "기준 없음 — top_product 존재"}


# ── response_agent 평가기 ────────────────────────────────────────────────

def elderly_friendliness_evaluator(outputs: dict, reference_outputs: dict) -> dict:
    """explanation 필드의 어르신 친화도 (0 or 1)."""
    explanation = (outputs or {}).get("explanation", "")
    if not explanation:
        return {"key": "elderly_friendliness", "score": None, "comment": "explanation 없음"}

    ok, reason = _check_elderly_friendly(explanation)
    return {
        "key": "elderly_friendliness",
        "score": 1.0 if ok else 0.0,
        "comment": reason or "통과",
    }


# ── 평가기 묶음 ──────────────────────────────────────────────────────────
INTENT_EVALUATORS = [intent_accuracy_evaluator, keyword_overlap_evaluator, needs_clarification_evaluator]
PRODUCT_EVALUATORS = [product_ranking_evaluator]
RESPONSE_EVALUATORS = [elderly_friendliness_evaluator]
E2E_EVALUATORS = [intent_accuracy_evaluator, elderly_friendliness_evaluator]
