"""
LangSmith 평가 함수 모음.

evaluator 시그니처: (outputs: dict, reference_outputs: dict) -> dict
  outputs          : predict 함수의 반환값
  reference_outputs: JSON 파일 case의 output 필드

반환:
  {"key": "metric_name", "score": 0.0~1.0, "comment": "..."}
"""
from __future__ import annotations
import math
import re

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


def schema_compliance_evaluator(outputs: dict, reference_outputs: dict) -> dict:
    """Pydantic structured output 성공 여부. predict 예외 시 _schema_ok=false."""
    ok = not (outputs or {}).get("_error") and (outputs or {}).get("_schema_ok", True)
    return {
        "key": "schema_compliance",
        "score": 1.0 if ok else 0.0,
        "comment": (outputs or {}).get("_error", "통과"),
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


def tool_call_success_evaluator(outputs: dict, reference_outputs: dict) -> dict:
    ok = bool((outputs or {}).get("tool_call_success"))
    return {
        "key": "tool_call_success_rate",
        "score": 1.0 if ok else 0.0,
        "comment": (outputs or {}).get("tool_call_error") or ("통과" if ok else "tool call 실패"),
    }


def mrr_evaluator(outputs: dict, reference_outputs: dict) -> dict:
    expected = (reference_outputs or {}).get("expected_top_product")
    ranked = (outputs or {}).get("ranked_products") or []
    if not expected or not ranked:
        return {"key": "mrr", "score": 0.0, "comment": "expected 또는 ranked 없음"}
    for idx, product in enumerate(ranked, start=1):
        if product.get("product_name") == expected:
            return {"key": "mrr", "score": round(1.0 / idx, 3), "comment": f"rank={idx}"}
    return {"key": "mrr", "score": 0.0, "comment": "정답 상품 없음"}


def ndcg_at_3_evaluator(outputs: dict, reference_outputs: dict) -> dict:
    ref = reference_outputs or {}
    relevant = ref.get("expected_ranking") or ref.get("relevant_products") or []
    ranked = (outputs or {}).get("ranked_products") or []
    if not relevant or not ranked:
        return {"key": "ndcg_at_3", "score": None, "comment": "expected_ranking 없음"}
    rel_map = {name: max(len(relevant) - i, 1) for i, name in enumerate(relevant)}
    gains = []
    for idx, product in enumerate(ranked[:3], start=1):
        rel = rel_map.get(product.get("product_name"), 0)
        gains.append((2**rel - 1) / math.log2(idx + 1))
    ideal_rels = sorted(rel_map.values(), reverse=True)[:3]
    ideal = sum((2**rel - 1) / math.log2(idx + 1) for idx, rel in enumerate(ideal_rels, start=1))
    score = sum(gains) / ideal if ideal else 0.0
    return {"key": "ndcg_at_3", "score": round(score, 3), "comment": f"expected={relevant[:3]}"}


def condition_adherence_evaluator(outputs: dict, reference_outputs: dict) -> dict:
    rule = (reference_outputs or {}).get("condition_rule")
    top = (outputs or {}).get("top_product")
    ranked = (outputs or {}).get("ranked_products") or []
    if not rule or not top or not ranked:
        return {"key": "condition_adherence", "score": None}

    if rule == "price_lowest":
        best = min(ranked, key=lambda p: p.get("price") or 999_999_999)
        ok = top.get("product_name") == best.get("product_name")
    elif rule == "review_count_highest":
        best = max(ranked, key=lambda p: p.get("review_count") or 0)
        ok = top.get("product_name") == best.get("product_name")
    elif rule == "rating_highest":
        best = max(ranked, key=lambda p: p.get("rating") or 0)
        ok = top.get("product_name") == best.get("product_name")
    elif rule == "free_shipping":
        ok = top.get("delivery_fee") == 0 or "무료" in str(top.get("delivery") or "")
    elif rule == "delivery_fastest":
        ok = any(k in str(top.get("delivery") or "") for k in ("오늘", "내일", "새벽", "로켓"))
    else:
        return {"key": "condition_adherence", "score": None, "comment": f"unknown rule={rule}"}

    return {"key": "condition_adherence", "score": 1.0 if ok else 0.0, "comment": rule}


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


def reflection_pass_evaluator(outputs: dict, reference_outputs: dict) -> dict:
    if "reflection_passed" in (outputs or {}):
        ok = bool(outputs["reflection_passed"])
    else:
        ok, _ = _check_elderly_friendly((outputs or {}).get("explanation", ""))
    return {"key": "reflection_pass_rate", "score": 1.0 if ok else 0.0}


def haiku_fallback_evaluator(outputs: dict, reference_outputs: dict) -> dict:
    used = bool((outputs or {}).get("haiku_fallback"))
    return {"key": "haiku_fallback_rate", "score": 1.0 if used else 0.0}


def g_eval_elderly_evaluator(outputs: dict, reference_outputs: dict) -> dict:
    """G-Eval 대체 로컬 루브릭. DeepEval judge 연결 전에도 추세 확인 가능."""
    text = (outputs or {}).get("explanation", "")
    if not text:
        return {"key": "g_eval_elderly", "score": 0.0, "comment": "explanation 없음"}
    ok, reason = _check_elderly_friendly(text)
    has_price = bool(re.search(r"\d", text))
    score_5 = 5
    if not ok:
        score_5 -= 2
    if not has_price:
        score_5 -= 1
    score_5 = max(1, min(5, score_5))
    return {
        "key": "g_eval_elderly",
        "score": score_5 / 5,
        "comment": f"{score_5}/5; {reason or '짧고 쉬움'}",
    }


def preference_adherence_evaluator(outputs: dict, reference_outputs: dict) -> dict:
    """선호도 반영 여부: expected_preference_terms 중 explanation에 포함된 비율."""
    hints = (reference_outputs or {}).get("expected_preference_terms")
    if not hints:
        return {"key": "preference_adherence", "score": None}
    text = (outputs or {}).get("explanation", "")
    hits = sum(1 for h in hints if h in text)
    return {
        "key": "preference_adherence",
        "score": round(hits / len(hints), 3),
        "comment": f"{hits}/{len(hints)} 반영",
    }


def latency_evaluator(outputs: dict, reference_outputs: dict) -> dict:
    latency_ms = (outputs or {}).get("latency_ms")
    if latency_ms is None:
        return {"key": "latency_ms", "score": None}
    return {"key": "latency_ms", "score": float(latency_ms), "comment": f"{latency_ms:.1f}ms"}


def ttft_evaluator(outputs: dict, reference_outputs: dict) -> dict:
    ttft_ms = (outputs or {}).get("ttft_ms", (outputs or {}).get("latency_ms"))
    if ttft_ms is None:
        return {"key": "ttft_ms", "score": None}
    return {"key": "ttft_ms", "score": float(ttft_ms), "comment": f"{ttft_ms:.1f}ms"}


# ── context_agent 평가기 ────────────────────────────────────────────────

def preferred_brands_evaluator(outputs: dict, reference_outputs: dict) -> dict:
    """expected_preferred_brands가 preferred_brands[].brand 목록에 모두 포함되는지 확인."""
    expected = (reference_outputs or {}).get("expected_preferred_brands")
    if not expected:
        return {"key": "preferred_brands", "score": None}
    actual_brands = {b.get("brand") for b in ((outputs or {}).get("preferred_brands") or [])}
    hits = [b for b in expected if b in actual_brands]
    score = len(hits) / len(expected)
    return {
        "key": "preferred_brands",
        "score": round(score, 3),
        "comment": f"found={hits}, missing={[b for b in expected if b not in actual_brands]}",
    }


def preferred_platform_evaluator(outputs: dict, reference_outputs: dict) -> dict:
    """preferred_platform이 expected_preferred_platform과 일치하는지 확인."""
    expected = (reference_outputs or {}).get("expected_preferred_platform")
    if not expected:
        return {"key": "preferred_platform", "score": None}
    actual = (outputs or {}).get("preferred_platform")
    ok = actual == expected
    return {
        "key": "preferred_platform",
        "score": 1.0 if ok else 0.0,
        "comment": f"actual={actual!r}, expected={expected!r}",
    }


def price_range_evaluator(outputs: dict, reference_outputs: dict) -> dict:
    """price_range의 min/max/avg가 expected_price_range와 일치하는지 확인."""
    expected = (reference_outputs or {}).get("expected_price_range")
    if not expected:
        return {"key": "price_range", "score": None}
    actual = (outputs or {}).get("price_range") or {}
    checks = []
    for field in ("min", "max", "avg"):
        if field in expected:
            checks.append(actual.get(field) == expected[field])
    if not checks:
        return {"key": "price_range", "score": None}
    score = sum(checks) / len(checks)
    return {
        "key": "price_range",
        "score": round(score, 3),
        "comment": f"actual={actual}, expected={expected}",
    }


def repurchase_patterns_evaluator(outputs: dict, reference_outputs: dict) -> dict:
    """expected_repurchase_patterns 항목이 repurchase_patterns에 포함되는지 확인."""
    expected = (reference_outputs or {}).get("expected_repurchase_patterns")
    if not expected:
        return {"key": "repurchase_patterns", "score": None}
    actual = set((outputs or {}).get("repurchase_patterns") or [])
    hits = [p for p in expected if p in actual]
    score = len(hits) / len(expected)
    return {
        "key": "repurchase_patterns",
        "score": round(score, 3),
        "comment": f"found={hits}, missing={[p for p in expected if p not in actual]}",
    }


def keyword_history_evaluator(outputs: dict, reference_outputs: dict) -> dict:
    """keyword_history 항목 수가 expected_keyword_history_min 이상인지 확인."""
    min_count = (reference_outputs or {}).get("expected_keyword_history_min")
    if min_count is None:
        return {"key": "keyword_history_count", "score": None}
    actual = len((outputs or {}).get("keyword_history") or [])
    ok = actual >= min_count
    return {
        "key": "keyword_history_count",
        "score": 1.0 if ok else 0.0,
        "comment": f"actual={actual}, min={min_count}",
    }


def keyword_history_products_evaluator(outputs: dict, reference_outputs: dict) -> dict:
    """expected_keyword_history_products가 keyword_history[].product_name에 포함되는지 확인."""
    expected = (reference_outputs or {}).get("expected_keyword_history_products")
    if not expected:
        return {"key": "keyword_history_products", "score": None}
    actual_names = {h.get("product_name") for h in ((outputs or {}).get("keyword_history") or [])}
    hits = [p for p in expected if p in actual_names]
    score = len(hits) / len(expected)
    return {
        "key": "keyword_history_products",
        "score": round(score, 3),
        "comment": f"found={hits}, missing={[p for p in expected if p not in actual_names]}",
    }


def context_faithfulness_evaluator(outputs: dict, reference_outputs: dict) -> dict:
    if (reference_outputs or {}).get("expected_cold_start"):
        return {"key": "faithfulness", "score": None, "comment": "cold-start N/A"}
    text = (outputs or {}).get("summary", "")
    expected_terms = (reference_outputs or {}).get("expected_terms") or []
    if not expected_terms:
        return {"key": "faithfulness", "score": None}
    hits = sum(1 for term in expected_terms if term in text)
    return {
        "key": "faithfulness",
        "score": round(hits / len(expected_terms), 3),
        "comment": f"covered={hits}/{len(expected_terms)}",
    }


def context_coverage_evaluator(outputs: dict, reference_outputs: dict) -> dict:
    text = (outputs or {}).get("summary", "")
    expected_terms = (reference_outputs or {}).get("expected_terms") or []
    if not expected_terms:
        return {"key": "coverage", "score": None}
    hits = [term for term in expected_terms if term in text]
    return {"key": "coverage", "score": 1.0 if len(hits) == len(expected_terms) else 0.0, "comment": f"{hits}"}


def context_cold_start_evaluator(outputs: dict, reference_outputs: dict) -> dict:
    if not (reference_outputs or {}).get("expected_cold_start"):
        return {"key": "cold_start_pass", "score": None}
    ok = not (outputs or {}).get("summary") and (outputs or {}).get("purchase_count", 0) == 0
    return {"key": "cold_start_pass", "score": 1.0 if ok else 0.0}


def context_conciseness_evaluator(outputs: dict, reference_outputs: dict) -> dict:
    text = (outputs or {}).get("summary", "")
    if not text:
        return {"key": "conciseness", "score": None}
    sentences = [s for s in re.split(r"[.!?。]\s*", text) if s.strip()]
    return {"key": "conciseness", "score": 1.0 if len(sentences) <= 3 else 0.0, "comment": f"{len(sentences)} sentences"}


# ── 평가기 묶음 ──────────────────────────────────────────────────────────
INTENT_EVALUATORS = [
    intent_accuracy_evaluator,
    keyword_overlap_evaluator,
    needs_clarification_evaluator,
    schema_compliance_evaluator,
    latency_evaluator,
]
PRODUCT_EVALUATORS = [
    tool_call_success_evaluator,
    mrr_evaluator,
    ndcg_at_3_evaluator,
    condition_adherence_evaluator,
    latency_evaluator,
]
RESPONSE_EVALUATORS = [
    reflection_pass_evaluator,
    haiku_fallback_evaluator,
    elderly_friendliness_evaluator,
    g_eval_elderly_evaluator,
    preference_adherence_evaluator,
    ttft_evaluator,
]
CONTEXT_EVALUATORS = [
    preferred_brands_evaluator,
    preferred_platform_evaluator,
    price_range_evaluator,
    repurchase_patterns_evaluator,
    keyword_history_evaluator,
    keyword_history_products_evaluator,
    context_faithfulness_evaluator,
    context_coverage_evaluator,
    context_cold_start_evaluator,
    context_conciseness_evaluator,
    latency_evaluator,
]
E2E_EVALUATORS = [intent_accuracy_evaluator, elderly_friendliness_evaluator]
