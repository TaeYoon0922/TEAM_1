#규칙 기반 탐지 && 밀도 점수화

import re
from hedge_dict_v1 import HEDGE_EFFORT, HEDGE_QUANTIFIER, HEDGE_UNCERTAINTY
from abstract_dict_v1 import ABSTRACT_TRAIT, ABSTRACT_GROWTH, EMOTION_VERB

# 전체 사전 통합
HEDGE_DICT = {
    "A_effort":     {"label": "노력·의지형",   "weight": 1.0, "terms": HEDGE_EFFORT},
    "B_quantifier": {"label": "양화 모호어",   "weight": 0.8, "terms": HEDGE_QUANTIFIER},
    "C_abstract":   {"label": "추상명사",      "weight": 1.2, "terms": ABSTRACT_TRAIT + ABSTRACT_GROWTH},
    "D_emotion":    {"label": "감정형 동사",   "weight": 1.2, "terms": EMOTION_VERB},
    "E_uncertainty":{"label": "불확실성 표지", "weight": 1.5, "terms": HEDGE_UNCERTAINTY},
}

FEEDBACK_MAP = {
    "열심히":       "→ 수치로 대체. 예: '주 5일 2시간씩 추가 학습'",
    "최선을":       "→ 구체적 행동으로. 예: '3회 프로토타입 제작 후 피드백 반영'",
    "다양한":       "→ 개수 명시. 예: '5개 프로젝트에서'",
    "여러":         "→ 횟수 명시. 예: '3차례 반복 실험을 통해'",
    "열정":         "→ 열정을 증명하는 행동 사례로 교체",
    "책임감":       "→ '기한을 한 번도 어기지 않았습니다'처럼 행동 증거로",
    "것 같습니다":  "→ 단정형으로. '~했습니다 / ~입니다'",
    "느꼈습니다":   "→ 결과·변화로. '이후 처리 시간이 30% 단축됐습니다'",
    "깨달았습니다": "→ 결과 행동으로. '이후 ○○를 도입했습니다'",
    "생각합니다":   "→ '~입니다'로 단정. 주장은 확신 있게",
    "노력했습니다": "→ 노력의 결과물로. 예: '그 결과 ○○를 달성했습니다'",
    "성장":         "→ 무엇이 얼마나 변했는지 수치로. 예: '처리 속도 2배 향상'",
}

COLOR_MAP = {
    "A_effort":      "#c084fc",
    "B_quantifier":  "#6ee7b7",
    "C_abstract":    "#fbbf24",
    "D_emotion":     "#fb923c",
    "E_uncertainty": "#f87171",
}


def detect_hedge_expressions(text: str) -> dict:
    """
    메인 함수 — ROLE 05 analyzer.py에서 이걸 호출
    
    Returns:
        score          : int 0~100 (높을수록 명확한 표현)
        grade          : str A/B/C/D
        summary        : str 한줄 요약
        hit_count      : int 탐지된 표현 수
        by_category    : dict 카테고리별 탐지 수
        highlighted_html: str 하이라이트된 HTML
        feedback_items : list [{original, category, suggestion}]
    """
    hits = _find_hits(text)
    hits = _deduplicate(hits)

    score = _calc_score(text, hits)
    grade, summary = _grade(score)

    return {
        "role": "ROLE_03",
        "metric": "hedge_density",
        "score": score,
        "grade": grade,
        "summary": summary,
        "hit_count": len(hits),
        "hits": hits,
        "by_category": _count_by_category(hits),
        "highlighted_html": _build_highlight(text, hits),
        "feedback_items": _build_feedback(hits),
    }


# ── 내부 함수들 ──────────────────────────────────────────

def _find_hits(text: str) -> list:
    hits = []
    for cat_key, cat_info in HEDGE_DICT.items():
        for term in cat_info["terms"]:
            for m in re.finditer(re.escape(term), text):
                hits.append({
                    "term":     term,
                    "category": cat_key,
                    "label":    cat_info["label"],
                    "weight":   cat_info["weight"],
                    "start":    m.start(),
                    "end":      m.end(),
                })
    return hits


def _deduplicate(hits: list) -> list:
    """겹치는 span 제거 — 긴 매칭 우선"""
    hits.sort(key=lambda x: (x["start"], -(x["end"] - x["start"])))
    result, last_end = [], -1
    for h in hits:
        if h["start"] >= last_end:
            result.append(h)
            last_end = h["end"]
    return result


def _calc_score(text: str, hits: list) -> int:
    token_count = max(len(text.split()), 1)
    weighted_sum = sum(h["weight"] for h in hits)
    weighted_density = weighted_sum / token_count
    score = max(0, 100 - round(weighted_density * 40))
    return score


def _grade(score: int) -> tuple:
    if score >= 80:
        return "A", "표현이 구체적이고 명확합니다."
    elif score >= 60:
        return "B", "일부 모호 표현이 있으나 전반적으로 양호합니다."
    elif score >= 40:
        return "C", "헤지 표현이 다수 발견됩니다. 구체화가 필요합니다."
    else:
        return "D", "대부분 문장이 모호합니다. 수치·행동 중심으로 재작성하세요."


def _count_by_category(hits: list) -> dict:
    counts = {k: 0 for k in HEDGE_DICT}
    for h in hits:
        counts[h["category"]] += 1
    return counts


def _build_highlight(text: str, hits: list) -> str:
    result, prev = [], 0
    for h in sorted(hits, key=lambda x: x["start"]):
        result.append(text[prev:h["start"]])
        color = COLOR_MAP.get(h["category"], "#e5e7eb")
        result.append(
            f'<mark style="background:{color};border-radius:3px;'
            f'padding:1px 4px;" title="{h["label"]}">'
            f'{text[h["start"]:h["end"]]}</mark>'
        )
        prev = h["end"]
    result.append(text[prev:])
    return "".join(result)


def _build_feedback(hits: list) -> list:
    feedbacks, seen = [], set()
    for h in hits:
        if h["term"] in FEEDBACK_MAP and h["term"] not in seen:
            feedbacks.append({
                "original":   h["term"],
                "category":   h["label"],
                "suggestion": FEEDBACK_MAP[h["term"]],
            })
            seen.add(h["term"])
    return feedbacks