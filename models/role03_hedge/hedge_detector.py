# hedge_detector.py — Phase 2: KoNLPy 형태소 기반 탐지

import re
from hedge_dict_v1 import HEDGE_EFFORT, HEDGE_QUANTIFIER, HEDGE_UNCERTAINTY
from abstract_dict_v1 import ABSTRACT_TRAIT, ABSTRACT_GROWTH, EMOTION_VERB

# ── KoNLPy 초기화 ──────────────────────────────────────────
try:
    from konlpy.tag import Okt
    _okt = Okt()
    KONLPY_AVAILABLE = True
except Exception:
    _okt = None
    KONLPY_AVAILABLE = False

# ── 어간 기반 사전 (Phase 2) ────────────────────────────────
STEM_HEDGE_DICT = {
    "A_effort": {
        "label": "노력·의지형",
        "weight": 1.0,
        "stems": ["노력", "열심히","최선", "성실하다"]
    },
    "B_quantifier": {
        "label": "양화 모호어",
        "weight": 0.5,
        "stems": ["어느정도"]
    },
    "C_abstract": {
        "label": "추상명사",
        "weight": 1.2,
        "stems": ["열정", "책임감", "성장하다", "자기개발"]
    },
    "D_emotion": {
        "label": "감정·인식형 동사",
        "weight": 1.2,
        "stems": ["생각하다"]
    },
    "E_uncertainty": {
        "label": "불확실성 표지어",
        "weight": 1.5,
        "stems": ["같다","싶다"]
    },
}

# ── Phase 1 폴백 사전 (KoNLPy 없을 때) ─────────────────────
HEDGE_DICT = {
    "A_effort":      {"label": "노력·의지형",   "weight": 1.0, "terms": HEDGE_EFFORT},
    "B_quantifier":  {"label": "양화 모호어",   "weight": 0.8, "terms": HEDGE_QUANTIFIER},
    "C_abstract":    {"label": "추상명사",      "weight": 1.2, "terms": ABSTRACT_TRAIT + ABSTRACT_GROWTH},
    "D_emotion":     {"label": "감정·인식형 동사","weight": 1.2, "terms": EMOTION_VERB},
    "E_uncertainty": {"label": "불확실성 표지어","weight": 1.5, "terms": HEDGE_UNCERTAINTY},
}

COLOR_MAP = {
    "A_effort":      "#c084fc",
    "B_quantifier":  "#6ee7b7",
    "C_abstract":    "#fbbf24",
    "D_emotion":     "#fb923c",
    "E_uncertainty": "#f87171",
}

FEEDBACK_MAP = {
    "노력":     "→ 노력의 결과물로.",
    "열심히":   "→ 수치로 대체.",
    "최선":     "→ 구체적 행동으로.",
    "같다":     "→ 단정형으로. '~했습니다 / ~입니다'",
    "싶다":     "→ '~하겠습니다'로 의지 표현 강화",
    "생각하다": "→ '~입니다'로 단정. 주장은 확신 있게",
    "하고 싶":  "→ '~하겠습니다'로 의지 표현 강화",
    "것 같":    "→ 단정형으로. '~했습니다 / ~입니다'",
}


def detect_hedge_expressions(text: str) -> dict:
    """메인 함수 — ROLE 05 analyzer.py에서 호출"""
    hits = _find_hits(text)
    hits = _deduplicate(hits)

    score = _calc_score(text, hits)
    grade, summary = _grade(score)

    return {
        "role":             "ROLE_03",
        "metric":           "expression_clarity",
        "score":            score,
        "grade":            grade,
        "summary":          summary,
        "hit_count":        len(hits),
        "hits":             hits,
        "by_category":      _count_by_category(hits),
        "highlighted_html": _build_highlight(text, hits),
        "feedback_items":   _build_feedback(hits),
        "clarity_label":    f"모호 표현 {len(hits)}개 발견",
        "konlpy_used":      KONLPY_AVAILABLE,   # Phase 2 여부 확인용
    }


# ── 내부 함수 ───────────────────────────────────────────────

def _find_hits(text: str) -> list:
    if KONLPY_AVAILABLE:
        return _find_hits_konlpy(text)
    return _find_hits_fallback(text)


def _find_hits_konlpy(text: str) -> list:
    """Phase 2 — 형태소 어간 기반 탐지 + 예외 문자열 매칭 병행"""
    hits = []
    morphs = _okt.morphs(text, stem=True)

    for cat_key, cat_info in STEM_HEDGE_DICT.items():
        for stem in cat_info["stems"]:
            if stem in morphs:
                search_word = stem.replace("하다", "").replace("다", "")
                if not search_word:
                    search_word = stem
                for m in re.finditer(re.escape(search_word), text):
                    hits.append({
                        "term":     stem,
                        "category": cat_key,
                        "label":    cat_info["label"],
                        "weight":   cat_info["weight"],
                        "start":    m.start(),
                        "end":      m.end(),
                    })

    # KoNLPy가 잘못 분리하는 예외 표현 — 문자열 직접 매칭
    EXCEPTION_TERMS = {
        "깨달": ("D_emotion", "감정·인식형 동사", 1.2),
        "하고 싶": ("E_uncertainty", "불확실성 표지어", 1.5),
        "것 같": ("E_uncertainty", "불확실성 표지어", 1.5),
    }
    for term, (cat_key, label, weight) in EXCEPTION_TERMS.items():
        for m in re.finditer(re.escape(term), text):
            hits.append({
                "term":     term,
                "category": cat_key,
                "label":    label,
                "weight":   weight,
                "start":    m.start(),
                "end":      m.end(),
            })

    return hits


def _find_hits_fallback(text: str) -> list:
    """Phase 1 폴백 — 문자열 매칭"""
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
    return max(0, min(100, round(100 - (weighted_density * 55))))


def _grade(score: int) -> tuple:
    if score >= 70:
        return "A", "표현이 구체적이고 명확합니다."
    elif score >= 50:
        return "B", "일부 모호 표현이 있으나 전반적으로 양호합니다."
    elif score >= 30:
        return "C", "헤지 표현이 다수 발견됩니다. 구체화가 필요합니다."
    else:
        return "D", "대부분 문장이 모호합니다. 수치·행동 중심으로 재작성하세요."


def _count_by_category(hits: list) -> dict:
    counts = {k: 0 for k in STEM_HEDGE_DICT}
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
        term = h["term"]
        key = term if term in FEEDBACK_MAP else term.replace("하다", "").replace("다", "")
        if key in FEEDBACK_MAP and term not in seen:
            feedbacks.append({
                "original":   term,
                "category":   h["label"],
                "suggestion": FEEDBACK_MAP[key],
            })
            seen.add(term)
    return feedbacks