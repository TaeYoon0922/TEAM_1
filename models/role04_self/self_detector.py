# 자기중심 vs 기여중심 언어 탐지기
# ROLE 05 analyzer.py에서 detect_self_language()를 호출

import re
from self_dict_v1 import (
    SELF_SUBJECT, SELF_EMOTION_VERB, SELF_COGNITION_VERB, SELF_TRAIT_NOUN
)
from contribution_dict_v1 import (
    CONTRIBUTION_RESULT_NOUN, CONTRIBUTION_EXTERNAL_SUBJECT,
    has_number_expression
)

# ── 전체 사전 통합 ─────────────────────────────────────────
SELF_DICT = {
    "A_subject":   {"label": "1인칭 주어",   "weight": 0.8, "terms": SELF_SUBJECT},
    "B_emotion":   {"label": "감정 동사",    "weight": 1.2, "terms": SELF_EMOTION_VERB},
    "C_cognition": {"label": "인식 동사",    "weight": 1.0, "terms": SELF_COGNITION_VERB},
    "D_trait":     {"label": "자기특성 명사", "weight": 1.3, "terms": SELF_TRAIT_NOUN},
}

CONTRIBUTION_DICT = {
    "E_result":   {"label": "결과 동사",   "weight": 1.5, "terms": CONTRIBUTION_RESULT_NOUN},
    "F_external": {"label": "외부 주어",   "weight": 1.0, "terms": CONTRIBUTION_EXTERNAL_SUBJECT},
}

FEEDBACK_MAP = {
    "느꼈습니다":       "→ 결과·변화로. '이후 처리 시간이 30% 단축됐습니다'",
    "깨달았습니다":     "→ 결과 행동으로. '이후 ○○를 도입했습니다'",
    "성장했습니다":     "→ 무엇이 얼마나 변했는지 수치로. '처리 속도 2배 향상'",
    "배웠습니다":       "→ 배움의 적용 결과로. '도입 후 오류율 40% 감소'",
    "뿌듯했습니다":     "→ 성과 수치로 대체. '팀 만족도 4.8점 달성'",
    "열정":             "→ 열정을 증명하는 행동 사례로. '6개월간 주 3회 스터디 운영'",
    "책임감":           "→ '기한을 한 번도 어기지 않았습니다'처럼 행동 증거로",
    "경험했습니다":     "→ 경험의 결과물로. '그 결과 ○○를 개선했습니다'",
    "자신감을 얻었습니다": "→ 구체적 도전 결과로. '이후 ○○ 프로젝트를 단독으로 완수'",
}

COLOR_MAP = {
    "A_subject":   "#a5b4fc",  # 연보라 — 1인칭 주어
    "B_emotion":   "#fb923c",  # 주황 — 감정 동사
    "C_cognition": "#fbbf24",  # 노랑 — 인식 동사
    "D_trait":     "#f87171",  # 빨강 — 자기특성 명사
    "E_result":    "#4ade80",  # 초록 — 결과 동사 (긍정)
    "F_external":  "#38bdf8",  # 파랑 — 외부 주어 (긍정)
}


def detect_self_language(text: str) -> dict:
    """
    메인 함수 — ROLE 05 analyzer.py에서 이걸 호출

    Returns:
        self_score       : int 0~100 (높을수록 자기중심 표현 많음)
        contribution_score: int 0~100 (높을수록 기여중심 표현 많음)
        grade            : str A/B/C/D
        summary          : str 한줄 요약
        self_hits        : list 자기중심 탐지 목록
        contribution_hits: list 기여중심 탐지 목록
        number_hits      : list 수치 표현 목록
        highlighted_html : str 하이라이트 HTML
        feedback_items   : list [{original, category, suggestion}]
    """
    self_hits = _find_hits(text, SELF_DICT)
    contrib_hits = _find_hits(text, CONTRIBUTION_DICT)
    number_hits = has_number_expression(text)

    self_hits = _deduplicate(self_hits)
    contrib_hits = _deduplicate(contrib_hits)

    self_score = _calc_self_score(text, self_hits)
    contrib_score = _calc_contribution_score(text, contrib_hits, number_hits)
    grade, summary = _grade(self_score, contrib_score)

    all_hits = self_hits + contrib_hits

    return {
        "role": "ROLE_04",
        "metric": "self_vs_contribution",
        "self_score": self_score,
        "contribution_score": contrib_score,
        "grade": grade,
        "summary": summary,
        "self_hit_count": len(self_hits),
        "contribution_hit_count": len(contrib_hits),
        "number_count": len(number_hits),
        "self_hits": self_hits,
        "contribution_hits": contrib_hits,
        "number_hits": number_hits,
        "highlighted_html": _build_highlight(text, all_hits),
        "feedback_items": _build_feedback(self_hits),
    }


# ── 내부 함수들 ──────────────────────────────────────────────

def _find_hits(text: str, dictionary: dict) -> list:
    hits = []
    for cat_key, cat_info in dictionary.items():
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


def _calc_self_score(text: str, hits: list) -> int:
    """자기중심 점수: 높을수록 자기중심 표현 많음 (0~100)"""
    token_count = max(len(text.split()), 1)
    weighted_sum = sum(h["weight"] for h in hits)
    density = weighted_sum / token_count
    return min(100, round(density * 50))


def _calc_contribution_score(text: str, hits: list, number_hits: list) -> int:
    """기여중심 점수: 높을수록 기여중심 표현 많음 (0~100)"""
    token_count = max(len(text.split()), 1)
    weighted_sum = sum(h["weight"] for h in hits)
    number_bonus = len(number_hits) * 1.5  # 수치 표현 가산
    density = (weighted_sum + number_bonus) / token_count
    return min(100, round(density * 40))


def _grade(self_score: int, contrib_score: int) -> tuple:
    diff = contrib_score - self_score
    if diff >= 30:
        return "A", "기여 중심 표현이 뚜렷합니다. 성과가 잘 드러납니다."
    elif diff >= 10:
        return "B", "기여 표현이 우세하나 자기중심 표현이 일부 섞여 있습니다."
    elif diff >= -10:
        return "C", "자기중심·기여 표현이 혼재합니다. 수치와 결과 중심으로 보완하세요."
    else:
        return "D", "자기중심 표현이 지배적입니다. 결과·수치·외부 주어로 재작성하세요."


def _build_highlight(text: str, hits: list) -> str:
    all_hits = sorted(hits, key=lambda x: x["start"])
    result, prev = [], 0
    for h in all_hits:
        if h["start"] < prev:
            continue
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


def _build_feedback(self_hits: list) -> list:
    feedbacks, seen = [], set()
    for h in self_hits:
        if h["term"] in FEEDBACK_MAP and h["term"] not in seen:
            feedbacks.append({
                "original":   h["term"],
                "category":   h["label"],
                "suggestion": FEEDBACK_MAP[h["term"]],
            })
            seen.add(h["term"])
    return feedbacks
