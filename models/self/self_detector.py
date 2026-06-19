import re
from self_dict_v1 import (
    SELF_SUBJECT_LEMMAS, SELF_EMOTION_STEMS, SELF_COGNITION_STEMS, SELF_TRAIT_NOUNS,
    SELF_SUBJECT, SELF_EMOTION_VERB, SELF_COGNITION_VERB, SELF_TRAIT_NOUN,
)
from contribution_dict_v1 import (
    CONTRIBUTION_RESULT_STEMS, CONTRIBUTION_EXTERNAL_NOUNS,
    CONTRIBUTION_RESULT_NOUN, CONTRIBUTION_EXTERNAL_SUBJECT,
    has_number_expression,
)




SELF_DICT = {
    "A_subject": {
        "label":   "1인칭 주어",

        "weight":  0.1,
        "tags":    {"NP"},
        "lemmas":  SELF_SUBJECT_LEMMAS,
        "terms":   SELF_SUBJECT,
    },
    "B_emotion": {
        "label":   "감정 동사",
        "weight":  1.8,
        "tags":    {"VV", "VV-I", "VA"},
        "lemmas":  SELF_EMOTION_STEMS,
        "terms":   SELF_EMOTION_VERB,
    },
    "C_cognition": {
        "label":   "인식 동사",
        "weight":  0.8,
        "tags":    {"VV", "VV-I"},
        "lemmas":  SELF_COGNITION_STEMS,
        "terms":   SELF_COGNITION_VERB,
    },
    "D_trait": {
        "label":   "자기특성 명사",
        "weight":  2.0,
        "tags":    {"NNG", "NNP"},
        "lemmas":  SELF_TRAIT_NOUNS,
        "terms":   SELF_TRAIT_NOUN,
    },
}

CONTRIBUTION_DICT = {
    "E_result": {
        "label":   "결과 동사",
        "weight":  1.5,
        "tags":    {"VV", "VV-I"},
        "lemmas":  CONTRIBUTION_RESULT_STEMS,
        "terms":   CONTRIBUTION_RESULT_NOUN,
    },
    "F_external": {
        "label":   "외부 주어",
        "weight":  1.0,
        "tags":    {"NNG", "NNP"},
        "lemmas":  CONTRIBUTION_EXTERNAL_NOUNS,
        "terms":   CONTRIBUTION_EXTERNAL_SUBJECT,
    },
}


FEEDBACK_MAP = {

    "느끼":   "→ 결과·변화로. '이후 처리 시간이 30% 단축됐습니다'",
    "깨닫":   "→ 결과 행동으로. '이후 ○○를 도입했습니다'",
    "성장하": "→ 무엇이 얼마나 변했는지 수치로. '처리 속도 2배 향상'",
    "배우":   "→ 배움의 적용 결과로. '도입 후 오류율 40% 감소'",
    "뿌듯하": "→ 성과 수치로 대체. '팀 만족도 4.8점 달성'",
    "경험하": "→ 경험의 결과물로. '그 결과 ○○를 개선했습니다'",
    "열정":   "→ 열정을 증명하는 행동 사례로. '6개월간 주 3회 스터디 운영'",
    "책임감": "→ '기한을 한 번도 어기지 않았습니다'처럼 행동 증거로",

    "느꼈습니다":          "→ 결과·변화로. '이후 처리 시간이 30% 단축됐습니다'",
    "깨달았습니다":        "→ 결과 행동으로. '이후 ○○를 도입했습니다'",
    "성장했습니다":        "→ 무엇이 얼마나 변했는지 수치로. '처리 속도 2배 향상'",
    "배웠습니다":          "→ 배움의 적용 결과로. '도입 후 오류율 40% 감소'",
    "뿌듯했습니다":        "→ 성과 수치로 대체. '팀 만족도 4.8점 달성'",
    "경험했습니다":        "→ 경험의 결과물로. '그 결과 ○○를 개선했습니다'",
    "자신감을 얻었습니다": "→ 구체적 도전 결과로. '이후 ○○ 프로젝트를 단독으로 완수'",
}

COLOR_MAP = {
    "A_subject":   "#a5b4fc",
    "B_emotion":   "#fb923c",
    "C_cognition": "#fbbf24",
    "D_trait":     "#f87171",
    "E_result":    "#4ade80",
    "F_external":  "#38bdf8",
}



_kiwi = None

def _get_kiwi():
    global _kiwi
    if _kiwi is None:
        try:
            import shutil, tempfile, os
            import kiwipiepy_model
            from kiwipiepy import Kiwi

            src = kiwipiepy_model.get_model_path()
            tmp = tempfile.mkdtemp(prefix="kiwi_model_")
            for fname in os.listdir(src):
                full = os.path.join(src, fname)
                if os.path.isfile(full):
                    shutil.copy2(full, tmp)

            _kiwi = Kiwi(model_path=tmp)
        except Exception:
            _kiwi = False
    return _kiwi if _kiwi is not False else None




def detect_self_language(text: str) -> dict:
    self_hits = _find_hits(text, SELF_DICT)
    contrib_hits = _find_hits(text, CONTRIBUTION_DICT)
    number_hits = has_number_expression(text)

    self_hits = _deduplicate(self_hits)
    contrib_hits = _deduplicate(contrib_hits)

    self_score = _calc_self_score(text, self_hits)
    contrib_score = _calc_contribution_score(contrib_hits, number_hits)
    grade, summary = _grade(self_score, contrib_score)

    all_hits = self_hits + contrib_hits

    return {
        "role":                   "ROLE_04",
        "metric":                 "self_vs_contribution",
        "self_score":             self_score,
        "contribution_score":     contrib_score,
        "grade":                  grade,
        "summary":                summary,
        "self_hit_count":         len(self_hits),
        "contribution_hit_count": len(contrib_hits),
        "number_count":           len(number_hits),
        "self_hits":              self_hits,
        "contribution_hits":      contrib_hits,
        "number_hits":            number_hits,
        "highlighted_html":       _build_highlight(text, all_hits),
        "feedback_items":         _build_feedback(self_hits),
    }




def _find_hits(text: str, dictionary: dict) -> list:

    kiwi = _get_kiwi()
    if kiwi is not None:
        return _find_hits_kiwi(text, dictionary, kiwi)
    return _find_hits_regex(text, dictionary)


def _find_hits_kiwi(text: str, dictionary: dict, kiwi) -> list:
    tokens = kiwi.tokenize(text)
    hits = []
    for i, tok in enumerate(tokens):
        tag = tok.tag
        form = tok.form

        for cat_key, cat_info in dictionary.items():
            cat_tags = cat_info["tags"]
            cat_lemmas = cat_info["lemmas"]


            if tag in cat_tags and form in cat_lemmas:
                hits.append({
                    "term":     form,
                    "category": cat_key,
                    "label":    cat_info["label"],
                    "weight":   cat_info["weight"],
                    "start":    tok.start,
                    "end":      tok.start + tok.len,
                })


            elif tag == "XSV" and "VV" in cat_tags and i > 0:
                prev = tokens[i - 1]
                if prev.tag == "NNG":
                    compound = prev.form + form
                    if compound in cat_lemmas:
                        hits.append({
                            "term":     compound,
                            "category": cat_key,
                            "label":    cat_info["label"],
                            "weight":   cat_info["weight"],
                            "start":    prev.start,
                            "end":      tok.start + tok.len,
                        })

    return hits


def _find_hits_regex(text: str, dictionary: dict) -> list:

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

    hits.sort(key=lambda x: (x["start"], -(x["end"] - x["start"])))
    result, last_end = [], -1
    for h in hits:
        if h["start"] >= last_end:
            result.append(h)
            last_end = h["end"]
    return result




def _sentence_count(text: str) -> int:
    nida = len(re.findall(r'니다', text))
    punct = len(re.findall(r'[.!?]\s', text))
    return max(nida + punct, 1)


def _calc_self_score(text: str, hits: list) -> int:
    sent = _sentence_count(text)
    weighted_sum = sum(h["weight"] for h in hits)
    density = weighted_sum / sent
    return min(100, round(density * 20))


def _calc_contribution_score(hits: list, number_hits: list) -> int:
    weighted_sum = sum(h["weight"] for h in hits)
    number_bonus = len(number_hits) * 2.0
    raw = weighted_sum + number_bonus
    return min(100, round(raw * 5))


def _grade(self_score: int, contrib_score: int) -> tuple:
    if contrib_score >= 80:
        return "A", "기여 중심 표현이 뚜렷합니다. 성과가 잘 드러납니다."
    elif contrib_score >= 8:
        return "B", "기여 표현이 우세하나 자기중심 표현이 일부 섞여 있습니다."
    elif (
        (self_score >= 4 and contrib_score == 0)
        or (self_score >= 8 and contrib_score <= 4)
        or self_score >= 16
    ):
        return "D", "자기중심 표현이 지배적입니다. 결과·수치·외부 주어로 재작성하세요."
    else:
        return "C", "자기중심·기여 표현이 혼재합니다. 수치와 결과 중심으로 보완하세요."




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
