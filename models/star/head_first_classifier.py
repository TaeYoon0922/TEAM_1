try:
    from head_first_rules import BAD_START_KEYWORDS, HEAD_KEYWORDS, analyze_head_first, split_sentences
except ImportError:
    from .head_first_rules import BAD_START_KEYWORDS, HEAD_KEYWORDS, analyze_head_first, split_sentences


DEFAULT_TOPIC_KEYWORDS = [
    "역량", "경험", "강점", "성과", "문제", "해결", "개선", "기여",
    "목표", "역할", "분석", "설계", "도입", "단축", "향상", "달성",
]


def classify_head_first(text: str, topic_keywords: list[str] | None = None) -> dict:

    sentences = split_sentences(text)
    keywords = normalize_keywords(topic_keywords or DEFAULT_TOPIC_KEYWORDS)
    base = analyze_head_first(text)

    if not sentences:
        return {
            "metric": "핵심 주장 명확성",
            "classifier": "core_claim_binary",
            "first_sentence": "",
            "is_head_first": 0,
            "topic_in_first": False,
            "display": "❌",
            "matched_keywords": [],
            "bad_start_keywords": [],
            "head_first_score": 0,
            "decision_boundary": base.get("decision_boundary", 35),
            "reason": "문장이 없습니다.",
        }

    first_sentence = sentences[0]
    matched_keywords = find_matches(first_sentence, keywords)
    claim_matches = find_matches(first_sentence, HEAD_KEYWORDS)
    bad_matches = find_matches(first_sentence, BAD_START_KEYWORDS)
    matches = sorted(set(matched_keywords + claim_matches))

    return {
        "metric": "핵심 주장 명확성",
        "classifier": "core_claim_binary",
        "first_sentence": first_sentence,
        "is_head_first": base["is_head_first"],
        "topic_in_first": bool(matches),
        "display": base["display"],
        "matched_keywords": matches,
        "bad_start_keywords": bad_matches,
        "head_first_score": base["head_first_score"],
        "decision_boundary": base.get("decision_boundary", 35),
        "reason": base["reason"],
    }


def normalize_keywords(keywords: list[str]) -> list[str]:
    return [keyword.strip() for keyword in keywords if keyword and keyword.strip()]


def find_matches(text: str, keywords: list[str]) -> list[str]:
    return [keyword for keyword in keywords if keyword in text]
