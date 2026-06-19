import re


HEAD_KEYWORDS = [
    "역량", "경험", "강점", "성과",
    "해결", "개선", "기여", "자신 있습니다",
    "갖추고 있습니다", "달성", "단축", "향상",
]

BAD_START_KEYWORDS = [
    "어렸을 때", "대학교", "처음에는",
    "당시", "예전에", "고등학교",
]

CORE_CLAIM_DECISION_BOUNDARY = 35


def split_sentences(text):
    sentences = re.split(r"(?<=[.!?。])\s+|(?<=다\.)\s+|(?<=요\.)\s+", str(text).strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def analyze_head_first(text):
    sentences = split_sentences(text)

    if not sentences:
        return {
            "metric": "핵심 주장 명확성",
            "first_sentence": "",
            "core_claim_clear": False,
            "display": "❌",
            "is_head_first": 0,
            "head_first_score": 0,
            "head_score": 0,
            "bad_score": 0,
            "reason": "문장이 없습니다.",
        }

    first = sentences[0]
    head_score = sum(1 for keyword in HEAD_KEYWORDS if keyword in first)
    bad_score = sum(1 for keyword in BAD_START_KEYWORDS if keyword in first)
    head_first_score = calc_core_claim_score(head_score, bad_score, first)
    is_clear = int(head_first_score >= CORE_CLAIM_DECISION_BOUNDARY)

    return {
        "metric": "핵심 주장 명확성",
        "first_sentence": first,
        "core_claim_clear": bool(is_clear),
        "display": "✅" if is_clear else "❌",
        "is_head_first": is_clear,
        "head_first_score": head_first_score,
        "head_score": head_score,
        "bad_score": bad_score,
        "decision_boundary": CORE_CLAIM_DECISION_BOUNDARY,
        "reason": (
            "첫 문장에 핵심 주장 신호가 있습니다."
            if is_clear
            else "첫 문장에 핵심 역량, 성과, 문제 해결 경험을 더 직접적으로 제시하세요."
        ),
    }


def calc_core_claim_score(head_score: int, bad_score: int, first_sentence: str = "") -> int:
    if head_score == 0 and bad_score == 0 and len(first_sentence.strip()) >= 40:
        return CORE_CLAIM_DECISION_BOUNDARY
    score = 30 + min(head_score, 4) * 20 - min(bad_score, 3) * 20
    return max(0, min(100, score))
