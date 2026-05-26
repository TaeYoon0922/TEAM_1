import re

HEAD_KEYWORDS = [
    "역량", "경험", "강점", "성과",
    "해결", "개선", "기여", "자신 있습니다",
    "갖추고 있습니다"
]

BAD_START_KEYWORDS = [
    "어렸을 때", "대학교", "처음에는",
    "당시", "예전에", "고등학교"
]

def split_sentences(text):
    sentences = re.split(r'(?<=[.!?。])\s+|(?<=다\.)\s+|(?<=요\.)\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

def analyze_head_first(text):
    sentences = split_sentences(text)

    if not sentences:
        return {
            "is_head_first": 0,
            "reason": "문장이 없습니다."
        }

    first = sentences[0]

    head_score = sum(1 for kw in HEAD_KEYWORDS if kw in first)
    bad_score = sum(1 for kw in BAD_START_KEYWORDS if kw in first)

    is_head_first = 1 if head_score >= 1 and head_score > bad_score else 0

    return {
        "first_sentence": first,
        "is_head_first": is_head_first,
        "head_score": head_score,
        "bad_score": bad_score
    }