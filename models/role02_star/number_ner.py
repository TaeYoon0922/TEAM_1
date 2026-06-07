import re

RESULT_MARKERS = [
    "결과", "성과", "이를 통해", "그 결과", "최종적으로", "마침내",
]

RESULT_CHANGE_KEYWORDS = [
    "향상", "감소", "증가", "단축", "절감", "달성", "완료",
    "높였", "낮췄", "줄였", "늘렸", "개선율", "성공",
]

NUMBER_PATTERNS = {
    "RATIO": r"\d+(?:\.\d+)?\s*(?:%|퍼센트|프로)",
    "PERIOD": r"\d+(?:\.\d+)?\s*(?:년|개월|달|주|일|시간|분|초)",
    "COUNT": r"\d+(?:\.\d+)?\s*(?:명|개|건|회|번|차례|문항|페이지)",
    "SCORE": r"\d+(?:\.\d+)?\s*(?:점|등급|위|위권)",
    "MONEY": r"\d+(?:\.\d+)?\s*(?:원|만원|억원|천원)",
    "MULTIPLE": r"\d+(?:\.\d+)?\s*(?:배)",
    "GENERAL_NUMBER": r"\d+(?:\.\d+)?",
}

LABEL_PRIORITY = {
    "RATIO": 0,
    "PERIOD": 1,
    "COUNT": 2,
    "SCORE": 3,
    "MONEY": 4,
    "MULTIPLE": 5,
    "GENERAL_NUMBER": 9,
}


def detect_numbers(text: str) -> dict:
    results = []

    for label, pattern in NUMBER_PATTERNS.items():
        for match in re.finditer(pattern, text):
            results.append({
                "label": label,
                "text": match.group(),
                "start": match.start(),
                "end": match.end()
            })

    entities = remove_overlapping_entities(results)

    return {
        "has_number": len(entities) > 0,
        "entities": entities,
        "count": len(entities)
    }


def detect_result_numbers(text: str) -> dict:
    """결과 문장 안에 있는 숫자, 기간, 비율 표현을 탐지한다."""
    sentences = split_sentences(text)
    result_sentences = [
        sentence for sentence in sentences
        if is_result_sentence(sentence)
    ]

    entities = []
    for sentence_index, sentence in enumerate(result_sentences, start=1):
        detected = detect_numbers(sentence)
        for entity in detected["entities"]:
            entity = dict(entity)
            entity["sentence_index"] = sentence_index
            entity["sentence"] = sentence
            entities.append(entity)

    return {
        "has_result_number": len(entities) > 0,
        "result_sentences": result_sentences,
        "entities": entities,
        "count": len(entities),
        "types": sorted({entity["label"] for entity in entities}),
    }


def number_score(text: str) -> int:
    result = detect_numbers(text)

    score = 0

    for ent in result["entities"]:
        if ent["label"] in ["RATIO", "PERIOD", "COUNT", "SCORE", "MONEY", "MULTIPLE"]:
            score += 2
        elif ent["label"] == "GENERAL_NUMBER":
            score += 1

    return score


def result_number_score(text: str) -> int:
    result = detect_result_numbers(text)
    score = 0

    for ent in result["entities"]:
        if ent["label"] in ["RATIO", "PERIOD", "COUNT", "SCORE", "MONEY", "MULTIPLE"]:
            score += 2
        elif ent["label"] == "GENERAL_NUMBER":
            score += 1

    return score


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    protected = re.sub(r"([.!?。])\s+", r"\1\n", text.strip())
    protected = re.sub(r"(?<=다\.)\s+", "\n", protected)
    return [sentence.strip() for sentence in protected.splitlines() if sentence.strip()]


def is_result_sentence(sentence: str) -> bool:
    return (
        any(marker in sentence for marker in RESULT_MARKERS)
        or any(keyword in sentence for keyword in RESULT_CHANGE_KEYWORDS)
    )


def remove_overlapping_entities(entities: list[dict]) -> list[dict]:
    sorted_entities = sorted(
        entities,
        key=lambda item: (
            item["start"],
            LABEL_PRIORITY.get(item["label"], 99),
            -(item["end"] - item["start"]),
        ),
    )
    selected = []

    for entity in sorted_entities:
        if any(ranges_overlap(entity, kept) for kept in selected):
            continue
        selected.append(entity)

    return sorted(selected, key=lambda item: item["start"])


def ranges_overlap(left: dict, right: dict) -> bool:
    return left["start"] < right["end"] and right["start"] < left["end"]
