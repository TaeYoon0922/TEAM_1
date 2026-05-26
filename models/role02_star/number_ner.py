import re

NUMBER_PATTERNS = {
    "PERCENT": r"\d+(\.\d+)?\s*(%|퍼센트|프로)",
    "PERIOD": r"\d+(\.\d+)?\s*(년|개월|달|주|일|시간|분|초)",
    "COUNT": r"\d+(\.\d+)?\s*(명|개|건|회|번|차례|문항|페이지)",
    "SCORE": r"\d+(\.\d+)?\s*(점|등급|위|위권)",
    "MONEY": r"\d+(\.\d+)?\s*(원|만원|억원|천원)",
    "MULTIPLE": r"\d+(\.\d+)?\s*(배)",
    "GENERAL_NUMBER": r"\d+(\.\d+)?"
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

    # GENERAL_NUMBER가 다른 패턴과 중복될 수 있어서 중복 제거
    unique = []
    seen = set()

    for item in results:
        key = (item["text"], item["start"], item["end"])
        if key not in seen:
            seen.add(key)
            unique.append(item)

    return {
        "has_number": len(unique) > 0,
        "entities": unique,
        "count": len(unique)
    }


def number_score(text: str) -> int:
    result = detect_numbers(text)

    score = 0

    for ent in result["entities"]:
        if ent["label"] in ["PERCENT", "PERIOD", "COUNT", "SCORE", "MONEY", "MULTIPLE"]:
            score += 2
        elif ent["label"] == "GENERAL_NUMBER":
            score += 1

    return score