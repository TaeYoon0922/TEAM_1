STAR_PATTERNS = {
    "S": {
        "name": "Situation",
        "description": "상황/배경",
        "keywords": ["상황", "당시", "문제", "어려움", "과제", "프로젝트", "현장", "배경", "초기"],
        "weight": 25,
    },
    "T": {
        "name": "Task",
        "description": "목표/역할/책임",
        "keywords": ["목표", "역할", "담당", "책임", "해야", "필요", "요구", "맡", "기한"],
        "weight": 25,
    },
    "A": {
        "name": "Action",
        "description": "행동/전략",
        "keywords": ["분석", "제안", "실행", "설계", "개선", "협업", "조정", "도입", "시도", "진행"],
        "weight": 25,
    },
    "R": {
        "name": "Result",
        "description": "결과/성과",
        "keywords": ["결과", "성과", "달성", "증가", "감소", "완료", "해결", "단축", "향상", "줄였"],
        "weight": 25,
    },
}


def score_star_completeness(text: str) -> dict:

    normalized = normalize_text(text)
    checklist = {}
    evidence = {}

    for code, pattern in STAR_PATTERNS.items():
        hits = find_keyword_hits(normalized, pattern["keywords"])
        checklist[code] = len(hits) > 0
        evidence[code] = hits

    component_scores = {
        code: pattern["weight"] if checklist[code] else 0
        for code, pattern in STAR_PATTERNS.items()
    }
    score = sum(component_scores.values())
    missing = [code for code, exists in checklist.items() if not exists]

    return {
        "metric": "경험 구체성",
        "score": score,
        "grade": grade_from_score(score),
        "checklist": checklist,
        "display": format_checklist(checklist),
        "missing": missing,
        "evidence": evidence,
        "component_scores": component_scores,
        "max_score": 100,
        "scoring_note": "경험 구체성은 S/T/A/R 체크리스트 충족 여부로 평가합니다.",
        "feedback": build_feedback(checklist, missing),
    }


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def find_keyword_hits(text: str, keywords: list[str]) -> list[str]:
    return [keyword for keyword in keywords if keyword in text]


def format_checklist(checklist: dict) -> str:
    return " ".join(
        f"{code}{'✅' if checklist.get(code) else '❌'}"
        for code in ["S", "T", "A", "R"]
    )


def grade_from_score(score: int) -> str:
    if score >= 100:
        return "우수"
    if score >= 75:
        return "양호"
    if score >= 50:
        return "보통"
    return "보완 필요"


def build_feedback(checklist: dict, missing: list[str]) -> str:
    display = format_checklist(checklist)
    if not missing:
        return f"{display} S/T/A/R 요소가 모두 확인됩니다."
    missing_text = ", ".join(missing)
    return f"{display} {missing_text} 요소가 약합니다. 빠진 요소를 한두 문장으로 보강하세요."
