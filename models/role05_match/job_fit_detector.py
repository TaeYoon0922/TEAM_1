"""
지원 적합성(직무/전공/회사 연결성) 탐지기 — ROLE05
data/labeled/기업 직무별 핵심 키워드 및 문장.xlsx 에서 추출한 직무 키워드 사전
(job_keywords.json)을 사용해, 답변이 지원 직무·전공과 얼마나 연결돼 있는지 본다.

analyzer.py 의 score_job_fit() 에서 detect_job_fit() 을 호출한다.
"""
import re
import json
from pathlib import Path

_DIR = Path(__file__).parent
_KEYWORDS_PATH = _DIR / "job_keywords.json"

# 직무 연결을 직접 가리키는 일반 표현(사전과 별개로 항상 신호로 인정)
LINK_WORDS = ["직무", "전공", "학과", "지원분야", "지원 직무", "회사", "당사", "귀사", "직무경험"]

# 사전 노이즈 제거용 — 영어 불용어 / 문장 조각 어미
_EN_STOP = {"and", "for", "is", "in", "of", "to", "the", "my", "have", "a0", "com"}
_TAIL_PAT = re.compile(r"(입니다|습니다|십시오|드립니다|하십시오|였습니다|니다|이며|점입니다|책입니다|있음)$")

_keywords = None  # {"by_group": {...}, "all": set, "groups": {...filtered}}


def _is_noise(token: str) -> bool:
    if len(token) < 2:
        return True
    low = token.lower()
    if low in _EN_STOP:
        return True
    if re.fullmatch(r"[a-z]{1,3}", low):          # 짧은 영문 토큰
        return True
    if re.fullmatch(r"\d+", token):               # 숫자만
        return True
    if _TAIL_PAT.search(token):                   # 문장 조각(서술어)
        return True
    return False


def _load():
    """job_keywords.json 로드 + 노이즈 필터링 (최초 1회)."""
    global _keywords
    if _keywords is not None:
        return _keywords
    try:
        raw = json.loads(_KEYWORDS_PATH.read_text(encoding="utf-8"))
    except Exception:
        _keywords = {"all": set(), "by_group": {}}
        return _keywords

    by_group = {
        g: sorted({k for k in kws if not _is_noise(k)})
        for g, kws in raw.get("by_group", {}).items()
    }
    allkw = {k for k in raw.get("all", []) if not _is_noise(k)}
    _keywords = {"all": allkw, "by_group": by_group}
    return _keywords


def _grade(score: int):
    if score >= 75:
        return "우수", "지원 직무·전공과의 연결이 분명합니다."
    if score >= 50:
        return "보통", "직무 연결 표현이 보이지만 더 구체화할 수 있습니다."
    return "보완 필요", "지원 직무·전공과 연결되는 구체적 표현이 부족합니다."


def detect_job_fit(text: str, question: str = "") -> dict:
    """
    답변이 지원 직무/전공/회사와 얼마나 연결돼 있는지 평가.

    Returns:
        score(0~100), grade, summary, matched_keywords, link_hits, feedback_items
    """
    kw = _load()
    job_terms = kw["all"]

    matched = sorted({t for t in job_terms if t in text})
    link_hits = sorted({w for w in LINK_WORDS if w in text})

    # 점수: 직무 사전 키워드(강한 신호) + 일반 연결어(약한 신호)
    score = min(100, 20 + len(matched) * 12 + len(link_hits) * 8)
    grade, summary = _grade(score)

    feedback_items = []
    if matched:
        preview = ", ".join(matched[:5])
        feedback_items.append(
            f"'{preview}' 등 직무 관련 표현이 보입니다. 지원 직무명·전공 지식과 더 직접 연결하면 설득력이 올라갑니다."
        )
    else:
        feedback_items.append(
            "지원 직무/전공과 연결되는 구체적 표현이 부족합니다. '○○ 직무에 필요한 ○○ 역량/지식'처럼 직무명과 묶어 서술하세요."
        )

    return {
        "role": "ROLE_05",
        "metric": "job_fit",
        "score": score,
        "grade": grade,
        "summary": summary,
        "matched_keywords": matched,
        "link_hits": link_hits,
        "feedback_items": feedback_items,
    }


if __name__ == "__main__":
    kw = _load()
    print(f"직무군 {len(kw['by_group'])}개 / 필터 후 키워드 {len(kw['all'])}개")
    sample = "저는 백엔드 개발 직무에 지원했습니다. 알고리즘과 데이터베이스 설계 프로젝트로 시스템 성능을 개선했습니다."
    r = detect_job_fit(sample, "지원 직무 관련 역량을 기술하시오.")
    print("score:", r["score"], r["grade"])
    print("matched:", r["matched_keywords"])
    print("feedback:", r["feedback_items"][0])
