# job_fit_scorer.py — R03-11 지원 적합성 점수 산출 (개선판)

import json
import os

try:
    from konlpy.tag import Okt
    _okt = Okt()
    KONLPY_AVAILABLE = True
except Exception:
    _okt = None
    KONLPY_AVAILABLE = False

_BASE_DIR = os.path.dirname(__file__)
with open(os.path.join(_BASE_DIR, 'job_keywords.json'), encoding='utf-8') as f:
    JOB_KEYWORDS = json.load(f)

# 모든 직무 공통 단어 (변별력 없음)
COMMON_WORDS = {
    '노력', '결과', '시간', '문제', '도전', '발전',
    '공부', '준비', '소통', '다른', '관련', '바탕',
    '시작', '이해', '방법', '상황', '직접', '관심',
    '입사', '지원', '계획', '역할', '친구', '학생',
    '생활', '또한', '이후', '분야',
}

# 최소 키워드 수 기준 (이하면 신뢰도 낮음)
MIN_KEYWORDS = 5


def score_job_fit(text: str, job: str) -> dict:
    if job not in JOB_KEYWORDS:
        return {
            "score": 0,
            "matched": [],
            "missing": [],
            "feedback": f"'{job}' 직무 정보가 없습니다.",
            "reliable": False,
            "available_jobs": list(JOB_KEYWORDS.keys())[:10]
        }

    keywords = [k for k in JOB_KEYWORDS[job] if k not in COMMON_WORDS]

    # 키워드 부족 → 신뢰도 낮음
    if len(keywords) < MIN_KEYWORDS:
        return {
            "score": None,
            "matched": [],
            "missing": [],
            "feedback": f"'{job}' 직무는 특화 키워드가 부족해 적합성 점수를 산출할 수 없습니다.",
            "reliable": False,
        }

    if KONLPY_AVAILABLE:
        nouns = set(_okt.nouns(text))
    else:
        nouns = set(text.split())

    matched = [k for k in keywords if k in nouns]
    missing = [k for k in keywords if k not in nouns]

    score = round(len(matched) / len(keywords) * 100)

    # 100점 방지 — 키워드 3개 이하 매칭은 최대 85점 제한
    if score == 100 and len(keywords) <= 7:
        score = round(85 + (len(matched) / len(keywords)) * 10)
        score = min(score, 92)

    if score >= 70:
        feedback = f"'{job}' 직무와 높은 적합성을 보입니다."
    elif score >= 40:
        feedback = f"'{job}' 직무 관련 키워드가 일부 있습니다. {missing[:3]} 등을 추가해보세요."
    else:
        feedback = f"'{job}' 직무 관련 키워드가 부족합니다. {missing[:5]} 등을 포함해보세요."

    return {
        "score":    score,
        "matched":  matched,
        "missing":  missing[:5],
        "feedback": feedback,
        "reliable": True,
    }


def list_jobs() -> list:
    return list(JOB_KEYWORDS.keys())