# job_fit_scorer.py — R03-11 지원 적합성 점수 산출

import json
import os

try:
    from konlpy.tag import Okt
    _okt = Okt()
    KONLPY_AVAILABLE = True
except Exception:
    _okt = None
    KONLPY_AVAILABLE = False

# 직무 키워드 사전 로드
_BASE_DIR = os.path.dirname(__file__)
with open(os.path.join(_BASE_DIR, 'job_keywords.json'), encoding='utf-8') as f:
    JOB_KEYWORDS = json.load(f)

# 모든 직무에 공통으로 나오는 단어 (변별력 없음)
COMMON_WORDS = {
    '노력', '결과', '시간', '문제', '도전', '발전',
    '공부', '준비', '소통', '다른', '관련', '바탕',
    '시작', '이해', '방법', '상황', '직접', '관심',
    '입사', '지원', '계획', '역할', '친구', '학생',
    '생활', '또한', '이후', '분야',
}


def score_job_fit(text: str, job: str) -> dict:
    """
    자소서 텍스트와 직무 키워드 매칭 점수 산출

    Args:
        text: 자소서 본문
        job : 직무명 (예: '백엔드개발자', 'AI/ML엔지니어')
    Returns:
        score   : 0~100 적합성 점수
        matched : 매칭된 키워드 리스트
        missing : 부족한 키워드 리스트
        feedback: 피드백 문자열
    """
    if job not in JOB_KEYWORDS:
        return {
            "score": 0,
            "matched": [],
            "missing": [],
            "feedback": f"'{job}' 직무 정보가 없습니다.",
            "available_jobs": list(JOB_KEYWORDS.keys())[:10]
        }

    # 공통 단어 제거 후 직무 특화 키워드만 사용
    keywords = [k for k in JOB_KEYWORDS[job] if k not in COMMON_WORDS]

    if not keywords:
        return {
            "score": 0,
            "matched": [],
            "missing": [],
            "feedback": "직무 특화 키워드가 없습니다."
        }

    # 자소서에서 명사 추출
    if KONLPY_AVAILABLE:
        nouns = set(_okt.nouns(text))
    else:
        nouns = set(text.split())

    matched = [k for k in keywords if k in nouns]
    missing = [k for k in keywords if k not in nouns]

    score = round(len(matched) / len(keywords) * 100) if keywords else 0

    if score >= 70:
        feedback = f"'{job}' 직무와 높은 적합성을 보입니다."
    elif score >= 40:
        feedback = f"'{job}' 직무 관련 키워드가 일부 있습니다. {missing[:3]} 등을 추가해보세요."
    else:
        feedback = f"'{job}' 직무 관련 키워드가 부족합니다. {missing[:5]} 등을 포함해보세요."

    return {
        "score":   score,
        "matched": matched,
        "missing": missing[:5],
        "feedback": feedback,
    }


def list_jobs() -> list:
    """사용 가능한 직무 목록 반환"""
    return list(JOB_KEYWORDS.keys())