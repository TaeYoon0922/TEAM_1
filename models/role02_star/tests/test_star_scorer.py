import sys
from pathlib import Path


ROLE02_DIR = Path(__file__).resolve().parents[1]
if str(ROLE02_DIR) not in sys.path:
    sys.path.insert(0, str(ROLE02_DIR))

from star_scorer import score_star_completeness


test_texts = [
    "저는 꾸준히 노력하는 사람입니다.",
    "프로젝트 초기 상황을 파악했습니다.",
    "문제를 해결했습니다.",
    (
        "프로젝트 당시 일정 지연 문제가 있었습니다. "
        "저는 병목 구간을 분석했습니다."
    ),
    (
        "프로젝트 당시 일정 지연 문제가 있었습니다. "
        "제 역할은 병목 구간을 분석하고 개선안을 제안하는 것이었습니다. "
        "그 결과 처리 시간을 20% 단축했습니다."
    ),
    (
        "당시 고객 문의가 급증해 응답 지연 문제가 발생했습니다. "
        "저는 담당자로서 응답 시간을 줄이는 목표를 맡았습니다. "
        "문의 유형을 분석하고 답변 템플릿을 설계해 팀에 도입했습니다. "
        "그 결과 평균 응답 시간을 30% 단축했습니다."
    ),
]


for text in test_texts:
    result = score_star_completeness(text)
    print("=" * 50)
    print("본문:", text)
    print("경험 구체성:", result["score"], result["grade"])
    print("체크리스트:", result["display"])
    print("피드백:", result["feedback"])
    print("근거:", result["evidence"])
