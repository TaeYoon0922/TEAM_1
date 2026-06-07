import sys
from pathlib import Path


ROLE02_DIR = Path(__file__).resolve().parents[1]
if str(ROLE02_DIR) not in sys.path:
    sys.path.insert(0, str(ROLE02_DIR))

from head_first_classifier import classify_head_first


test_cases = [
    (
        "저는 데이터 분석 역량으로 처리 지연 문제를 해결한 경험이 있습니다. 당시 프로젝트 일정이 늦어졌습니다.",
        1,
    ),
    (
        "프로젝트 당시 일정 지연 문제가 있었습니다. 저는 데이터 분석으로 원인을 찾았습니다.",
        0,
    ),
    (
        "대학교 2학년 때 프로젝트를 진행했습니다. 저는 문제를 해결하기 위해 분석했습니다.",
        0,
    ),
    (
        "고객 응답 시간을 30% 단축한 경험이 있습니다. 문의 유형을 분석하고 템플릿을 도입했습니다.",
        1,
    ),
]


for text, expected in test_cases:
    result = classify_head_first(text)
    print("=" * 50)
    print("본문:", text)
    print("예상:", expected)
    print("분류 결과:", result)
    assert result["is_head_first"] == expected
