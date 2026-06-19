import sys
from pathlib import Path


STAR_DIR = Path(__file__).resolve().parents[1]
if str(STAR_DIR) not in sys.path:
    sys.path.insert(0, str(STAR_DIR))

from number_ner import detect_numbers, detect_result_numbers, number_score, result_number_score

test_sentences = [
    "그 결과 처리 시간을 20% 단축했습니다.",
    "3개월 동안 프로젝트를 진행했습니다.",
    "총 5명의 팀원과 12회 회의를 진행했습니다.",
    "정확도를 85점까지 향상시켰습니다.",
    "문제를 해결했습니다."
]

for s in test_sentences:
    print("=" * 50)
    print("문장:", s)
    print("탐지 결과:", detect_numbers(s))
    print("수치 점수:", number_score(s))

print("=" * 50)
test_answer = (
    "프로젝트 초기에는 처리 속도가 느렸습니다. "
    "3개월 동안 병목 구간을 분석하고 쿼리를 개선했습니다. "
    "그 결과 처리 시간을 20% 단축했고 오류 건수를 5건 줄였습니다."
)
print("본문:", test_answer)
print("결과 문장 수치 탐지:", detect_result_numbers(test_answer))
print("결과 문장 수치 점수:", result_number_score(test_answer))
