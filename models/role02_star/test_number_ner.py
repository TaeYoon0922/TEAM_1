from number_ner import detect_numbers, number_score

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