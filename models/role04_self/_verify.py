import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from self_detector import detect_self_language
from dependency_parser import parse_sentence, parse_paragraph
from contribution_dict_v1 import has_number_expression

print("=== 2. self_detector 기능 테스트 ===")
cases = [
    ("자기중심 문장", "저는 이 경험을 통해 많은 것을 느꼈습니다."),
    ("기여중심 문장", "팀 성과를 30% 향상시켰습니다."),
    ("혼합형 문장",   "저는 열심히 노력한 결과 팀 성과를 20% 개선했습니다."),
]
for label, text in cases:
    r = detect_self_language(text)
    print(f"  [{label}]")
    print(f"    자기중심 점수: {r['self_score']}  기여중심 점수: {r['contribution_score']}  등급: {r['grade']}")
    print(f"    요약: {r['summary']}")
    print(f"    피드백 수: {len(r['feedback_items'])}개")

print()
print("=== 3. 수치 표현 탐지 테스트 ===")
num_cases = [
    "처리 시간을 30% 단축했습니다.",
    "5명의 팀원과 협업했습니다.",
    "2배 향상, 500만 원 절감, 상위 10% 달성",
    "열심히 노력했습니다.",  # 수치 없음
]
for text in num_cases:
    hits = has_number_expression(text)
    print(f"  {text}")
    print(f"    탐지된 수치: {hits if hits else '없음'}")

print()
print("=== 4. 의존 파싱 테스트 ===")
parse_cases = [
    ("자기중심", "저는 이 프로젝트에서 많은 것을 배웠습니다."),
    ("기여중심", "시스템 응답 속도를 2배 향상시켰습니다."),
    ("외부주어", "시스템이 안정적으로 배포되었습니다."),
]
for label, text in parse_cases:
    r = parse_sentence(text)
    print(f"  [{label}] {text}")
    print(f"    주어: {r['subject']} ({r['subject_type']})  술어: {r['predicate']} ({r['verb_type']})")
    print(f"    판정: {r['sentence_type']}")

print()
print("=== 5. 엣지 케이스 테스트 ===")
edge_cases = [
    ("빈 문장",     ""),
    ("단어만",      "노력"),
    ("숫자만",      "30%"),
    ("긴 문단",     "저는 3년간 개발 경험을 쌓았습니다. 그 결과 API 성능을 40% 개선했습니다. 팀원들과 협업하며 느꼈습니다."),
]
for label, text in edge_cases:
    try:
        r = detect_self_language(text)
        print(f"  [{label}] -> 자기: {r['self_score']} / 기여: {r['contribution_score']} / 등급: {r['grade']}  OK")
    except Exception as e:
        print(f"  [{label}] -> FAIL: {e}")

print()
print("모든 검증 완료")
