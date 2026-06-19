import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from dependency_parser import parse_sentence, parse_paragraph


DUMMY_SENTENCES = {
    "자기중심_감정": "저는 이 프로젝트를 통해 많은 것을 느꼈습니다.",
    "자기중심_인식": "저는 협업의 중요성을 깨달았습니다.",
    "자기중심_성장": "저는 이 경험을 통해 성장할 수 있었습니다.",
    "기여중심_수치": "팀 프로젝트에서 API 응답 속도를 30% 단축했습니다.",
    "기여중심_결과": "고객 만족도를 4.2점에서 4.8점으로 향상시켰습니다.",
    "기여중심_외부": "시스템이 안정적으로 배포되었습니다.",
    "혼합형":        "저는 열심히 노력한 결과 팀 성과를 20% 개선했습니다.",
}

DUMMY_PARAGRAPH = """
저는 대학교 3학년 때 처음으로 팀 프로젝트를 경험했습니다.
처음에는 협업이 어렵다고 느꼈습니다.
하지만 매주 회의를 3회 진행하며 소통 방식을 개선했습니다.
그 결과 프로젝트 완성도를 전년 대비 40% 향상시켰습니다.
팀원들의 만족도도 크게 높아졌습니다.
"""


def print_separator(title: str):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print('='*50)


def test_single_sentences():
    print_separator("단일 문장 의존 파싱 테스트")

    for label, sentence in DUMMY_SENTENCES.items():
        result = parse_sentence(sentence)
        print(f"\n[{label}]")
        print(f"  문장       : {result['text']}")
        print(f"  주어       : {result['subject']} ({result['subject_type']})")
        print(f"  술어       : {result['predicate']} ({result['verb_type']})")
        print(f"  문장 유형  : {result['sentence_type']}")
        print(f"  의존 관계  :", end=" ")
        for dep in result['dep_pairs']:
            print(f"{dep['token']}({dep['dep']})", end=" ")
        print()


def test_paragraph():
    print_separator("문단 전체 분석 테스트")

    result = parse_paragraph(DUMMY_PARAGRAPH.strip())

    print(f"\n  총 문장 수       : {result['total_count']}")
    print(f"  자기중심 문장    : {result['self_count']}개 ({result['self_ratio']*100:.0f}%)")
    print(f"  기여중심 문장    : {result['contribution_count']}개 ({result['contribution_ratio']*100:.0f}%)")
    print(f"  중립 문장        : {result['neutral_count']}개")

    print("\n  [문장별 분류]")
    for i, s in enumerate(result['sentences'], 1):
        icon = {"self_centered": "🔴", "contribution": "🟢", "neutral": "⚪"}.get(s['sentence_type'], "⚪")
        print(f"  {i}. {icon} [{s['sentence_type']:15s}] {s['text']}")


if __name__ == "__main__":
    test_single_sentences()
    test_paragraph()
    print("\n Phase 1 의존 파싱 파이프라인 테스트 완료")
