import sys
sys.path.insert(0, 'models/role04_self')
from KoSimCSE import analyze_essay

text = """개발자가 되고 싶다고 결심한 뒤, 제가 결정한 저의 미래의 모습입니다.
저는 어떤 목표를 이루기 위해서는 꿈을 잃지 않는 것과, 그 꿈을 현실로 만들기 위해 노력하는 것이 중요하다 생각합니다.
제가 이곳 아이티센에 지원한 이유도 마찬가지로, 여러 방식을 통해 발전하기를 추구하는 이 회사와 함께 더 꿈을 꾸며 도전하고 싶습니다.
저는 java programming 쪽을 지원하였는데 개발자가 되고 싶다고 결정한 뒤 가장 먼저 시도하고, 현재도 가장 힘을 쏟고 있는 언어가 java이기 때문입니다.
java를 중심으로 공부하기로 결정한 뒤, 저는 매해 발전을 위해 다양한 프로젝트와 알고리즘 공부를 병행하고 있습니다."""

result = analyze_essay(text)
print(result['overall_feedback'])
print()
for r in result['paragraph_results']:
    print(f'[문단 {r["paragraph_idx"]+1}] {r["paragraph_preview"]}')
    print(r['feedback'])
    print()
print('=' * 60)
print('핵심 요약 (중심문장 모음)')
for i, s in enumerate(result['center_sentences'], 1):
    print(f'  {i}. {s}')
