import sys
from pathlib import Path


ROLE02_DIR = Path(__file__).resolve().parents[1]
if str(ROLE02_DIR) not in sys.path:
    sys.path.insert(0, str(ROLE02_DIR))

from head_first_rules import analyze_head_first

test_texts = [
    "저는 데이터 분석을 통해 문제를 해결한 경험이 있습니다. 프로젝트 당시 일정 지연 문제가 있었습니다.",
    "대학교 2학년 때 프로젝트를 진행했습니다. 저는 데이터를 분석해 문제를 해결했습니다.",
    "저는 협업 과정에서 갈등을 조율하며 성과를 만든 경험이 있습니다. 팀원 간 의견 차이가 있었습니다.",
    "처음에는 역할 분담이 명확하지 않았습니다. 이후 공유 문서를 만들어 문제를 개선했습니다."
]

for text in test_texts:
    print("=" * 50)
    print("본문:", text)
    print("분석 결과:", analyze_head_first(text))
    
