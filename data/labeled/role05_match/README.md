# ROLE05_MATCH 더미데이터

`role05_match_dummy_300.csv`는 질문-답변 적합도 모델을 빠르게 실험하기 위한 합성 데이터입니다.

원본 `직접 라벨링 200개.xlsx`는 문장 단위 STAR 라벨링 파일이며, `question`, `sentence`, `context_before`, `context_after`, `manual_label`, `S/T/A/R`, 세부 품질 점수를 포함합니다. 이 더미데이터는 원본의 문항 스타일과 자소서 평가 관점을 참고하되, ROLE05_MATCH 목적에 맞게 질문-답변 쌍 단위로 구성했습니다.

원본 확인 결과:

- 최종 라벨링 시트: 200개 문장
- 라벨 분포: GOOD 102개, BAD 98개
- 원본 기준: 지원자 행동, 문항 관련성, 구체성, STAR 요소를 문장 단위로 평가
- ROLE05_MATCH 변환 관점: 문장 자체의 품질보다 질문의 요구 조건을 답변이 충족하는지 평가

## 라벨 구성

- `EXCELLENT`: 질문 핵심 조건을 모두 반영하고 직무/회사 연결, 구체적 행동, 결과가 있는 답변
- `MEDIUM`: 질문 방향은 맞지만 일부 조건이 추상적이거나 결과/직무 연결이 약한 답변
- `OFF_TOPIC`: 문장은 자연스럽지만 질문과 거의 맞지 않는 답변

각 라벨은 100개씩, 총 300개입니다.

문항 유형은 10개이며 각 유형이 30개씩 들어 있습니다.

- `motivation_plan`
- `collaboration`
- `problem_solving`
- `competency`
- `weakness`
- `growth`
- `customer`
- `challenge`
- `ethics`
- `learning`

## 주요 컬럼

- `question_type`: 문항 유형
- `question`: 자소서 질문
- `answer`: 합성 답변
- `manual_label`: 3분류 정답 라벨
- `match_score_0_100`: 질문 적합도 기준 더미 점수
- `question_requirements`: 질문이 요구하는 조건
- `covered_requirements`: 답변이 충족한 조건
- `expected_keywords`: 질문 적합도 모델이 참고할 수 있는 핵심어
- `mismatch_reason`: 라벨 판단 이유

재생성은 `python models/role05_match/generate_dummy_data.py`로 할 수 있습니다.
