# MATCH_MATCH 질문-답변 적합도 모델

자소서 답변이 질문의 의도에 맞는지 평가하는 규칙 기반 베이스라인입니다.

기존 두괄식, STAR, 모호도, 자기중심 지표는 답변 내부 품질을 봅니다. 이 모델은 질문과 답변을 함께 받아서 답변이 엉뚱한 방향으로 흘러가는 문제를 잡습니다.

질문 군집화 학습/캐시는 `data/processed/jobkorea_train.csv` 기준으로 생성합니다. `data/raw` 원문 전체나 test split을 학습 캐시에 섞지 않습니다.

## 사용 예시

```python
from models.match.relevance_detector import detect_answer_relevance

result = detect_answer_relevance(
    "지원동기와 입사 후 기여 계획을 기술해 주세요.",
    "저는 데이터 분석 프로젝트 경험을 바탕으로 고객 이탈률을 낮추는 분석 체계를 만들고 싶습니다.",
)

print(result["score"], result["grade"])
```

## 출력

- `score`: 0~100 질문 적합도 점수
- `grade`: 우수, 보통, 주의, 위험
- `question_intents`: 감지된 질문 의도
- `matched_keywords`: 답변에서 확인된 질문 핵심어
- `missing_keywords`: 답변에서 약한 질문 핵심어
- `subquestion_coverage`: 복합 문항 조건 충족률
- `feedback_items`: 개선 피드백

## 현재 방식

1. 질문 핵심어 추출
2. 지원동기, 경험, 역량, 단점, 협업, 성장과정, 입사 후 계획 등의 질문 의도 탐지
3. 답변 내 핵심어/의도 반영도 계산
4. 복합 문항의 하위 조건 반영률 계산
5. 질문 핵심어가 거의 없거나 일반적인 좋은 말만 많은 답변에 패널티 적용

라벨 데이터가 쌓이면 같은 `detect_answer_relevance(question, answer)` 인터페이스를 유지한 채 임베딩/분류 모델로 교체할 수 있습니다.
