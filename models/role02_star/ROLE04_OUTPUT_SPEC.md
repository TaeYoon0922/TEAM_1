# ROLE02 -> ROLE04 Output Spec

## 목적

ROLE02 모델의 출력 형식을 고정해 ROLE04 또는 통합 analyzer가 같은 기준으로 점수를 받을 수 있게 한다.

ROLE02가 전달하는 지표는 두 개다.

| ROLE02 지표 | 의미 | 방식 |
|---|---|---|
| 핵심 주장 명확성 | 첫 문장에 핵심 메시지가 명확한지 | 규칙 기반 점수 + 이진 표시 |
| 경험 구체성 | S/T/A/R 경험 요소가 드러나는지 | STAR 문장 라벨 기반 multi-label 모델 |

## 생성 함수

```python
from models.role02_star.role04_output import build_role04_payload

payload = build_role04_payload(
    answer=answer_text,
    question=question_text,
    metadata={
        "row_no": 1,
        "url": "...",
        "company": "...",
        "season": "...",
        "question_no": "1",
    },
)
```

## 최상위 구조

```json
{
  "schema_version": "role02_to_role04.v1",
  "source_role": "ROLE02",
  "target_role": "ROLE04",
  "question": "...",
  "answer_id": "1",
  "metadata": {},
  "role02_summary": {},
  "role04_metrics": {},
  "raw_outputs": {}
}
```

## ROLE04 연동 필드

`role04_metrics`는 `app/analyzer.py`의 `MetricResult` 구조와 맞춘다.

```json
{
  "role04_metrics": {
    "핵심 주장 명확성": {
      "label": "핵심 주장 명확성",
      "score": 0,
      "level": "보완 필요",
      "feedback": "첫 문장에 핵심 역량, 성과, 문제 해결 경험을 더 직접적으로 제시하세요.",
      "applicable": true,
      "details": {}
    },
    "경험 구체성": {
      "label": "경험 구체성",
      "score": 75,
      "level": "보통",
      "feedback": "S✅ T✅ A✅ R❌ R 요소가 약합니다.",
      "applicable": true,
      "details": {}
    }
  }
}
```

## 통합 스코어 기준

ROLE02 내부 통합 점수는 ROLE04가 보조 feature로 사용할 수 있도록 `role02_summary.score`에 제공한다.

```text
ROLE02 통합 점수 = 0.4 * 핵심 주장 명확성 + 0.6 * 경험 구체성
```

가중치 기준:

| 지표 | 가중치 | 이유 |
|---|---:|---|
| 핵심 주장 명확성 | 0.4 | 규칙 기반 이진 판단이므로 보조 지표 |
| 경험 구체성 | 0.6 | 라벨 데이터로 학습/평가된 STAR 지표 |

ROLE04가 자체 통합 스코어를 만들 때는 `role04_metrics`의 개별 점수를 우선 사용하고, 단일 ROLE02 대표값이 필요할 때만 `role02_summary.score`를 사용한다.

## 상세 필드

### 핵심 주장 명확성 details

| 필드 | 타입 | 설명 |
|---|---|---|
| display | string | `✅` 또는 `❌` |
| is_clear | boolean | 핵심 주장 명확성 충족 여부 |
| first_sentence | string | 판단 대상 첫 문장 |
| matched_keywords | list[string] | 첫 문장에서 탐지된 핵심 주장 키워드 |
| bad_start_keywords | list[string] | 약한 시작 표현 |
| decision_boundary | number | ✅ 판정 기준. 현재 50점 |
| model_type | string | `rule_based_binary` |

### 경험 구체성 details

| 필드 | 타입 | 설명 |
|---|---|---|
| display | string | `S✅ T✅ A✅ R❌` 형식 |
| checklist | object | `S/T/A/R` boolean |
| component_scores | object | 각 요소별 25점 또는 0점 |
| missing | list[string] | 빠진 STAR 요소 |
| evidence | object | 요소별 근거 문장 |
| decision_threshold | number | STAR 문장 예측 threshold. 현재 0.92 |
| model_type | string | `trained_multilabel_sentence_classifier` |
| scoring_note | string | 학습 모델 또는 fallback 설명 |

## 산출 예시 확인

```bash
python3 models/role02_star/role04_output.py
```

첫 번째 train 답변을 사용해 ROLE04 전달용 JSON 예시를 출력한다.
