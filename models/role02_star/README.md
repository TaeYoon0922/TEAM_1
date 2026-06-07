# ROLE02 핵심 주장 명확성 & 경험 구체성 분석 모델

## 1. 목적
자소서 답변 문장을 입력받아 핵심 주장 명확성과 경험 구체성을 분석한다.

- 핵심 주장 명확성: 첫 문장이 핵심 역량, 성과, 문제 해결 경험을 먼저 제시하는지 판단
- 경험 구체성: S/T/A/R 요소가 답변에 드러나는지 체크리스트로 판단

## 2. 경험 구체성 판단 방식

`data/labeled/jobkorea_star_full_labeled_sentence_dataset.csv`의 문장 단위 STAR 라벨을 사용해 multi-label 문장 분류기를 학습한다.

| 요소 | 의미 | 정답 라벨 | 점수 |
|---|---|---|---|
| S | Situation | 문제 상황, 배경, 맥락 설명 문장 | 25 |
| T | Task | 역할, 목표, 책임 설명 문장 | 25 |
| A | Action | 실제 수행 행동 설명 문장 | 25 |
| R | Result | 결과, 성과, 변화 설명 문장 | 25 |

라벨값은 `S`, `T`, `A`, `R`, `X`뿐 아니라 `T+A`, `S+T+A+R`처럼 복수 라벨도 포함하므로, 단일 클래스가 아니라 S/T/A/R 각각의 이진 분류로 평가한다. 답변 단위 점수는 문장별 예측 결과를 모아 하나라도 검출된 요소를 충족으로 본다.

## 3. 핵심 주장 명확성 탐지 규칙

첫 문장이 핵심 역량, 성과, 문제 해결 경험을 요약하면 핵심 주장 명확성 충족으로 판단한다.

- 긍정 신호: 역량, 경험, 강점, 성과, 해결, 개선, 기여, 자신 있습니다, 갖추고 있습니다
- 약한 시작 신호: 어렸을 때, 대학교, 처음에는, 당시, 예전에, 고등학교
- 출력: 충족 시 `✅`, 미충족 시 `❌`

내부 구현에서는 기존 두괄식 개념을 유지하되, 사용자 노출 지표명은 `핵심 주장 명확성`으로 사용한다.

## 4. 구현 파일

- `head_first_rules.py`: 핵심 주장 명확성 탐지
- `head_first_classifier.py`: 핵심 주장 명확성 이진 분류
- `number_ner.py`: 숫자, 기간, 비율 및 결과 문장 수치 탐지
- `star_scorer.py`: 경험 구체성 S/T/A/R 체크리스트 산출
- `star_label_model.py`: 라벨 CSV 기반 STAR 문장 multi-label 모델 학습/예측
- `role04_output.py`: ROLE04 전달용 출력 포맷 생성
- `ROLE04_OUTPUT_SPEC.md`: ROLE04 통합 스코어 연동 명세
- `run_role02_demo.py`: `data/processed/jobkorea_train.csv` 기반 ROLE02 분석 데모
- `run_role02_evaluation.py`: STAR 라벨 CSV 기준 Precision / Recall / F1 평가 및 시각화
- `visualize.py`: STAR radar, ROLE02 요약, 모델 충족 여부 시각화
- `tests/test_head_first.py`: 핵심 주장 명확성 더미 테스트
- `tests/test_head_first_classifier.py`: 두괄식 이진 분류 더미 테스트
- `tests/test_number_ner.py`: 수치 탐지 더미 테스트
- `tests/test_star_scorer.py`: 경험 구체성 스코어링 더미 테스트
- `notebooks/role02_star/role02_star_pattern_test.ipynb`: train 데이터 기반 ROLE02 모듈 import 및 간단 검증

## 5. 평가 실행

```bash
python3 models/role02_star/run_role02_evaluation.py
```

평가 기준:

- 데이터: `data/labeled/jobkorea_star_full_labeled_sentence_dataset.csv`
- 분리 방식: URL 기준 train/test split
- 경험 구체성: `star_label`을 S/T/A/R multi-label 정답으로 변환해 요소별 이진 분류 평가
- 핵심 주장 명확성: 현재 라벨 CSV에는 별도 정답 컬럼이 없어 평가 대상에서 제외하고 demo 출력만 유지

산출물:

- `models/role02_star/artifacts/star_sentence_classifier.joblib`: 학습된 STAR 문장 분류기
- `data/processed/role02_eval_predictions.csv`: 문장별 정답/예측 결과
- `data/processed/role02_eval_metrics.csv`: Precision / Recall / F1 / Accuracy
- `models/role02_star/figures/role02_precision_recall_f1.png`: 경험 구체성 S/T/A/R 지표 막대그래프
- `models/role02_star/figures/role02_confusion_matrices.png`: confusion matrix

현재 모델은 라벨링된 문장 전체를 학습 데이터로 사용하며, evaluation에서는 test URL을 분리해 URL overlap 없이 성능을 측정한다.

## 6. 데모 시각화

```bash
python3 models/role02_star/run_role02_demo.py
```

산출물:

- `data/processed/role02_train_scores.csv`: train 답변 1000개 단위 batch scoring 결과
- `models/role02_star/figures/role02_demo_star_radar.png`: STAR 충족도 radar chart
- `models/role02_star/figures/role02_demo_summary.png`: 핵심 주장 명확성/경험 구체성 요약
- `models/role02_star/figures/role02_demo_requirement_status.png`: 핵심 주장 명확성 및 S/T/A/R 체크리스트 충족 여부

노트북 `notebooks/role02_star/role02_star_pattern_test.ipynb`도 동일한 시각화를 생성하고 셀 출력에서 바로 표시한다.

## 7. 다음 구현

ROLE04 전달용 출력 포맷은 `ROLE04_OUTPUT_SPEC.md`와 `role04_output.py`에 정의되어 있다.

```bash
python3 models/role02_star/role04_output.py
```

통합 기준:

- `role04_metrics["핵심 주장 명확성"]`: `app/analyzer.py`의 `MetricResult` 호환
- `role04_metrics["경험 구체성"]`: `app/analyzer.py`의 `MetricResult` 호환
- `role02_summary.score`: `0.4 * 핵심 주장 명확성 + 0.6 * 경험 구체성`
