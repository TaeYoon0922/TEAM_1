# TEAM_1

자소서 질문과 답변을 함께 입력하면, 먼저 답변이 문항에 맞는지 확인한 뒤 질문 유형에 맞는 지표만 골라 개선 피드백을 제공하는 텍스트 마이닝 시스템입니다.

최종 화면은 점수보다 "어디를 어떻게 고칠지"에 집중합니다.

## ROLE 05 UI 실행

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

> 최초 실행 시 SBERT(`ko-sroberta-multitask`)와 KoSimCSE 모델을 자동 다운로드합니다(수백 MB, 인터넷 필요). 이후에는 캐시되어 빠릅니다.

## 선택 의존성 (전체 기능 활성화)

기본 설치만으로도 앱은 동작합니다. 아래는 *일부 지표를 더 정밀하게* 쓰기 위한 선택 사항이며, **미설치 시 자동으로 더 가벼운 방식으로 폴백**하므로 앱은 멈추지 않습니다.

| 기능 | 추가 설치 | 미설치 시 |
|---|---|---|
| 표현 명료성 형태소 정밀 모드 (`hedge_detector`의 Okt) | **Java JDK 17+** 설치 후 `JAVA_HOME` 설정 (konlpy/JPype1은 requirements에 포함) | 형태소 없이 규칙 기반 폴백 |
| 자기표현 차별성 의존구문 블렌딩 (`dependency_parser`) | `python -m spacy download ko_core_news_sm` | 사전 점수만 사용(의존구문 가중 제외) |

```bash
# 자바(예: winget) — 표현 명료성 형태소 모드용
winget install --id Microsoft.OpenJDK.17
# 설치 후 JAVA_HOME 설정 필요 (예: setx JAVA_HOME "C:\Program Files\Microsoft\jdk-17")

# spacy 한국어 모델 — 자기표현 차별성 의존구문용
python -m spacy download ko_core_news_sm
```

> 참고: `지원 적합성`이 쓰던 role03 `job_fit_scorer`도 konlpy(Okt)를 사용하지만, 현재 지표에서 제외돼 있어 앱 실행에는 영향이 없습니다.

## 분석 흐름

```text
[질문 + 자소서 답변]
        |
        v
1. 문항 적합성 게이트
   - 질문과 동떨어진 답변이면 세부 분석을 중단하고 질문에 맞게 고치도록 안내

2. 질문 유형 분류
   - SBERT + K-Means 기반으로 질문을 6개 유형 중 하나로 분류

3. 군집별 지표 선택
   - cluster_map.json에 정의된 지표만 분석

4. 피드백 출력
   - 지표별 피드백, 강점, 우선 개선 포인트, 문장별 코멘트 제공
```

> 문항 적합성 게이트 임계값(`GATE_THRESHOLD = 20`)은 현재 합성 라벨 데이터로는 신뢰할 보정이 어려워, 명백한 동문서답만 거르도록 **보수적으로** 설정했습니다. 사람이 라벨링한 실제 동문서답 데이터가 쌓이면 `scripts/calibrate_relevance_gate.py`로 재보정할 수 있습니다.

## 현재 노출 지표

- 문항 적합성: 답변이 질문의 핵심 의도와 조건에 맞는지 확인
- 핵심 주장 명확성: 첫 문장이나 앞부분에 핵심 메시지가 드러나는지 확인
- 경험 구체성: 상황, 과제, 행동, 결과(STAR) 흐름이 있는지 확인
- 표현 명료성: 모호어, 추상어, 상투어가 적고 구체적으로 표현됐는지 확인
- 자기표현 차별성: 나 중심 서술보다 결과, 기여, 외부 영향이 드러나는지 확인
- 문장 완성도: 문장 길이, 반복, 가독성 관점에서 읽기 좋은지 확인

## 지원 적합성 제외 결정

`지원 적합성`은 최종 노출 지표에서 제외합니다.

이유:

- 현재 직무 선택 드롭다운을 제거해 사용자가 직무를 별도로 고르지 않습니다.
- 직무 키워드 사전 기반 평가는 직무 데이터 품질에 따라 편차가 커 최종 피드백의 신뢰도를 낮출 수 있습니다.
- 질문 자체에 직무 관련 요구가 있으면 `문항 적합성`, `핵심 주장 명확성`, `경험 구체성` 피드백에서 간접적으로 보완할 수 있습니다.

단, 관련 코드는 삭제하지 않고 보존합니다. 추후 직무 데이터와 UX가 정리되면 다시 연결할 수 있도록 `app/analyzer.py`의 `score_job_fit()` 및 role03 직무 관련 모듈은 유지합니다.

## 주요 파일

- `app/streamlit_app.py`: Streamlit UI
- `app/analyzer.py`: 전체 분석 파이프라인 통합
- `models/role05_match/relevance_detector.py`: 질문-답변 문항 적합성 분석
- `models/role05_match/question_clusterer.py`: 질문 군집화
- `models/role05_match/cluster_map.json`: 질문 유형별 적용 지표 설정
- `models/role02_star/`: 두괄식, 수치, STAR 관련 규칙
- `models/role03_hedge/`: 모호 표현 및 추상 표현 탐지
- `models/role04_self/`: 자기중심/기여중심 표현 및 구조 분석

## STAR 분류법

200개 문장을 두 명이 분류한 뒤, 불일치 항목은 상의 후 재결정했습니다.

| 라벨 | 성능 |
|---|---|
| S | 0.808 매우 높음 |
| T | 0.660 양호 |
| A | 0.667 양호 |
| R | 0.728 양호 |
| X | 0.792 양호 |
| Macro 평균 | 0.731 양호 |
