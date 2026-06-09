# TEAM_1

자소서 질문과 답변을 함께 입력하면, 먼저 답변이 문항에 맞는지 확인한 뒤 질문 유형에 맞는 지표만 골라 개선 피드백을 제공하는 텍스트 마이닝 시스템입니다.

점수보다 "어디를 어떻게 고칠지"에 집중합니다.

## 실행

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

> 최초 실행 시 SBERT(`ko-sroberta-multitask`)와 KoSimCSE 모델을 자동 다운로드합니다(수백 MB, 인터넷 필요). 이후에는 캐시되어 빠릅니다.

## 선택 의존성

기본 설치만으로도 앱은 동작합니다. 미설치 시 자동으로 더 가벼운 방식으로 폴백합니다.

| 기능 | 추가 설치 |
|---|---|
| 표현 명료성 형태소 정밀 모드 | **Java JDK 17+** 설치 후 `JAVA_HOME` 설정 |
| 자기표현 차별성 의존구문 분석 | `python -m spacy download ko_core_news_sm` |

## 분석 흐름

```
[질문 + 자소서 답변]
        |
        v
1. 문항 적합성 게이트
   - 질문과 동떨어진 답변이면 세부 분석을 중단하고 안내

2. 질문 유형 분류
   - SBERT + K-Means 기반으로 질문을 6개 유형 중 하나로 분류

3. 군집별 지표 선택
   - cluster_map.json에 정의된 지표만 분석

4. 피드백 출력
   - 지표별 피드백, 강점, 우선 개선 포인트, 문장별 코멘트 제공
```

## 분석 지표

| 지표 | 설명 |
|---|---|
| 문항 적합성 | 답변이 질문의 핵심 의도와 조건에 맞는지 |
| 핵심 주장 명확성 | 첫 문장이나 앞부분에 핵심 메시지가 드러나는지 |
| 경험 구체성 | 상황, 과제, 행동, 결과(STAR) 흐름이 있는지 |
| 표현 명료성 | 모호어·추상어·상투어 없이 구체적으로 표현됐는지 |
| 자기표현 차별성 | 나 중심 서술보다 결과·기여·외부 영향이 드러나는지 |
| 문장 완성도 | 문장 길이·반복·가독성 관점에서 읽기 좋은지 |

## 주요 파일

| 파일 | 역할 |
|---|---|
| `app/streamlit_app.py` | Streamlit UI |
| `app/analyzer.py` | 전체 분석 파이프라인 |
| `models/role05_match/relevance_detector.py` | 문항 적합성 분석 |
| `models/role05_match/question_clusterer.py` | 질문 군집화 |
| `models/role05_match/cluster_map.json` | 질문 유형별 지표 설정 |
| `models/role02_star/` | 두괄식·수치·STAR 관련 규칙 |
| `models/role03_hedge/` | 모호·추상 표현 탐지 |
| `models/role04_self/` | 자기중심/기여중심 표현 분석 |
