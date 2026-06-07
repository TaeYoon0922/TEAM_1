# ROLE05 인수인계 / 프로젝트 정리

다른 팀원이 이어받아 작업·발표·제출할 수 있도록 정리한 문서.

---

## 1. 프로젝트 한 줄 요약

자소서 **질문 + 답변**을 입력받아 → **질문에 맞는 답인지 먼저 거르고(게이트)** → **질문 유형을 자동 분류(군집)** → **유형에 맞는 지표만 골라** → **점수 대신 고칠 수 있는 피드백**을 보여주는 시스템.

---

## 2. 전체 흐름

```
[질문 + 답변]
   → ① 문항 적합성 게이트 (동문서답이면 세부분석 중단)
   → ② 질문 군집화 (SBERT + KMeans, 6유형)
   → ③ 군집별 지표 게이팅 (cluster_map.json)
   → ④ 6지표 피드백 + 문장별 코멘트 (점수 미노출)
```

---

## 3. 최종 지표(6개)와 연결 모듈

| 지표 | 연결 | 상태 |
|---|---|---|
| 문항 적합성(게이트) | relevance_detector + SBERT 코사인 하이브리드 | 연결 |
| 핵심 주장 명확성 | KoSimCSE(두괄식), 규칙 폴백 | 연결 |
| 경험 구체성 | STAR 키워드 규칙(role02) | 규칙 |
| 표현 명료성 | hedge_detector(role03, konlpy/Okt) | 연결(폴백 有) |
| 자기표현 차별성 | self_detector + dependency_parser(spacy+kiwipiepy) | 연결 |
| 문장 완성도 | 길이·종결어 반복 규칙 | 규칙 |

- **지원 적합성**은 직무 사전 변별력 부족으로 **평가지표에서 제외**(코드는 `score_job_fit`/role03 `job_fit_scorer`로 **보존**, 사전 보강 시 재연결 가능).
- 군집 6유형: 0 성장과정·성격 / 1 협업·팀워크 / 2 문제해결·성과경험 / 3 지원동기·포부 / 4 직무역량·강점 / 5 가치관·견해. **이 인덱스는 `cluster_map.json` ↔ `_rule_cluster_override` ↔ KMeans 라벨이 모두 일치해야 함.**

---

## 4. 실행 방법 (clone 후 동일 동작)

```bash
git clone <repo> && cd TEAM_1
pip install -r requirements.txt        # spacy 한국어 모델까지 함께 설치됨
streamlit run app/streamlit_app.py
```
- 최초 실행 시 SBERT(ko-sroberta)·KoSimCSE 모델 자동 다운로드(인터넷 필요).
- 군집 모델(`models/role05_match/cache/train_kmeans_model.pkl`)은 git에 포함 → 군집화 바로 동작.
- **선택**: 표현 명료성 형태소 정밀모드는 Java JDK 필요(`winget install Microsoft.OpenJDK.17` + `JAVA_HOME`). 없으면 규칙 폴백(앱 정상).

---

## 5. 이번 세션에 한 일

- 질문 군집화 **SBERT k=6** 전환(명사 TF-IDF의 catch-all 43% → 균형화).
- 출력 구조를 **7→6지표 피드백 중심**으로 전환(`overall_score`/점수 막대·차트 제거).
- **문항 적합성 게이트 + 군집 기반 지표 게이팅** 구현(`cluster_map.json` indicators).
- 문항 적합성 **키워드 + SBERT 하이브리드**(패러프레이즈 오차단 방지).
- 🐛 `relevance_detector.py` **두 모듈 concat → SyntaxError** 버그 수정.
- 지원 적합성 **지표 제외**(코드 보존), 문장 완성도 등급/피드백 불일치 수정.
- streamlit HTML **compact 전환**(카드 raw 출력 버그 해결).
- dev 최신 머지 반영 + 데이터 소스 교체(jobkorea_train.csv)에 맞춰 **군집·규칙 인덱스 재정렬**.
- **clone 재현성 복구**: requirements 전체 복원(+spacy 모델 wheel), KMeans 모델 git 포함, import 가드 `except Exception` 폴백.
- 환경: konlpy+JPype1+OpenJDK17 설치, README 설치/임계값 가이드.

주요 커밋: `a63e015`(deploy 복구) · `957af43`(군집 재정렬) · `78c8844`(정리) · `10524e9`(지원적합성 제외) 등.

---

## 6. 남은 할 일 (TODO)

1. **role05 → dev PR 생성** — 현재 role05가 dev보다 2커밋 앞섬(`957af43`, `a63e015`). `gh` 미설치라 웹에서: `compare/dev...role05`.
2. **게이트 임계값(GATE_THRESHOLD=20) 재보정** — 현재 합성 라벨로는 보정 불가라 보수값. 사람이 라벨링한 동문서답 Q-A 데이터가 생기면 `scripts/calibrate_relevance_gate.py`로 재산출.
3. **경험 구체성 · 문장 완성도** = 규칙 기반 → 전용 모델/검사기 붙이면 개선 여지.
4. **지원 적합성 재연결(선택)** — 잡코리아 원문에서 직무 고유어(반도체·회로 등) 사전 재구축 후 `score_job_fit` 되살리기. (현 role03 사전은 generic해서 변별력 낮음)
5. ⚠️ **`run_clustering_sbert_k6.py` footgun** — 실행 시 `save_cluster_map`이 cluster_map.json을 placeholder 이름으로 덮어씀. 재학습 시 이름/정렬 날아감 → "캐시만 재생성, cluster_map 유지"로 분리 권장.
6. **군집 데이터 소스가 또 바뀌면**(role01/02) → KMeans 라벨이 바뀌므로 `cluster_map.json` + `_rule_cluster_override` **인덱스 재정렬 필수**(이번에 한 작업과 동일).
7. (선택) Java JDK 미설치 환경 안내 — 표현 명료성 정밀모드용.

---

## 7. 주요 파일 위치

- `app/streamlit_app.py` — UI(피드백 카드 + 문장 코멘트)
- `app/analyzer.py` — 파이프라인 통합(게이트·게이팅·6지표·폴백 가드)
- `models/role05_match/relevance_detector.py` — 문항 적합성(키워드)
- `models/role05_match/question_clusterer.py` — 군집화 + 규칙 라우팅(`_rule_cluster_override`)
- `models/role05_match/cluster_map.json` — 군집별 적용 지표/이름
- `models/role05_match/cache/train_kmeans_model.pkl` — 런타임 군집 모델(git 포함)
- `scripts/calibrate_relevance_gate.py` — 게이트 임계값 보정용
- `docs/ROLE05_평가파이프라인_정리.md` — 발표/보고서용 상세 정리
```
