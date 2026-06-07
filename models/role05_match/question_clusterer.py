"""
질문 군집화 모듈
Sentence-BERT(jhgan/ko-sroberta-multitask) + K-Means 기반 자소서 질문 유형 분류
SBERT 사용 불가 시 TF-IDF char n-gram + SVD(LSA) 방식으로 자동 폴백

설치:
    pip install sentence-transformers scikit-learn pandas matplotlib
"""

import os
import re
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
_DIR = Path(__file__).parent
_ROOT = _DIR.parent.parent
_DATA_PATH = _ROOT / "data" / "raw" / "잡코리아 데이터셋 원문 7000개.csv"
_CACHE_DIR = _DIR / "cache"
_CLUSTER_MAP_PATH = _DIR / "cluster_map.json"

_CACHE_DIR.mkdir(parents=True, exist_ok=True)

EMBED_CACHE     = _CACHE_DIR / "sbert_embeddings.npy"
TFIDF_EMB_CACHE = _CACHE_DIR / "tfidf_lsa_embeddings.npy"
QUESTIONS_CACHE = _CACHE_DIR / "unique_questions.pkl"
KMEANS_CACHE    = _CACHE_DIR / "kmeans_model.pkl"
TFIDF_CACHE     = _CACHE_DIR / "tfidf_embeddings.npy"
VECTORIZER_CACHE = _CACHE_DIR / "tfidf_vectorizer.pkl"

# ── SBERT 사용 가능 여부 체크 ─────────────────────────────────────────────────
_SBERT_AVAILABLE = False
_sbert_model = None

try:
    from sentence_transformers import SentenceTransformer as _ST
    _SBERT_AVAILABLE = True
except ImportError:
    pass


def get_sbert_model():
    global _sbert_model
    if not _SBERT_AVAILABLE:
        raise RuntimeError("sentence_transformers 미설치")
    if _sbert_model is None:
        print("[QuestionClusterer] Sentence-BERT 모델 로딩 중... (최초 1회)")
        _sbert_model = _ST("jhgan/ko-sroberta-multitask")
        print("[QuestionClusterer] 모델 로딩 완료")
    return _sbert_model


# ══════════════════════════════════════════════════════════════════════════════
# Step 1: 데이터 로드 및 전처리
# ══════════════════════════════════════════════════════════════════════════════

def _preprocess(text: str) -> str:
    """질문 텍스트 정규화 — 안내문·괄호 내용·공백 정리"""
    text = re.sub(r"※[^\n]*", "", text)       # ※ 이하 안내문 제거
    text = re.sub(r"\([^)]{20,}\)", "", text)  # 긴 괄호(안내문) 제거
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_questions(csv_path: str = None, force_reload: bool = False) -> list:
    """
    CSV에서 question 컬럼 추출 → 전처리 → 유니크 질문 리스트 반환.
    캐시가 있으면 재사용.

    Returns:
        list[str]: 전처리된 유니크 질문 목록
    """
    if not force_reload and QUESTIONS_CACHE.exists():
        with open(QUESTIONS_CACHE, "rb") as f:
            questions = pickle.load(f)
        print(f"[캐시] 유니크 질문 {len(questions)}개 로드")
        return questions

    path = csv_path or str(_DATA_PATH)
    print(f"데이터 로드 중: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")

    if "question" not in df.columns:
        raise ValueError(f"'question' 컬럼 없음. 존재하는 컬럼: {list(df.columns)}")

    raw = df["question"].dropna().astype(str).tolist()
    processed = [_preprocess(q) for q in raw]
    processed = [q for q in processed if len(q) >= 10]
    questions = list(dict.fromkeys(processed))  # 순서 유지 중복 제거

    print(f"전처리 완료: 원본 {len(raw)}개 → 유니크 {len(questions)}개")

    with open(QUESTIONS_CACHE, "wb") as f:
        pickle.dump(questions, f)

    return questions


# ══════════════════════════════════════════════════════════════════════════════
# Step 2: Sentence-BERT 임베딩
# ══════════════════════════════════════════════════════════════════════════════

def get_sbert_embeddings(questions: list, force_reload: bool = False,
                          sample_n: int = None) -> np.ndarray:
    """
    질문 리스트 → SBERT 임베딩 numpy 배열 (shape: N × 768).
    캐시가 있으면 재사용.

    Args:
        sample_n: 메모리 절약을 위해 질문을 샘플링할 개수. None이면 전체 사용.
    """
    if not force_reload and EMBED_CACHE.exists():
        embeddings = np.load(EMBED_CACHE)
        print(f"[캐시] SBERT 임베딩 로드: {embeddings.shape}")
        return embeddings

    if sample_n and len(questions) > sample_n:
        import random
        random.seed(42)
        questions = random.sample(questions, sample_n)
        print(f"메모리 절약: {sample_n}개 샘플 사용")

    print(f"SBERT 임베딩 계산 중... ({len(questions)}개, batch_size=16)")
    model = get_sbert_model()

    # 메모리 절약: 청크 단위로 인코딩 후 합치기
    chunk = 500
    parts = []
    for i in range(0, len(questions), chunk):
        batch = questions[i:i + chunk]
        emb = model.encode(
            batch,
            batch_size=16,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        parts.append(emb)
        print(f"  {min(i + chunk, len(questions))}/{len(questions)} 완료")

    embeddings = np.vstack(parts)
    np.save(EMBED_CACHE, embeddings)
    print(f"임베딩 저장: {EMBED_CACHE}  shape={embeddings.shape}")
    return embeddings


# ══════════════════════════════════════════════════════════════════════════════
# Step 2b: TF-IDF + SVD(LSA) 임베딩 (SBERT 폴백)
# ══════════════════════════════════════════════════════════════════════════════

def get_tfidf_lsa_embeddings(questions: list, n_components: int = 128,
                              force_reload: bool = False) -> np.ndarray:
    """
    TF-IDF char n-gram(2-4) → TruncatedSVD(LSA) → dense 임베딩.
    SBERT 사용 불가 환경에서의 폴백. 캐시 지원.

    Returns:
        np.ndarray: shape (N, n_components)
    """
    if not force_reload and TFIDF_EMB_CACHE.exists():
        emb = np.load(TFIDF_EMB_CACHE)
        print(f"[캐시] TF-IDF+LSA 임베딩 로드: {emb.shape}")
        return emb

    print(f"TF-IDF+LSA 임베딩 계산 중... ({len(questions)}개, dim={n_components})")
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        max_features=50000,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(questions)

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    embeddings = svd.fit_transform(tfidf_matrix)

    # 벡터 L2 정규화 (코사인 유사도 준비)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms

    with open(VECTORIZER_CACHE, "wb") as f:
        pickle.dump((vectorizer, svd), f)
    np.save(TFIDF_EMB_CACHE, embeddings)
    print(f"TF-IDF+LSA 임베딩 저장: {TFIDF_EMB_CACHE}  shape={embeddings.shape}")
    return embeddings


def get_embeddings(questions: list, force_reload: bool = False,
                   sample_n: int = None) -> tuple:
    """
    SBERT 가능 → SBERT 임베딩 반환.
    SBERT 불가 → TF-IDF+LSA 임베딩 반환.

    Returns:
        (embeddings: np.ndarray, method: str)
    """
    if _SBERT_AVAILABLE:
        try:
            emb = get_sbert_embeddings(questions, force_reload=force_reload,
                                        sample_n=sample_n)
            return emb, "sbert"
        except Exception as e:
            print(f"[SBERT 실패] {e}  → TF-IDF+LSA로 폴백")

    emb = get_tfidf_lsa_embeddings(questions, force_reload=force_reload)
    return emb, "tfidf_lsa"


# ══════════════════════════════════════════════════════════════════════════════
# Step 3: TF-IDF vs Sentence-BERT 비교 실험
# ══════════════════════════════════════════════════════════════════════════════

def compare_tfidf_sbert(questions: list, sbert_embeddings: np.ndarray,
                        k: int = 7) -> tuple:
    """
    두 임베딩 방식으로 K-Means 학습 후,
    의미상 유사한 질문 쌍이 같은 군집에 속하는지 비교.

    Returns:
        (km_tfidf, km_sbert, vectorizer)
    """
    test_pairs = [
        ("팀 갈등 해결 경험", "협업 중 어려움을 극복한 사례"),
        ("지원 동기", "우리 회사를 선택한 이유"),
        ("입사 후 포부", "5년 후 목표와 계획"),
        ("성격의 장단점", "본인의 강점과 약점"),
    ]

    # TF-IDF (문자 n-gram — 한국어에 효과적)
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    tfidf_matrix = vectorizer.fit_transform(questions)

    km_tfidf = KMeans(n_clusters=k, random_state=42, n_init=10)
    km_tfidf.fit(tfidf_matrix)

    km_sbert = KMeans(n_clusters=k, random_state=42, n_init=10)
    km_sbert.fit(sbert_embeddings)

    sbert_model = get_sbert_model()

    print("\n" + "=" * 65)
    print(f"  TF-IDF vs Sentence-BERT 군집화 비교  (k={k})")
    print("=" * 65)

    for q1, q2 in test_pairs:
        v1 = vectorizer.transform([q1])
        v2 = vectorizer.transform([q2])
        c1_tf = km_tfidf.predict(v1)[0]
        c2_tf = km_tfidf.predict(v2)[0]

        e1 = sbert_model.encode([q1])
        e2 = sbert_model.encode([q2])
        c1_sb = km_sbert.predict(e1)[0]
        c2_sb = km_sbert.predict(e2)[0]

        tf_label  = "✅ 같은 군집" if c1_tf == c2_tf else "❌ 다른 군집"
        sb_label  = "✅ 같은 군집" if c1_sb == c2_sb else "❌ 다른 군집"

        print(f"\n  Q1: \"{q1}\"")
        print(f"  Q2: \"{q2}\"")
        print(f"  TF-IDF : {tf_label}  (cluster {c1_tf} / {c2_tf})")
        print(f"  SBERT  : {sb_label}  (cluster {c1_sb} / {c2_sb})")

    print("\n" + "=" * 65)

    np.save(TFIDF_CACHE, tfidf_matrix.toarray())
    return km_tfidf, km_sbert, vectorizer


# ══════════════════════════════════════════════════════════════════════════════
# Step 4: Elbow Method
# ══════════════════════════════════════════════════════════════════════════════

def plot_elbow(embeddings: np.ndarray, k_range=range(2, 11),
               save_path: str = None) -> int:
    """
    Elbow Method로 최적 k 결정.
    2차 미분 최대점을 자동 추천하고 matplotlib으로 시각화.

    Returns:
        int: 추천 k
    """
    inertias = []
    print("\nElbow Method 계산 중...")
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(embeddings)
        inertias.append(km.inertia_)
        print(f"  k={k:2d}  inertia={km.inertia_:,.1f}")

    # 2차 미분으로 꺾이는 지점 자동 탐지
    diffs  = np.diff(inertias)
    diffs2 = np.diff(diffs)
    optimal_k = list(k_range)[int(np.argmax(np.abs(diffs2))) + 1]

    matplotlib.use("Agg")  # 헤드리스 환경 — 창 없이 파일로만 저장
    matplotlib.rcParams["font.family"]     = "Malgun Gothic"
    matplotlib.rcParams["axes.unicode_minus"] = False

    plt.figure(figsize=(9, 5))
    plt.plot(list(k_range), inertias, "bo-", linewidth=2, markersize=8)
    plt.axvline(x=optimal_k, color="red", linestyle="--",
                label=f"추천 k = {optimal_k}")
    plt.xlabel("군집 수 (k)")
    plt.ylabel("Inertia (Within-cluster SSE)")
    plt.title("Elbow Method — 최적 군집 수 결정 (SBERT 임베딩)")
    plt.xticks(list(k_range))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out = save_path or str(_CACHE_DIR / "elbow_curve.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\n[Elbow Method] 추천 k = {optimal_k}  (elbow curve → {out})")
    return optimal_k


# ══════════════════════════════════════════════════════════════════════════════
# Step 5: K-Means 군집화
# ══════════════════════════════════════════════════════════════════════════════

def fit_kmeans(embeddings: np.ndarray, k: int,
               force_reload: bool = False) -> KMeans:
    """K-Means 학습 후 pickle 캐시. 동일 k면 캐시 재사용."""
    if not force_reload and KMEANS_CACHE.exists():
        with open(KMEANS_CACHE, "rb") as f:
            km = pickle.load(f)
        if km.n_clusters == k:
            print(f"[캐시] K-Means 모델 로드 (k={k})")
            return km
        print(f"캐시 k={km.n_clusters} ≠ 요청 k={k} → 재학습")

    print(f"K-Means 학습 중 (k={k}, n_init=10)...")
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(embeddings)

    with open(KMEANS_CACHE, "wb") as f:
        pickle.dump(km, f)
    print(f"K-Means 모델 저장: {KMEANS_CACHE}")
    return km


def print_cluster_representatives(km: KMeans, questions: list,
                                   embeddings: np.ndarray, top_n: int = 5) -> dict:
    """
    각 군집의 중심점에 가장 가까운 대표 질문 top_n개 출력.
    군집 이름은 출력 결과를 보고 사용자가 직접 cluster_map.json에 기입.

    Returns:
        dict: {"cluster_0": {"size": N, "representative_questions": [...]}, ...}
    """
    labels   = km.labels_
    centers  = km.cluster_centers_
    q_arr    = np.array(questions)

    print("\n" + "=" * 65)
    print(f"  군집별 대표 질문  (k={km.n_clusters}, top {top_n}개)")
    print("=" * 65)
    print("아래를 보고 cluster_map.json의 'name' 필드를 직접 수정하세요.\n")

    cluster_info = {}
    for c in range(km.n_clusters):
        mask = labels == c
        c_embs = embeddings[mask]
        c_qs   = q_arr[mask].tolist()

        sims      = cosine_similarity(c_embs, centers[c].reshape(1, -1)).flatten()
        top_idx   = sims.argsort()[-top_n:][::-1]
        rep_qs    = [c_qs[i] for i in top_idx]

        print(f"[군집 {c}]  ({len(c_qs)}개 질문)")
        for rank, q in enumerate(rep_qs, 1):
            print(f"  {rank}. {q[:70]}{'…' if len(q) > 70 else ''}")
        print()

        cluster_info[f"cluster_{c}"] = {
            "size": len(c_qs),
            "representative_questions": rep_qs,
        }

    return cluster_info


# ══════════════════════════════════════════════════════════════════════════════
# Step 6: 군집-평가모델 매핑
# ══════════════════════════════════════════════════════════════════════════════

def load_cluster_map(path: str = None) -> dict:
    """cluster_map.json 로드. 없으면 빈 dict."""
    p = Path(path or _CLUSTER_MAP_PATH)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def get_evaluation_models(cluster_id: int, cluster_map: dict = None) -> dict:
    """
    군집 번호 → 평가 모델 정보 반환.

    Returns:
        {
          "name": "경험형",
          "models": ["star", "hedge", "self"],
          "star_required": True
        }
    """
    cmap = cluster_map if cluster_map is not None else load_cluster_map()
    key  = f"cluster_{cluster_id}"
    if key in cmap:
        return {k: v for k, v in cmap[key].items()
                if k not in ("representative_questions", "description")}
    return {
        "name": f"cluster_{cluster_id} (매핑 없음)",
        "models": ["star", "hedge"],
        "star_required": True,
    }


def save_cluster_map(cluster_info: dict, k: int, path: str = None):
    """
    군집화 결과로 cluster_map.json 초안 생성.
    name / models / star_required 는 사용자가 직접 수정해야 함.
    """
    p = Path(path or _CLUSTER_MAP_PATH)

    cmap = {}
    for i in range(k):
        key  = f"cluster_{i}"
        info = cluster_info.get(key, {})
        cmap[key] = {
            "name": f"군집{i}_이름미정",
            "models": ["star", "hedge"],
            "star_required": True,
            "description": "대표 질문을 보고 name/models/star_required를 수정하세요",
            "representative_questions": info.get("representative_questions", [])[:3],
        }

    with open(p, "w", encoding="utf-8") as f:
        json.dump(cmap, f, ensure_ascii=False, indent=2)

    print(f"\ncluster_map.json 초안 저장: {p}")
    print("→ 'name', 'models', 'star_required'를 직접 채워주세요.")
    print("  models 가능 값: star | hedge | self | clarity")


# ══════════════════════════════════════════════════════════════════════════════
# Step 7: 새 질문 입력 시 군집 예측
# ══════════════════════════════════════════════════════════════════════════════

def _load_km() -> KMeans:
    if not KMEANS_CACHE.exists():
        raise RuntimeError("K-Means 모델 없음. fit_kmeans()를 먼저 실행하세요.")
    with open(KMEANS_CACHE, "rb") as f:
        return pickle.load(f)


def _embed_single(question: str) -> np.ndarray:
    """질문 1개를 임베딩 (SBERT 우선, 없으면 TF-IDF+LSA)."""
    q = _preprocess(question)
    if _SBERT_AVAILABLE:
        try:
            return get_sbert_model().encode([q])
        except Exception:
            pass
    # TF-IDF+LSA 폴백
    if not VECTORIZER_CACHE.exists():
        raise RuntimeError("벡터라이저 없음. get_tfidf_lsa_embeddings()를 먼저 실행하세요.")
    with open(VECTORIZER_CACHE, "rb") as f:
        vectorizer, svd = pickle.load(f)
    tfidf = vectorizer.transform([q])
    emb = svd.transform(tfidf)
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 0 else emb


def _rule_cluster_override(question: str) -> int | None:
    """명확한 질문 신호는 K-Means보다 우선한다.

    K-Means 군집은 지표 라우팅용 보조 장치라, 흔한 자소서 문항의 강한
    키워드는 규칙으로 먼저 잡아 군집 쏠림을 줄인다.
    """
    q = _preprocess(question).lower()

    # 가치관/견해형은 "지원 직무 관련 Trend"처럼 직무 단어가 섞여도 견해가 핵심이다.
    if any(signal in q for signal in ("trend", "트렌드", "견해", "가치관", "좌우명", "인재상", "핵심가치")):
        return 5
    if any(signal in q for signal in ("협업", "팀워크", "공동", "갈등", "타인과", "팀원")):
        return 4
    if any(signal in q for signal in ("성장과정", "성격", "장단점", "장/단점")):
        return 3
    if any(signal in q for signal in ("지원동기", "지원 동기", "입사 후", "포부")):
        return 0
    if any(signal in q for signal in ("역량", "강점", "전문성", "직무 수행", "직무와 관련")):
        return 2
    if any(signal in q for signal in ("문제", "해결", "도전", "성과", "어려움", "극복", "목표를 달성")):
        return 1

    return None


def predict_cluster(question: str, km: KMeans = None,
                    cluster_map: dict = None) -> dict:
    """
    새 질문 텍스트 → 임베딩 → 가장 가까운 군집 + 평가 모델 반환.

    Returns:
        {
          "question": "...",
          "cluster_id": 2,
          "cluster_info": {"name": "경험형", "models": [...], "star_required": True}
        }
    """
    override = _rule_cluster_override(question)
    if override is None:
        emb = _embed_single(question)
        km = km or _load_km()
        cid = int(km.predict(emb)[0])
    else:
        cid = override
    cmap = cluster_map if cluster_map is not None else load_cluster_map()

    return {
        "question":     question,
        "cluster_id":   cid,
        "cluster_info": get_evaluation_models(cid, cmap),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Step 8: 혼합(복합) 질문 처리
# ══════════════════════════════════════════════════════════════════════════════

_COMPOUND_MARKERS = ["바탕으로", "연계하여", "통해", "함께", "더불어", "활용하여"]


def split_subquestions(question: str) -> list:
    """
    복합 질문을 단위 질문으로 분리.
    ① ② 번호형, 줄바꿈형, 복합 접속표현 순으로 시도.

    Examples:
        "경험을 바탕으로 입사 후 포부를 서술하시오"
        → ["경험을", "바탕으로 입사 후 포부를 서술하시오"]
    """
    # 번호·기호 분리
    for pat in [r"[①②③④⑤]", r"\d+\.\s+", r"\n"]:
        parts = re.split(pat, question)
        clean = [p.strip() for p in parts if len(p.strip()) >= 10]
        if len(clean) >= 2:
            return clean

    # 복합 접속표현으로 의미 단위 분리
    for marker in _COMPOUND_MARKERS:
        if marker in question:
            idx   = question.index(marker)
            left  = question[:idx].strip()
            right = question[idx:].strip()
            if len(left) >= 10 and len(right) >= 10:
                return [left, right]

    return [question]


def predict_cluster_multi(question: str, km: KMeans = None,
                           cluster_map: dict = None) -> dict:
    """
    혼합 질문 처리: 분리 → 각각 군집 분류 → 평가 모델 합집합.

    Returns:
        {
          "original_question": "...",
          "subquestions": [...],
          "cluster_ids": [0, 2],
          "merged_models": ["clarity", "hedge", "star"],
          "star_required": True,
          "sub_results": [...]
        }
    """
    km       = km or _load_km()
    cmap     = cluster_map if cluster_map is not None else load_cluster_map()
    subqs    = split_subquestions(question)

    sub_results    = []
    merged_models  = set()
    star_required  = False

    for sq in subqs:
        res = predict_cluster(sq, km=km, cluster_map=cmap)
        sub_results.append(res)
        merged_models.update(res["cluster_info"].get("models", []))
        if res["cluster_info"].get("star_required", False):
            star_required = True

    return {
        "original_question": question,
        "subquestions":      subqs,
        "cluster_ids":       [r["cluster_id"] for r in sub_results],
        "merged_models":     sorted(merged_models),
        "star_required":     star_required,
        "sub_results":       sub_results,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLI 실행 예시
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # 1. 데이터 로드
    questions = load_questions()

    # 2. SBERT 임베딩
    embeddings = get_sbert_embeddings(questions)

    # 3. TF-IDF vs SBERT 비교
    compare_tfidf_sbert(questions, embeddings, k=7)

    # 4. Elbow Method
    optimal_k = plot_elbow(embeddings)

    # 5. K-Means 학습
    km = fit_kmeans(embeddings, k=optimal_k)

    # 6. 대표 질문 출력 + cluster_map.json 초안 저장
    cluster_info = print_cluster_representatives(km, questions, embeddings)
    save_cluster_map(cluster_info, k=optimal_k)

    # 7. 새 질문 예측 테스트
    print("\n[예측 테스트]")
    test_questions = [
        "팀 갈등 해결 경험에 대해 서술하시오",
        "입사 후 포부를 기술하시오",
        "지원 동기를 작성하시오",
    ]
    cmap = load_cluster_map()
    for q in test_questions:
        result = predict_cluster(q, km=km, cluster_map=cmap)
        print(f"  Q: {q}")
        print(f"  → 군집 {result['cluster_id']}  {result['cluster_info']}\n")

    # 8. 혼합 질문 테스트
    print("[혼합 질문 테스트]")
    mixed = "경험을 바탕으로 입사 후 포부를 서술하시오"
    res = predict_cluster_multi(mixed, km=km, cluster_map=cmap)
    print(f"  원본: {res['original_question']}")
    print(f"  분리: {res['subquestions']}")
    print(f"  군집: {res['cluster_ids']}")
    print(f"  평가 모델: {res['merged_models']}")
    print(f"  STAR 필요: {res['star_required']}")
