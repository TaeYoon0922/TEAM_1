"""
명사형 어근 추출 + TF-IDF(word) + SVD(LSA) + K-Means 군집화
형태소 분석기 없이 정규식으로 조사·어미 제거
"""
import sys, os, pickle, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.modules.setdefault("sentence_transformers", None)
sys.path.insert(0, os.path.dirname(__file__))

import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path

from models.role05_match.question_clusterer import (
    load_questions, _CACHE_DIR, _CLUSTER_MAP_PATH, save_cluster_map
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.cluster import KMeans

CACHE          = _CACHE_DIR
NOUN_Q_CACHE   = CACHE / "noun_questions.pkl"
NOUN_EMB_CACHE = CACHE / "noun_tfidf_lsa_embeddings.npy"
NOUN_VEC_CACHE = CACHE / "noun_tfidf_vectorizer.pkl"
NOUN_KM_CACHE  = CACHE / "noun_kmeans_model.pkl"

# ── 한국어 폰트 ────────────────────────────────────────────────────────────────
for fn in ["Malgun Gothic", "NanumGothic", "DejaVu Sans"]:
    if fn in {f.name for f in fm.fontManager.ttflist}:
        matplotlib.rcParams["font.family"] = fn; break
matplotlib.rcParams["axes.unicode_minus"] = False
COLORS = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2","#937860"]

# ── 조사·어미 제거 패턴 ────────────────────────────────────────────────────────
# 한국어 공백 분리 토큰에서 뒤에 붙는 조사·어미를 제거해 어근(명사)에 가깝게 만듦
_PARTICLE_PAT = re.compile(
    r"(이|가|을|를|은|는|의|에|에서|으로|로|와|과|도|만|랑|이랑"
    r"|에게|한테|께|부터|까지|처럼|보다|이나|나|라도|이라도"
    r"|하고|하여|하며|하는|하고|하니|하면|하여서|해서|해야"
    r"|이다|입니다|합니다|됩니다|습니다|십시오|시오|세요|주세요"
    r"|이며|이고|이지|이면|하지|않고|않은|않는|없고|없는|있고|있는"
    r"|위해|위한|통해|통한|관한|관련|따른|대한|기반|바탕)$"
)
_REMOVE_PAT = re.compile(r"[^가-힣a-zA-Z0-9]")  # 특수문자 제거


def extract_noun_like(text: str) -> str:
    """
    공백 단위 토큰 → 조사/어미 제거 → 2자 이상 어근만 반환.
    형태소 분석기 없이 규칙 기반으로 명사에 가까운 토큰 추출.
    """
    # 괄호 내 안내문·숫자 제거
    text = re.sub(r"\([^)]{5,}\)", " ", text)
    text = re.sub(r"\d+자\s*이내|\d+자\s*이상", " ", text)
    text = re.sub(r"[①②③④⑤]|\d+\.", " ", text)

    tokens = text.split()
    result = []
    for tok in tokens:
        tok = _REMOVE_PAT.sub("", tok)          # 특수문자 제거
        tok = _PARTICLE_PAT.sub("", tok)         # 조사·어미 제거
        if len(tok) >= 2:
            result.append(tok)
    return " ".join(result)


# ══════════════════════════════════════════════════════════════════════════════
# Step 1: 전처리 (명사형 어근 추출)
# ══════════════════════════════════════════════════════════════════════════════
questions = load_questions()
print(f"질문 수: {len(questions)}")

if NOUN_Q_CACHE.exists():
    with open(NOUN_Q_CACHE, "rb") as f:
        noun_docs = pickle.load(f)
    print(f"[캐시] 명사형 문서 로드: {len(noun_docs)}개")
else:
    print("명사형 어근 추출 중...")
    noun_docs = [extract_noun_like(q) for q in questions]
    with open(NOUN_Q_CACHE, "wb") as f:
        pickle.dump(noun_docs, f)
    print(f"저장 완료: {NOUN_Q_CACHE}")

print("\n[샘플 5개]")
for q, nd in list(zip(questions, noun_docs))[:5]:
    print(f"  원문: {q[:70]}")
    print(f"  처리: {nd[:70]}")
    print()

# ══════════════════════════════════════════════════════════════════════════════
# Step 2: TF-IDF (단어 1~2gram) + SVD(LSA)
# ══════════════════════════════════════════════════════════════════════════════
if NOUN_EMB_CACHE.exists():
    embeddings = np.load(NOUN_EMB_CACHE)
    with open(NOUN_VEC_CACHE, "rb") as f:
        vectorizer, svd = pickle.load(f)
    print(f"[캐시] 명사 임베딩 로드: {embeddings.shape}")
else:
    print("TF-IDF(단어) + LSA 임베딩 계산 중...")
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=20000,
        min_df=2,
        sublinear_tf=True,
    )
    tfidf = vectorizer.fit_transform(noun_docs)
    svd   = TruncatedSVD(n_components=128, random_state=42)
    emb   = svd.fit_transform(tfidf)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = emb / norms
    with open(NOUN_VEC_CACHE, "wb") as f:
        pickle.dump((vectorizer, svd), f)
    np.save(NOUN_EMB_CACHE, embeddings)
    print(f"임베딩 저장: {NOUN_EMB_CACHE}  shape={embeddings.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# Step 3: Elbow Method
# ══════════════════════════════════════════════════════════════════════════════
print("\nElbow Method 계산 중...")
inertias = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(embeddings)
    inertias.append(km.inertia_)
    print(f"  k={k:2d}  inertia={km.inertia_:,.1f}")

diffs     = np.diff(inertias)
diffs2    = np.diff(diffs)
optimal_k = list(range(2, 11))[int(np.argmax(np.abs(diffs2))) + 1]

plt.figure(figsize=(9, 5))
plt.plot(range(2, 11), inertias, "bo-", linewidth=2, markersize=8)
plt.axvline(x=optimal_k, color="red", linestyle="--", label=f"추천 k={optimal_k}")
plt.xlabel("군집 수 (k)"); plt.ylabel("Inertia")
plt.title("Elbow Method — 명사형 어근 TF-IDF 기반")
plt.xticks(range(2, 11)); plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(CACHE / "elbow_noun.png", dpi=150); plt.close()
print(f"\n추천 k = {optimal_k}")

# ══════════════════════════════════════════════════════════════════════════════
# Step 4: K-Means 학습
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nK-Means 학습 중 (k={optimal_k})...")
km = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
km.fit(embeddings)
with open(NOUN_KM_CACHE, "wb") as f:
    pickle.dump(km, f)
print(f"K-Means 저장: {NOUN_KM_CACHE}")

# ══════════════════════════════════════════════════════════════════════════════
# Step 5: 대표 질문 출력 (중심점 가장 가까운 top 5)
# ══════════════════════════════════════════════════════════════════════════════
labels  = km.labels_
centers = km.cluster_centers_
sizes   = [(labels == c).sum() for c in range(optimal_k)]

print(f"\n{'='*65}")
print(f"  군집별 대표 질문  (k={optimal_k})")
print(f"{'='*65}")

cluster_info = {}
for c in range(optimal_k):
    scored = []
    for i, (lbl, emb) in enumerate(zip(labels, embeddings)):
        if lbl != c:
            continue
        ne = np.linalg.norm(emb); nc = np.linalg.norm(centers[c])
        sim = float(np.dot(emb, centers[c]) / (ne * nc + 1e-9))
        scored.append((sim, i))
    scored.sort(reverse=True)

    print(f"\n[군집 {c}]  ({len(scored)}개 질문)")
    rep_qs = []
    for rank, (_, idx) in enumerate(scored[:5], 1):
        q  = questions[idx]
        nd = noun_docs[idx]
        print(f"  {rank}. {q[:100]}{'…' if len(q)>100 else ''}")
        print(f"     └ 어근: {nd[:60]}")
        rep_qs.append(q)
    cluster_info[f"cluster_{c}"] = {"size": len(scored),
                                     "representative_questions": rep_qs[:3]}

# ══════════════════════════════════════════════════════════════════════════════
# Step 6: 시각화
# ══════════════════════════════════════════════════════════════════════════════
print("\n시각화 중...")
features  = vectorizer.get_feature_names_out()
tfidf_all = vectorizer.transform(noun_docs)

# PCA 산점도
pca   = PCA(n_components=2, random_state=42)
emb2d = pca.fit_transform(embeddings)
fig, ax = plt.subplots(figsize=(10, 7))
for c in range(optimal_k):
    mask = labels == c
    ax.scatter(emb2d[mask,0], emb2d[mask,1], s=6, alpha=0.35,
               color=COLORS[c%len(COLORS)],
               label=f"군집{c} ({sizes[c]:,}개)")
c2d = pca.transform(centers)
for c in range(optimal_k):
    ax.scatter(c2d[c,0], c2d[c,1], s=200, marker="*",
               color=COLORS[c%len(COLORS)], edgecolors="black", zorder=5)
    ax.annotate(f" 군집{c}", xy=(c2d[c,0], c2d[c,1]), fontsize=9,
                fontweight="bold", color=COLORS[c%len(COLORS)])
vr = pca.explained_variance_ratio_
ax.set_xlabel(f"PC1 ({vr[0]*100:.1f}%)"); ax.set_ylabel(f"PC2 ({vr[1]*100:.1f}%)")
ax.set_title(f"자소서 질문 군집화 — 명사형 어근 TF-IDF → K-Means(k={optimal_k}) → PCA 2D")
ax.legend(fontsize=9, markerscale=3); ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(CACHE / "noun_pca_scatter.png", dpi=150); plt.close()

# 군집 크기
fig, ax = plt.subplots(figsize=(max(7, optimal_k*1.5), 4))
bars = ax.bar([f"군집{c}" for c in range(optimal_k)], sizes,
              color=COLORS[:optimal_k], edgecolor="white", width=0.5)
for bar, s in zip(bars, sizes):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+30,
            f"{s:,}개\n({s/len(labels)*100:.1f}%)",
            ha="center", va="bottom", fontsize=9)
ax.set_ylabel("질문 수"); ax.set_title("군집별 질문 수 (명사형 어근 기반)")
ax.set_ylim(0, max(sizes)*1.2); ax.grid(axis="y", alpha=0.3)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig(CACHE / "noun_cluster_size.png", dpi=150); plt.close()

# 군집별 상위 명사 키워드
cols = min(optimal_k, 4)
rows = (optimal_k + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
axes_flat = np.array(axes).flatten() if optimal_k > 1 else [axes]
for c in range(optimal_k):
    idxs   = [i for i, l in enumerate(labels) if l == c]
    sub    = tfidf_all[idxs]
    mean_v = np.asarray(sub.mean(axis=0)).flatten()
    top15  = mean_v.argsort()[-15:][::-1]
    words  = [features[i] for i in top15 if len(features[i]) >= 2][:15]
    scores = [mean_v[i]   for i in top15 if len(features[i]) >= 2][:15]
    ax = axes_flat[c]
    ax.barh(range(len(words)), scores, color=COLORS[c%len(COLORS)], alpha=0.85)
    ax.set_yticks(range(len(words))); ax.set_yticklabels(words, fontsize=9)
    ax.invert_yaxis()
    ax.set_title(f"군집{c} ({sizes[c]:,}개)", fontsize=10)
    ax.set_xlabel("평균 TF-IDF", fontsize=8)
    ax.grid(axis="x", alpha=0.3); ax.spines[["top","right"]].set_visible(False)
for ax in axes_flat[optimal_k:]: ax.set_visible(False)
plt.suptitle("군집별 상위 명사형 키워드", fontsize=13)
plt.tight_layout()
plt.savefig(CACHE / "noun_top_keywords.png", dpi=150, bbox_inches="tight"); plt.close()

print(f"  PCA 산점도:     {CACHE}/noun_pca_scatter.png")
print(f"  군집 크기:      {CACHE}/noun_cluster_size.png")
print(f"  상위 키워드:    {CACHE}/noun_top_keywords.png")

# cluster_map.json 초안 갱신
save_cluster_map(cluster_info, k=optimal_k)
print("\nALL DONE")
