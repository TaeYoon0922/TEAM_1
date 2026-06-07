"""
군집화 결과 시각화 스크립트
생성 파일:
  models/role05_match/cache/viz_pca_scatter.png  — PCA 2D 산점도
  models/role05_match/cache/viz_cluster_size.png — 군집 크기 막대 그래프
  models/role05_match/cache/viz_top_terms.png    — 군집별 상위 키워드
"""
import sys, os, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.dirname(__file__))
sys.modules.setdefault("sentence_transformers", None)  # type: ignore

# ── 경로 ──────────────────────────────────────────────────────────────────────
CACHE = Path("models/role05_match/cache")
OUT   = CACHE  # 같은 폴더에 저장

# ── 한국어 폰트 설정 ──────────────────────────────────────────────────────────
def _set_korean_font():
    candidates = ["Malgun Gothic", "NanumGothic", "AppleGothic", "DejaVu Sans"]
    available  = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            matplotlib.rcParams["font.family"] = font
            break
    matplotlib.rcParams["axes.unicode_minus"] = False

_set_korean_font()

CLUSTER_COLORS = ["#4C72B0", "#DD8452", "#55A868"]
CLUSTER_LABELS = ["군집 0", "군집 1", "군집 2"]   # cluster_map.json 이름 붙이면 여기도 수정

# ── 데이터 로드 ───────────────────────────────────────────────────────────────
print("데이터 로드 중...")
embeddings = np.load(CACHE / "train_tfidf_lsa_embeddings.npy")

with open(CACHE / "train_kmeans_model.pkl", "rb") as f:
    km = pickle.load(f)
labels = km.labels_

with open(CACHE / "train_unique_questions.pkl", "rb") as f:
    questions = pickle.load(f)

with open(CACHE / "train_tfidf_vectorizer.pkl", "rb") as f:
    vectorizer, svd = pickle.load(f)

k = km.n_clusters
print(f"군집 수: {k},  질문 수: {len(questions)}")


# ════════════════════════════════════════════════════════════════════════════
# 그림 1: PCA 2D 산점도
# ════════════════════════════════════════════════════════════════════════════
print("PCA 2D 산점도 생성 중...")
pca   = PCA(n_components=2, random_state=42)
emb2d = pca.fit_transform(embeddings)

fig, ax = plt.subplots(figsize=(10, 7))

for c in range(k):
    mask = labels == c
    ax.scatter(
        emb2d[mask, 0], emb2d[mask, 1],
        s=6, alpha=0.35,
        color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)],
        label=f"{CLUSTER_LABELS[c]} ({mask.sum():,}개)",
    )

# 군집 중심점 표시
centers_2d = pca.transform(km.cluster_centers_)
for c in range(k):
    ax.scatter(
        centers_2d[c, 0], centers_2d[c, 1],
        s=220, marker="*", linewidths=1.2,
        color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)],
        edgecolors="black", zorder=5,
    )
    ax.annotate(
        f" {CLUSTER_LABELS[c]}\n 중심",
        xy=(centers_2d[c, 0], centers_2d[c, 1]),
        fontsize=9, fontweight="bold",
        color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)],
    )

var_ratio = pca.explained_variance_ratio_
ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}% 분산 설명)", fontsize=11)
ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}% 분산 설명)", fontsize=11)
ax.set_title("자소서 질문 군집화 결과\nTF-IDF+LSA → K-Means(k=3) → PCA 2D", fontsize=13)
ax.legend(fontsize=10, markerscale=3)
ax.grid(True, alpha=0.2)
plt.tight_layout()
out1 = OUT / "viz_pca_scatter.png"
plt.savefig(out1, dpi=150)
plt.close()
print(f"  저장: {out1}")


# ════════════════════════════════════════════════════════════════════════════
# 그림 2: 군집 크기 막대 그래프
# ════════════════════════════════════════════════════════════════════════════
print("군집 크기 막대 그래프 생성 중...")
sizes = [(labels == c).sum() for c in range(k)]

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(
    CLUSTER_LABELS[:k], sizes,
    color=CLUSTER_COLORS[:k],
    edgecolor="white", linewidth=1.5,
    width=0.5,
)
for bar, size in zip(bars, sizes):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 80,
        f"{size:,}개\n({size/len(questions)*100:.1f}%)",
        ha="center", va="bottom", fontsize=10,
    )
ax.set_ylabel("질문 수", fontsize=11)
ax.set_title("군집별 질문 수 분포", fontsize=13)
ax.set_ylim(0, max(sizes) * 1.2)
ax.grid(axis="y", alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
out2 = OUT / "viz_cluster_size.png"
plt.savefig(out2, dpi=150)
plt.close()
print(f"  저장: {out2}")


# ════════════════════════════════════════════════════════════════════════════
# 그림 3: 군집별 상위 TF-IDF 단어 수평 막대
# ════════════════════════════════════════════════════════════════════════════
print("군집별 상위 키워드 그래프 생성 중...")

q_arr    = np.array(questions)
features = vectorizer.get_feature_names_out()
top_n    = 15

fig, axes = plt.subplots(1, k, figsize=(6 * k, 6), sharey=False)
if k == 1:
    axes = [axes]

for c, ax in enumerate(axes):
    cluster_qs = q_arr[labels == c].tolist()
    # 해당 군집 질문들의 TF-IDF 평균 벡터
    tfidf_mat  = vectorizer.transform(cluster_qs)
    mean_vec   = np.asarray(tfidf_mat.mean(axis=0)).flatten()

    top_idx    = mean_vec.argsort()[-top_n:][::-1]
    top_words  = [features[i] for i in top_idx]
    top_scores = [mean_vec[i] for i in top_idx]

    # 2글자 이상 필터
    pairs = [(w, s) for w, s in zip(top_words, top_scores) if len(w) >= 2][:top_n]
    words, scores = zip(*pairs) if pairs else ([], [])

    color = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
    bars  = ax.barh(range(len(words)), scores, color=color, alpha=0.8)
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("평균 TF-IDF 가중치", fontsize=9)
    ax.set_title(f"{CLUSTER_LABELS[c]}\n({sizes[c]:,}개 질문)", fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

plt.suptitle("군집별 상위 특징 키워드 (TF-IDF 평균)", fontsize=13, y=1.01)
plt.tight_layout()
out3 = OUT / "viz_top_terms.png"
plt.savefig(out3, dpi=150, bbox_inches="tight")
plt.close()
print(f"  저장: {out3}")


print("\n=== 시각화 완료 ===")
print(f"  {out1}")
print(f"  {out2}")
print(f"  {out3}")
