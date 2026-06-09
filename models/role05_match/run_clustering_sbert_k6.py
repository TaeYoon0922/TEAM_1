"""
SBERT(ko-sroberta-multitask) 의미 임베딩 + K-Means(k=6) 재군집.
런타임 예측이 쓰는 train_kmeans_model.pkl(KMEANS_CACHE)에 기록 → SBERT가 실배포 군집이 됨.
"""
import sys, os, io, pickle
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from models.role05_match.question_clusterer import (
    load_questions, get_sbert_embeddings, fit_kmeans,
    print_cluster_representatives, save_cluster_map,
    _CACHE_DIR, _SBERT_AVAILABLE,
)

K = 6
CACHE   = _CACHE_DIR
OUT_DIR = Path("models/role05_match/clusters")
OUT_DIR.mkdir(parents=True, exist_ok=True)

for fn in ["Malgun Gothic", "NanumGothic", "DejaVu Sans"]:
    if fn in {f.name for f in fm.fontManager.ttflist}:
        matplotlib.rcParams["font.family"] = fn; break
matplotlib.rcParams["axes.unicode_minus"] = False
COLORS = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2","#937860"]

print(f"SBERT 사용 가능: {_SBERT_AVAILABLE}")
if not _SBERT_AVAILABLE:
    raise SystemExit("sentence-transformers 미설치 — 설치 후 다시 실행하세요.")

# 1) 질문 + SBERT 임베딩 (캐시 없으면 계산·저장)
questions  = load_questions()
embeddings = get_sbert_embeddings(questions)
print(f"질문 {len(questions)}개 / SBERT 임베딩 {embeddings.shape}")

# 2) K-Means k=6 → train_kmeans_model.pkl (런타임 모델)
km = fit_kmeans(embeddings, k=K, force_reload=True)
labels  = km.labels_
sizes   = [int((labels == c).sum()) for c in range(K)]

# 3) 대표 질문 + cluster_map 초안 (cluster_map.draft.json — 운영 cluster_map.json은 보존됨)
cluster_info = print_cluster_representatives(km, questions, embeddings, top_n=5)
save_cluster_map(cluster_info, k=K)

# 4) 군집별 CSV (SBERT는 noun_form 없음 → question만)
df = pd.DataFrame({"cluster_id": labels, "question": questions})
for c in range(K):
    sub = df[df["cluster_id"] == c].reset_index(drop=True)
    sub.index += 1
    sub.to_csv(OUT_DIR / f"cluster_{c}.csv", encoding="utf-8-sig", index_label="rank")
    print(f"  cluster_{c}.csv ({len(sub)}개)")
df.sort_values("cluster_id").to_csv(OUT_DIR / "all_clusters.csv",
                                    encoding="utf-8-sig", index=False)

# 5) 시각화: 군집 크기
fig, ax = plt.subplots(figsize=(max(7, K*1.5), 4))
bars = ax.bar([f"군집{c}" for c in range(K)], sizes, color=COLORS[:K], edgecolor="white", width=0.5)
for bar, s in zip(bars, sizes):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+30,
            f"{s:,}개\n({s/len(labels)*100:.1f}%)", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("질문 수"); ax.set_title(f"군집별 질문 수 (SBERT 의미 임베딩, k={K})")
ax.set_ylim(0, max(sizes)*1.2); ax.grid(axis="y", alpha=0.3)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout(); plt.savefig(CACHE / "sbert_cluster_size.png", dpi=150); plt.close()

# 6) 시각화: PCA 산점도
pca = PCA(n_components=2, random_state=42)
emb2d = pca.fit_transform(embeddings)
fig, ax = plt.subplots(figsize=(10, 7))
for c in range(K):
    m = labels == c
    ax.scatter(emb2d[m,0], emb2d[m,1], s=6, alpha=0.35, color=COLORS[c%len(COLORS)],
               label=f"군집{c} ({sizes[c]:,}개)")
c2d = pca.transform(km.cluster_centers_)
for c in range(K):
    ax.scatter(c2d[c,0], c2d[c,1], s=200, marker="*", color=COLORS[c%len(COLORS)],
               edgecolors="black", zorder=5)
    ax.annotate(f" 군집{c}", xy=(c2d[c,0], c2d[c,1]), fontsize=9, fontweight="bold")
vr = pca.explained_variance_ratio_
ax.set_xlabel(f"PC1 ({vr[0]*100:.1f}%)"); ax.set_ylabel(f"PC2 ({vr[1]*100:.1f}%)")
ax.set_title(f"자소서 질문 군집화 — SBERT → K-Means(k={K}) → PCA 2D")
ax.legend(fontsize=9, markerscale=3); ax.grid(True, alpha=0.2)
plt.tight_layout(); plt.savefig(CACHE / "sbert_pca_scatter.png", dpi=150); plt.close()

# 7) 실루엣 점수 (참고)
from sklearn.metrics import silhouette_score
sil = silhouette_score(embeddings, labels, sample_size=4000, random_state=42)
print(f"\n실루엣 점수(k={K}, SBERT): {sil:.4f}")
print("\nALL DONE")
