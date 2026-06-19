from __future__ import annotations

import os
import pickle
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DIR = Path(__file__).resolve().parent
CACHE_DIR = _DIR / "cache"
FIGURE_DIR = _DIR / "figures"
CLUSTER_MAP_PATH = _DIR / "cluster_map.json"

EMBED_PATH = CACHE_DIR / "train_sbert_embeddings.npy"
KMEANS_PATH = CACHE_DIR / "train_kmeans_model.pkl"

SILHOUETTE_FIGURE_PATH = FIGURE_DIR / "match_cluster_silhouette_by_k.png"
PCA_SCATTER_FIGURE_PATH = FIGURE_DIR / "match_cluster_pca_scatter.png"
CLUSTER_SIZE_FIGURE_PATH = FIGURE_DIR / "match_cluster_sizes.png"

K_RANGE = range(2, 11)
CHOSEN_K = 6
SAMPLE_SIZE = 3000
RANDOM_STATE = 42

CLUSTER_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]


def main() -> None:
    embeddings = np.load(EMBED_PATH)
    with open(KMEANS_PATH, "rb") as f:
        km = pickle.load(f)
    labels = km.labels_
    cluster_map = load_cluster_map()

    print(f"[MATCH 군집화 평가] 임베딩 shape={embeddings.shape}, k={km.n_clusters}, 표본 수={len(labels)}")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nk별 실루엣 점수 계산 중 (k={list(K_RANGE)}, 평가 표본={SAMPLE_SIZE}개, seed={RANDOM_STATE})...")
    scores = compute_silhouette_by_k(embeddings, cached_label_for_k={CHOSEN_K: labels})
    for k, score in scores.items():
        marker = "  <- 채택" if k == CHOSEN_K else ""
        print(f"  k={k:2d}  실루엣 점수={score:.4f}{marker}")
    plot_silhouette_by_k(scores, SILHOUETTE_FIGURE_PATH)

    plot_pca_scatter(embeddings, labels, cluster_map, PCA_SCATTER_FIGURE_PATH)
    plot_cluster_sizes(labels, cluster_map, CLUSTER_SIZE_FIGURE_PATH)

    print(f"\n실루엣 점수(k 비교) 그림: {SILHOUETTE_FIGURE_PATH.relative_to(PROJECT_ROOT)}")
    print(f"PCA 2D 산점도 그림: {PCA_SCATTER_FIGURE_PATH.relative_to(PROJECT_ROOT)}")
    print(f"군집 크기 분포 그림: {CLUSTER_SIZE_FIGURE_PATH.relative_to(PROJECT_ROOT)}")


def load_cluster_map() -> dict:
    if CLUSTER_MAP_PATH.exists():
        with open(CLUSTER_MAP_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def cluster_label(cluster_id: int, cluster_map: dict) -> str:
    name = cluster_map.get(f"cluster_{cluster_id}", {}).get("name")
    return f"{cluster_id}: {name}" if name else f"군집 {cluster_id}"


def compute_silhouette_by_k(embeddings: np.ndarray, cached_label_for_k: dict[int, np.ndarray]) -> dict[int, float]:
    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = rng.choice(len(embeddings), size=min(SAMPLE_SIZE, len(embeddings)), replace=False)
    sample = embeddings[sample_idx]

    scores: dict[int, float] = {}
    for k in K_RANGE:
        if k in cached_label_for_k:
            labels = cached_label_for_k[k]
        else:
            km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
            labels = km.fit_predict(embeddings)
        scores[k] = float(silhouette_score(sample, labels[sample_idx]))
    return scores


def plot_silhouette_by_k(scores: dict[int, float], path: Path) -> None:
    has_korean_font = set_korean_font()
    ks = list(scores.keys())
    values = [scores[k] for k in ks]
    colors = ["#C44E52" if k == CHOSEN_K else "#4C72B0" for k in ks]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar([str(k) for k in ks], values, color=colors, width=0.6)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}", ha="center", fontsize=9)

    ax.set_xlabel("군집 수 (k)" if has_korean_font else "number of clusters (k)")
    ax.set_ylabel("실루엣 점수" if has_korean_font else "silhouette score")
    title = (f"군집 수(k) 별 실루엣 점수 — k={CHOSEN_K} 채택 근거 (표본 {SAMPLE_SIZE}개)"
             if has_korean_font else f"Silhouette score by k — why k={CHOSEN_K} (n={SAMPLE_SIZE} sample)")
    ax.set_title(title)
    ax.set_ylim(0, max(values) * 1.25)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_pca_scatter(embeddings: np.ndarray, labels: np.ndarray, cluster_map: dict, path: Path) -> None:
    has_korean_font = set_korean_font()
    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = rng.choice(len(embeddings), size=min(SAMPLE_SIZE, len(embeddings)), replace=False)

    coords_2d = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(embeddings[sample_idx])
    sample_labels = labels[sample_idx]

    fig, ax = plt.subplots(figsize=(8, 6))
    for cluster_id in sorted(np.unique(sample_labels)):
        mask = sample_labels == cluster_id
        label = cluster_label(int(cluster_id), cluster_map) if has_korean_font else f"cluster {cluster_id}"
        ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1], s=10, alpha=0.5,
                   color=CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)], label=label)

    title = (f"SBERT 임베딩 PCA 2D 산점도 — 군집별 분리 (표본 {SAMPLE_SIZE}개)"
             if has_korean_font else f"SBERT embeddings, PCA 2D — cluster separation (n={SAMPLE_SIZE} sample)")
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=8, markerscale=2, loc="best")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_cluster_sizes(labels: np.ndarray, cluster_map: dict, path: Path) -> None:
    has_korean_font = set_korean_font()
    cluster_ids = sorted(np.unique(labels))
    counts = [int((labels == c).sum()) for c in cluster_ids]
    labels_text = [cluster_label(c, cluster_map) if has_korean_font else f"cluster {c}" for c in cluster_ids]
    colors = [CLUSTER_COLORS[c % len(CLUSTER_COLORS)] for c in cluster_ids]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(labels_text, counts, color=colors)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{count:,}", va="center", fontsize=9)

    ax.invert_yaxis()
    ax.set_xlabel("질문 수" if has_korean_font else "number of questions")
    title = "군집별 표본 수 분포 (전체 {:,}개)".format(len(labels)) if has_korean_font \
        else f"Cluster size distribution (n={len(labels):,})"
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def set_korean_font() -> bool:
    candidates = ["Malgun Gothic", "NanumGothic", "AppleGothic", "DejaVu Sans"]
    available = {font.name for font in fm.fontManager.ttflist}
    for font in candidates:
        if font in available and font != "DejaVu Sans":
            matplotlib.rcParams["font.family"] = font
            matplotlib.rcParams["axes.unicode_minus"] = False
            return True
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    matplotlib.rcParams["axes.unicode_minus"] = False
    return False


if __name__ == "__main__":
    main()
