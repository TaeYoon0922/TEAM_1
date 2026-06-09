import sys, pickle, io
import numpy as np
sys.modules.setdefault("sentence_transformers", None)
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

CACHE = Path("models/role05_match/cache")
embeddings = np.load(CACHE / "tfidf_lsa_embeddings.npy")
with open(CACHE / "kmeans_model.pkl", "rb") as f: km = pickle.load(f)
with open(CACHE / "unique_questions.pkl", "rb") as f: questions = pickle.load(f)

labels  = km.labels_
centers = km.cluster_centers_
TOP_N   = 20

for c in [2]:  # 군집 2만
    center = centers[c]
    # 메모리 절약: 행 단위로 거리 계산
    scored = []
    for i, (lbl, emb) in enumerate(zip(labels, embeddings)):
        if lbl != c:
            continue
        sim = float(np.dot(emb, center) / (np.linalg.norm(emb) * np.linalg.norm(center) + 1e-9))
        scored.append((sim, i))
    scored.sort(reverse=True)

    print(f"[군집 {c}]  ({len(scored)}개 질문)")
    for rank, (sim, idx) in enumerate(scored[:TOP_N], 1):
        q = questions[idx]
        print(f"  {rank}. {q[:130]}{'...' if len(q) > 130 else ''}")
    print()
