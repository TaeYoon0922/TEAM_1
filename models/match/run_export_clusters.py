import sys, os, pickle, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.modules.setdefault("sentence_transformers", None)
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from models.match.question_clusterer import load_questions, _CACHE_DIR

CACHE   = _CACHE_DIR
OUT_DIR = Path("models/match/clusters")
OUT_DIR.mkdir(parents=True, exist_ok=True)


questions  = load_questions()
with open(CACHE / "train_noun_kmeans_model.pkl", "rb") as f: km = pickle.load(f)
with open(CACHE / "train_noun_questions.pkl",    "rb") as f: noun_docs = pickle.load(f)

labels  = km.labels_
centers = km.cluster_centers_
k       = km.n_clusters

print(f"군집 수: {k},  총 질문: {len(questions)}")
print(f"출력 폴더: {OUT_DIR.resolve()}\n")


df = pd.DataFrame({
    "cluster_id": labels,
    "question":   questions,
    "noun_form":  noun_docs,
})


for c in range(k):
    sub = (df[df["cluster_id"] == c]
           .reset_index(drop=True))
    sub.index += 1

    out = OUT_DIR / f"cluster_{c}.csv"
    sub.to_csv(out, encoding="utf-8-sig", index_label="rank")
    print(f"  cluster_{c}.csv  ({len(sub)}개 질문)  → {out}")


all_out = OUT_DIR / "all_clusters.csv"
df_sorted = df.sort_values("cluster_id").reset_index(drop=True)
df_sorted.to_csv(all_out, encoding="utf-8-sig", index=False)
print(f"\n  all_clusters.csv  (전체 {len(df)}개)  → {all_out}")
print("\nDONE")
