"""
Step 2~5 실행 스크립트
현재 환경에서 SBERT가 segfault를 내므로 TF-IDF+LSA 방식으로 실행.
SBERT가 정상 동작하는 환경에서는 question_clusterer.get_embeddings()로 교체 가능.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# sentence_transformers import 자체를 차단해 segfault 방지
import sys
sys.modules["sentence_transformers"] = None  # type: ignore

from models.role05_match.question_clusterer import (
    load_questions,
    get_tfidf_lsa_embeddings,
    plot_elbow,
    fit_kmeans,
    print_cluster_representatives,
    save_cluster_map,
)

# Step 1
questions = load_questions()
print(f"질문 수: {len(questions)}")

# Step 2: TF-IDF char n-gram (2-4) + SVD(LSA 128차원)
embeddings = get_tfidf_lsa_embeddings(questions)
print(f"임베딩 shape: {embeddings.shape}")

# Step 4: Elbow Method
optimal_k = plot_elbow(
    embeddings,
    save_path="models/role05_match/cache/elbow_curve.png",
)
print(f"최적 k: {optimal_k}")

# Step 5: K-Means + 대표 질문 출력
km = fit_kmeans(embeddings, k=optimal_k)
cluster_info = print_cluster_representatives(km, questions, embeddings, top_n=5)

# cluster_map.json 초안 저장
save_cluster_map(cluster_info, k=optimal_k)
print("\nALL DONE")
