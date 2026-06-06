"""
문항 적합성 게이트 임계값/하이브리드 가중치 데이터 기반 보정.

라벨 데이터(role05_match_dummy_300.csv, manual_label: EXCELLENT/MEDIUM/OFF_TOPIC)에서
- 하이브리드 가중치 w (score = w*keyword + (1-w)*SBERT코사인) 를 AUC로 선택
- 게이트 임계값 T 를 Youden's J / F1 로 선택
하여, OFF_TOPIC을 가장 잘 가르는 값을 출력한다.

근거 방법론:
- SBERT 코사인 의미 유사도 (Reimers & Gurevych, Sentence-BERT, EMNLP 2019)
- 임계값은 라벨 데이터의 ROC 기반 Youden's J / F1 최적화로 결정(표준 컷오프 선택법)
"""
import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "role05_match"))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

from relevance_detector import detect_answer_relevance
from question_clusterer import get_sbert_model

CSV = "data/labeled/role05_match/role05_match_dummy_300.csv"

df = pd.read_csv(CSV, encoding="utf-8-sig").dropna(subset=["question", "answer", "manual_label"])
print(f"라벨 데이터 {len(df)}행 / 라벨 분포:\n{df['manual_label'].value_counts().to_string()}\n")

is_off = (df["manual_label"].str.upper() == "OFF_TOPIC").to_numpy()

# 1) 점수 계산: 키워드 + SBERT 코사인
print("점수 계산 중(키워드 + SBERT 인코딩)...")
kw = np.array([detect_answer_relevance(q, a)["score"] for q, a in zip(df["question"], df["answer"])], dtype=float)

model = get_sbert_model()
q_emb = model.encode(df["question"].tolist(), batch_size=32, show_progress_bar=False)
a_emb = model.encode(df["answer"].tolist(), batch_size=32, show_progress_bar=False)
sem = np.array([float(cosine_similarity([q_emb[i]], [a_emb[i]])[0][0]) * 100 for i in range(len(df))])

print(f"  키워드 점수  평균: 온토픽 {kw[~is_off].mean():.1f} / OFF {kw[is_off].mean():.1f}")
print(f"  SBERT 코사인 평균: 온토픽 {sem[~is_off].mean():.1f} / OFF {sem[is_off].mean():.1f}\n")

# 2) 하이브리드 가중치 w 선택 (OFF_TOPIC 탐지 AUC 최대)
print("=== 하이브리드 가중치 w (score = w*키워드 + (1-w)*SBERT) ===")
best = (None, -1)
for w in [round(x * 0.1, 1) for x in range(0, 11)]:
    h = w * kw + (1 - w) * sem
    auc = roc_auc_score(is_off, -h)  # OFF가 양성, 점수 낮을수록 OFF
    print(f"  w={w:.1f}  AUC(OFF탐지)={auc:.4f}")
    if auc > best[1]:
        best = (w, auc)
w_opt = best[0]
print(f"→ 최적 가중치 w={w_opt} (키워드 {w_opt:.0%} / SBERT {1-w_opt:.0%}), AUC={best[1]:.4f}\n")

# 3) 게이트 임계값 T 선택 (Youden's J, F1)
hyb = w_opt * kw + (1 - w_opt) * sem
print("=== 게이트 임계값 T (hybrid < T → OFF로 차단) ===")
rows = []
for T in range(0, 101):
    pred_off = hyb < T
    TP = int((pred_off & is_off).sum()); FN = int((~pred_off & is_off).sum())
    TN = int((~pred_off & ~is_off).sum()); FP = int((pred_off & ~is_off).sum())
    sens = TP / (TP + FN) if TP + FN else 0     # OFF 검출률(재현율)
    spec = TN / (TN + FP) if TN + FP else 0     # 온토픽 통과율
    prec = TP / (TP + FP) if TP + FP else 0
    f1 = 2 * prec * sens / (prec + sens) if prec + sens else 0
    rows.append((T, sens, spec, sens + spec - 1, f1))

t_j = max(rows, key=lambda r: r[3])
t_f1 = max(rows, key=lambda r: r[4])
print(f"  Youden's J 최적 T={t_j[0]:>3}  (OFF검출 {t_j[1]:.2f} / 온토픽통과 {t_j[2]:.2f} / J={t_j[3]:.3f} / F1={t_j[4]:.3f})")
print(f"  F1        최적 T={t_f1[0]:>3}  (OFF검출 {t_f1[1]:.2f} / 온토픽통과 {t_f1[2]:.2f} / J={t_f1[3]:.3f} / F1={t_f1[4]:.3f})")
print(f"\n현재 코드값: w=0.4, GATE_THRESHOLD=20 (참고)")
