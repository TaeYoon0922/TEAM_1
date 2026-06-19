import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "match"))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

from relevance_detector import detect_answer_relevance
from question_clusterer import get_sbert_model

CSV = "data/labeled/match/match_labeled_300.csv"

df = pd.read_csv(CSV, encoding="utf-8-sig").dropna(subset=["question", "answer", "manual_label"])
print(f"라벨 데이터 {len(df)}행 / 라벨 분포:\n{df['manual_label'].value_counts().to_string()}\n")

is_off = (df["manual_label"].str.upper() == "OFF_TOPIC").to_numpy()


print("점수 계산 중(키워드 + SBERT 인코딩)...")
kw = np.array([detect_answer_relevance(q, a)["score"] for q, a in zip(df["question"], df["answer"])], dtype=float)

model = get_sbert_model()
q_emb = model.encode(df["question"].tolist(), batch_size=32, show_progress_bar=False)
a_emb = model.encode(df["answer"].tolist(), batch_size=32, show_progress_bar=False)
sem = np.array([float(cosine_similarity([q_emb[i]], [a_emb[i]])[0][0]) * 100 for i in range(len(df))])

print(f"  키워드 점수  평균: 온토픽 {kw[~is_off].mean():.1f} / OFF {kw[is_off].mean():.1f}")
print(f"  SBERT 코사인 평균: 온토픽 {sem[~is_off].mean():.1f} / OFF {sem[is_off].mean():.1f}\n")


print("=== 하이브리드 가중치 w (score = w*키워드 + (1-w)*SBERT) — 참고용 비교 ===")
W_DEPLOYED = 0.4
best = (None, -1)
for w in [round(x * 0.1, 1) for x in range(0, 11)]:
    h = w * kw + (1 - w) * sem
    auc = roc_auc_score(is_off, -h)
    mark = "  ← 배포값" if w == W_DEPLOYED else ""
    print(f"  w={w:.1f}  AUC(OFF탐지)={auc:.4f}{mark}")
    if auc > best[1]:
        best = (w, auc)
print(f"→ AUC 최댓값은 w={best[0]} 부근이지만, 배포 가중치(w={W_DEPLOYED})에서도 AUC={dict(zip([round(x*0.1,1) for x in range(11)], [roc_auc_score(is_off, -(w*kw+(1-w)*sem)) for w in [round(x*0.1,1) for x in range(11)]]))[W_DEPLOYED]:.4f}로 충분히 분리되므로, 가중치는 그대로 두고 임계값만 보정한다.\n")


hyb = W_DEPLOYED * kw + (1 - W_DEPLOYED) * sem
print(f"=== 게이트 임계값 T 보정 (배포 가중치 w={W_DEPLOYED} 고정, hybrid < T → OFF 차단) ===")
rows = []
for T in range(0, 101):
    pred_off = hyb < T
    TP = int((pred_off & is_off).sum()); FN = int((~pred_off & is_off).sum())
    TN = int((~pred_off & ~is_off).sum()); FP = int((pred_off & ~is_off).sum())
    sens = TP / (TP + FN) if TP + FN else 0
    spec = TN / (TN + FP) if TN + FP else 0
    prec = TP / (TP + FP) if TP + FP else 0
    f1 = 2 * prec * sens / (prec + sens) if prec + sens else 0
    rows.append((T, sens, spec, sens + spec - 1, f1))

best_j = max(r[3] for r in rows)
tied_j = [r[0] for r in rows if r[3] == best_j]
t_j_lo, t_j_hi = min(tied_j), max(tied_j)
print(f"  Youden's J 최댓값(={best_j:.3f})을 만족하는 T 구간: [{t_j_lo}, {t_j_hi}]  (이 구간 내 모든 T에서 OFF검출·온토픽통과 동률)")



off_max = float(hyb[is_off].max())
on_min = float(hyb[~is_off].min())
if off_max < on_min:
    t_margin = round((off_max + on_min) / 2)
    print(f"\n  [최대 마진 임계값] OFF_TOPIC 최댓값={off_max:.2f} < ON_TOPIC 최솟값={on_min:.2f}")
    print(f"  → 두 그룹이 완전히 분리되는 구간 [{off_max:.2f}, {on_min:.2f}]의 정중앙 T={t_margin}")
    print(f"     선정 근거: 양쪽 경계까지 거리(margin)가 동일(±{(on_min-off_max)/2:.1f}점)해 분포가 다소 흔들려도 가장 안정적 — SVM 결정경계와 동일한 원리(최대 마진 분리).")
    pred_off = hyb < t_margin
    TP = int((pred_off & is_off).sum()); FN = int((~pred_off & is_off).sum())
    TN = int((~pred_off & ~is_off).sum()); FP = int((pred_off & ~is_off).sum())
    print(f"     라벨 데이터 검증: OFF검출 {TP}/{TP+FN}={TP/(TP+FN):.0%}, 온토픽 오탐 {FP}/{FP+TN}={FP/(FP+TN):.0%}")
    print(f"\n→ 권장: GATE_THRESHOLD = {t_margin}  (가중치는 배포값 {W_DEPLOYED} 유지)")
else:
    print(f"\n  분포가 겹쳐 완전 분리 불가(OFF 최댓값 {off_max:.2f} ≥ ON 최솟값 {on_min:.2f}) — Youden's J 구간 [{t_j_lo}, {t_j_hi}]의 중앙값을 권장.")
    print(f"\n→ 권장: GATE_THRESHOLD = {(t_j_lo + t_j_hi)//2}  (가중치는 배포값 {W_DEPLOYED} 유지)")

print(f"\n(변경 전 코드값: GATE_THRESHOLD=20 — 동문서답 검출률 최적화보다 거짓양성(정상 답변 오차단) 방지에 치우친 보수값이었음)")
