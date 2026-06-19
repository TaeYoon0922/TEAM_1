import sys, os, csv
sys.path.insert(0, os.path.dirname(__file__))
from self_detector import detect_self_language




GROUND_TRUTH = {
    0: "C",
    1: "C",
    2: "D",
    3: "D",
    4: "D",
    5: "C",
    6: "D",
    7: "D",
    8: "D",
    9: "C",
    10: "B",
    11: "C",
    12: "D",
    13: "D",
    14: "B",
    15: "D",
    16: "C",
    17: "C",
    18: "D",
    19: "B",
    20: "B",
    21: "D",
    22: "B",
    23: "D",
    24: "D",
    25: "C",
    26: "B",
    27: "B",
    28: "B",
    29: "C",
    30: "C",
    31: "D",
    32: "D",
    33: "D",
    34: "C",
    35: "C",
    36: "B",
    37: "C",
    38: "D",
    39: "B",
    40: "B",
    41: "C",
    42: "C",
    43: "C",
    44: "D",
    45: "C",
    46: "B",
    47: "D",
    48: "B",
    49: "D",
    50: "B",
    51: "D",
    52: "D",
    53: "D",
    54: "C",
    55: "D",
    56: "C",
    57: "D",
    58: "B",
    59: "D",
    60: "D",
    61: "D",
    62: "D",
    63: "D",
    64: "D",
    65: "B",
    66: "B",
    67: "B",
    68: "B",
    69: "C",
    70: "D",
    71: "B",
    72: "D",
    73: "D",
    74: "D",
    75: "C",
    76: "D",
    77: "C",
    78: "B",
    79: "B",
    80: "D",
    81: "A",
    82: "C",
    83: "D",
    84: "C",
    85: "B",
    86: "D",
    87: "B",
    88: "D",
    89: "D",
    90: "B",
    91: "D",
    92: "D",
    93: "D",
    94: "D",
    95: "B",
    96: "C",
    97: "B",
    98: "B",
    99: "B",
}


DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "labeled", "self", "sampled_100.csv"
)

def load_data():
    rows = {}
    with open(DATA_PATH, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["idx"])
            rows[idx] = row["answer"]
    return rows


def compute_metrics(true_labels, pred_labels, target_class):
    tp = sum(t == target_class and p == target_class for t, p in zip(true_labels, pred_labels))
    fp = sum(t != target_class and p == target_class for t, p in zip(true_labels, pred_labels))
    fn = sum(t == target_class and p != target_class for t, p in zip(true_labels, pred_labels))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1, tp, fp, fn


def main():
    data = load_data()
    true_labels, pred_labels, details = [], [], []

    print("샘플 예측 중...")
    for idx in sorted(GROUND_TRUTH.keys()):
        text = data.get(idx, "")
        if not text:
            print(f"  [WARN] idx={idx} 텍스트 없음")
            continue
        result = detect_self_language(text)
        pred = result["grade"]
        true = GROUND_TRUTH[idx]
        true_labels.append(true)
        pred_labels.append(pred)
        details.append((idx, true, pred, result["self_score"], result["contribution_score"]))

    total   = len(true_labels)
    correct = sum(t == p for t, p in zip(true_labels, pred_labels))

    print("\n" + "=" * 66)
    print("ROLE 04  자기중심 탐지기  성능 평가  (sampled_100.csv)")
    print(f"테스트셋: {total}개 / 정답: 사람이 직접 라벨링 (A/B/C/D)")
    print("=" * 66)

    print(f"\n[전체 정확도]  {correct}/{total} = {correct/total:.1%}\n")

    classes = ["A", "B", "C", "D"]
    class_label = {"A": "기여 뚜렷", "B": "기여 우세", "C": "혼  재", "D": "자기중심"}

    print(f"{'클래스':<6} {'의미':<10} {'N(정답)':>8} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("-" * 56)
    macro_f1_sum = 0
    for cls in classes:
        p, r, f1, tp, fp, fn = compute_metrics(true_labels, pred_labels, cls)
        n = sum(t == cls for t in true_labels)
        macro_f1_sum += f1
        print(f"  {cls}    {class_label[cls]:<10} {n:>8}   {p:>9.3f}  {r:>7.3f}  {f1:>7.3f}   TP={tp} FP={fp} FN={fn}")

    macro_f1 = macro_f1_sum / len(classes)
    print("-" * 56)
    print(f"  Macro F1 (A/B/C/D)                              {macro_f1:>7.3f}")


    print("\n[정답 라벨 분포]")
    for cls in classes:
        n = sum(t == cls for t in true_labels)
        print(f"  {cls}: {n}개")


    print("\n[예측 라벨 분포]")
    for cls in classes:
        n = sum(p == cls for p in pred_labels)
        print(f"  {cls}: {n}개")


    errors = [(idx, t, p, ss, cs) for idx, t, p, ss, cs in details if t != p]
    print(f"\n[오분류 케이스] {len(errors)}건")
    print(f"{'idx':>4}  {'정답':>4}  {'예측':>4}  {'self':>5}  {'contrib':>7}")
    print("-" * 32)
    for idx, t, p, ss, cs in errors:
        print(f"{idx:>4}  {t:>4}  {p:>4}  {ss:>5}  {cs:>7}")

if __name__ == "__main__":
    main()
