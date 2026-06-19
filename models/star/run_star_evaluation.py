from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.star.star_label_model import (
    LABELED_SENTENCE_PATH,
    STAR_CODES,
    STAR_DECISION_THRESHOLD,
    STAR_MODEL_PATH,
    load_labeled_sentence_rows,
    parse_star_label,
    predict_sentence_labels,
    save_star_model,
    split_by_url,
    train_star_sentence_model,
)

PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "star_eval_predictions.csv"
METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "star_eval_metrics.csv"
FIGURE_DIR = PROJECT_ROOT / "models" / "star" / "figures"
METRICS_FIGURE_PATH = FIGURE_DIR / "star_precision_recall_f1.png"
CONFUSION_FIGURE_PATH = FIGURE_DIR / "star_confusion_matrices.png"
THRESHOLD_SUFFIX = str(int(round(STAR_DECISION_THRESHOLD * 100))).zfill(2)
METRICS_THRESHOLD_FIGURE_PATH = FIGURE_DIR / f"star_precision_recall_f1_threshold_{THRESHOLD_SUFFIX}.png"
CONFUSION_THRESHOLD_FIGURE_PATH = FIGURE_DIR / f"star_confusion_matrices_threshold_{THRESHOLD_SUFFIX}.png"


def main() -> None:
    rows = load_labeled_sentence_rows(LABELED_SENTENCE_PATH)
    train_rows, test_rows = split_by_url(rows)
    model = train_star_sentence_model(train_rows)
    save_star_model(model, STAR_MODEL_PATH)

    prediction_rows = predict_rows(model, test_rows)
    metric_rows = evaluate_predictions(prediction_rows)

    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    write_csv(PREDICTIONS_PATH, prediction_rows)
    write_csv(METRICS_PATH, metric_rows)
    plot_metric_bars(metric_rows, METRICS_FIGURE_PATH)
    plot_confusion_matrices(metric_rows, CONFUSION_FIGURE_PATH)
    plot_metric_bars(metric_rows, METRICS_THRESHOLD_FIGURE_PATH)
    plot_confusion_matrices(metric_rows, CONFUSION_THRESHOLD_FIGURE_PATH)

    train_urls = {row["url"] for row in train_rows}
    test_urls = {row["url"] for row in test_rows}
    print(f"[STAR 평가] 라벨 CSV: {LABELED_SENTENCE_PATH.relative_to(PROJECT_ROOT)}")
    print(f"train 문장 수: {len(train_rows)} / test 문장 수: {len(test_rows)}")
    print(f"train URL 수: {len(train_urls)} / test URL 수: {len(test_urls)} / URL overlap: {len(train_urls & test_urls)}")
    print(f"학습 모델: {STAR_MODEL_PATH.relative_to(PROJECT_ROOT)}")
    print(f"예측 결과: {PREDICTIONS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"평가 지표: {METRICS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"PRF 시각화: {METRICS_FIGURE_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Confusion matrix: {CONFUSION_FIGURE_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Threshold PRF 시각화: {METRICS_THRESHOLD_FIGURE_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Threshold confusion matrix: {CONFUSION_THRESHOLD_FIGURE_PATH.relative_to(PROJECT_ROOT)}")
    print()
    print_table(metric_rows)


def predict_rows(model, rows: list[dict]) -> list[dict]:
    sentence_predictions = predict_sentence_labels(model, [row["sentence"] for row in rows])
    return [build_prediction_row(row, prediction) for row, prediction in zip(rows, sentence_predictions)]


def build_prediction_row(row: dict, prediction: dict) -> dict:
    true_labels = parse_star_label(row.get("star_label", ""))

    output = {
        "sample_id": row.get("sample_id"),
        "source_row_id": row.get("source_row_id"),
        "sentence_no": row.get("sentence_no"),
        "url": row.get("url"),
        "company": row.get("company"),
        "season": row.get("season"),
        "question_no": row.get("question_no"),
        "question": row.get("question"),
        "sentence": row.get("sentence"),
        "star_label": row.get("star_label"),
        "pred_display": prediction["display"],
        "label_source": row.get("label_source"),
        "confidence": row.get("confidence"),
    }

    for code in STAR_CODES:
        output[f"true_{code}"] = true_labels[code]
        output[f"pred_{code}"] = int(bool(prediction["checklist"].get(code)))
        output[f"prob_{code}"] = prediction["confidence"][code]

    return output


def evaluate_predictions(rows: list[dict]) -> list[dict]:
    metric_rows = []
    for key in STAR_CODES:
        true_key = f"true_{key}"
        pred_key = f"pred_{key}"
        y_true = [int(row[true_key]) for row in rows]
        y_pred = [int(row[pred_key]) for row in rows]
        metric_rows.append(calc_binary_metrics(f"경험 구체성_{key}", y_true, y_pred))

    return metric_rows


def calc_binary_metrics(metric_name: str, y_true: list[int], y_pred: list[int]) -> dict:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    accuracy = safe_div(tp + tn, len(y_true))

    return {
        "metric": metric_name,
        "support_positive": sum(y_true),
        "support_negative": len(y_true) - sum(y_true),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
    }


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_metric_bars(metric_rows: list[dict], path: Path) -> None:
    has_korean_font = set_korean_font()
    labels = [row["metric"] for row in metric_rows]
    if not has_korean_font:
        labels = ["Experience_S", "Experience_T", "Experience_A", "Experience_R"]
    x = np.arange(len(labels))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10, 5))
    for offset, key, color in [
        (-width, "precision", "#4C72B0"),
        (0, "recall", "#55A868"),
        (width, "f1", "#C44E52"),
    ]:
        values = [row[key] for row in metric_rows]
        bars = ax.bar(x + offset, values, width, label=key, color=color)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.2f}", ha="center", fontsize=8)

    ax.set_ylim(0, 1.1)
    ax.set_ylabel("score")
    ax.set_title("STAR Precision / Recall / F1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_confusion_matrices(metric_rows: list[dict], path: Path) -> None:
    has_korean_font = set_korean_font()
    cols = 2
    rows = 2
    fig, axes = plt.subplots(rows, cols, figsize=(10, 7))
    axes_flat = axes.flatten()

    for ax, metric in zip(axes_flat, metric_rows):
        matrix = np.array([[metric["tn"], metric["fp"]], [metric["fn"], metric["tp"]]])
        image = ax.imshow(matrix, cmap="Blues")
        title = metric["metric"]
        if not has_korean_font:
            title = {
                "경험 구체성_S": "Experience_S",
                "경험 구체성_T": "Experience_T",
                "경험 구체성_A": "Experience_A",
                "경험 구체성_R": "Experience_R",
            }.get(title, title)
        ax.set_title(title)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticklabels(["True 0", "True 1"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color="#111111")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes_flat[len(metric_rows):]:
        ax.axis("off")

    plt.suptitle("STAR Confusion Matrices", fontsize=14)
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


def print_table(rows: list[dict]) -> None:
    columns = ["metric", "precision", "recall", "f1", "accuracy", "tp", "fp", "fn", "tn"]
    widths = {
        column: max(len(column), *(len(str(row[column])) for row in rows))
        for column in columns
    }
    print(" | ".join(column.ljust(widths[column]) for column in columns))
    print("-+-".join("-" * widths[column] for column in columns))
    for row in rows:
        print(" | ".join(str(row[column]).ljust(widths[column]) for column in columns))


if __name__ == "__main__":
    main()
