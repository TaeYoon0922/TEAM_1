from pathlib import Path
import csv
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.role02_star.head_first_rules import analyze_head_first
from models.role02_star.head_first_classifier import classify_head_first
from models.role02_star.star_label_model import (
    LABELED_SENTENCE_PATH,
    STAR_DECISION_THRESHOLD,
    STAR_MODEL_PATH,
    load_labeled_sentence_rows,
    save_star_model,
    score_star_with_model,
    train_star_sentence_model,
)
from models.role02_star.visualize import (
    plot_role02_requirement_status,
    plot_role02_summary,
    plot_star_radar,
)


def print_table(rows, columns):
    widths = {
        col: max(len(str(col)), *(len(str(row.get(col, ""))) for row in rows))
        for col in columns
    }
    print(" | ".join(str(col).ljust(widths[col]) for col in columns))
    print("-+-".join("-" * widths[col] for col in columns))
    for row in rows:
        print(" | ".join(str(row.get(col, "")).ljust(widths[col]) for col in columns))


DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "jobkorea_train.csv"
DEMO_ANSWER_COUNT = 3
BATCH_SIZE = 1000
TRAIN_SCORE_PATH = PROJECT_ROOT / "data" / "processed" / "role02_train_scores.csv"
FIGURE_DIR = PROJECT_ROOT / "models" / "role02_star" / "figures"

labeled_rows = load_labeled_sentence_rows(LABELED_SENTENCE_PATH)
star_model = train_star_sentence_model(labeled_rows)
save_star_model(star_model, STAR_MODEL_PATH)

with DATASET_PATH.open(encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

answer_rows = [row for row in rows if row.get("answer", "").strip()]
demo_rows = answer_rows[:DEMO_ANSWER_COUNT]
answer_text = " ".join(row["answer"] for row in demo_rows)
head = analyze_head_first(answer_text)
head_binary = classify_head_first(answer_text)
star = score_star_with_model(answer_text, star_model)
star_radar_path = plot_star_radar(star, FIGURE_DIR / "role02_demo_star_radar.png")
summary_path = plot_role02_summary(head_binary, star, {}, FIGURE_DIR / "role02_demo_summary.png")
status_path = plot_role02_requirement_status(
    head_binary,
    star,
    {},
    FIGURE_DIR / "role02_demo_requirement_status.png",
)

print(f"[데이터셋] {DATASET_PATH.relative_to(PROJECT_ROOT)}")
print(f"[STAR 라벨셋] {LABELED_SENTENCE_PATH.relative_to(PROJECT_ROOT)}")
print(f"STAR 학습 문장 수: {len(labeled_rows)}")
print(f"STAR decision threshold: {STAR_DECISION_THRESHOLD}")
print(f"STAR 모델: {STAR_MODEL_PATH.relative_to(PROJECT_ROOT)}")
print(f"전체 행 수: {len(rows)}")
print(f"답변 있는 행 수: {len(answer_rows)}")
print(f"분석 대상 답변 수: {len(demo_rows)}")
print(f"STAR radar: {star_radar_path.relative_to(PROJECT_ROOT)}")
print(f"분석 요약 그래프: {summary_path.relative_to(PROJECT_ROOT)}")
print(f"모델 충족 여부 그래프: {status_path.relative_to(PROJECT_ROOT)}")

print("\n[분석 요약]")
print_table(
    [
        {
            "지표": "핵심 주장 명확성",
            "결과": head["display"],
            "점수": head["head_first_score"],
            "근거/피드백": head["reason"],
        },
        {
            "지표": "경험 구체성",
            "결과": star["display"],
            "점수": star["score"],
            "근거/피드백": star["feedback"],
        },
    ],
    ["지표", "결과", "점수", "근거/피드백"],
)

print("\n[STAR 체크리스트]")
print_table(
    [
        {
            "요소": code,
            "충족": "✅" if star["checklist"][code] else "❌",
            "근거 문장": " / ".join(star["evidence"][code]) or "-",
        }
        for code in ["S", "T", "A", "R"]
    ],
    ["요소", "충족", "근거 문장"],
)

print("\n[답변별 데이터셋 확인]")
print_table(
    [
        {
            "No": index,
            "기업": row["company"],
            "문항번호": row["question_no"],
            "답변": row["answer"].replace("\n", " ")[:120],
        }
        for index, row in enumerate(demo_rows, start=1)
    ],
    ["No", "기업", "문항번호", "답변"],
)


def score_answer_row(row, index):
    answer = row.get("answer", "")
    head_result = classify_head_first(answer)
    star_result = score_star_with_model(answer, star_model)

    return {
        "row_no": index,
        "url": row.get("url", ""),
        "company": row.get("company", ""),
        "season": row.get("season", ""),
        "question_no": row.get("question_no", ""),
        "question": row.get("question", ""),
        "core_claim_display": head_result["display"],
        "core_claim_score": head_result["head_first_score"],
        "core_claim_reason": head_result["reason"],
        "experience_display": star_result["display"],
        "experience_score": star_result["score"],
        "experience_feedback": star_result["feedback"],
        "S": int(star_result["checklist"]["S"]),
        "T": int(star_result["checklist"]["T"]),
        "A": int(star_result["checklist"]["A"]),
        "R": int(star_result["checklist"]["R"]),
        "evidence_S": "|".join(star_result["evidence"]["S"]),
        "evidence_T": "|".join(star_result["evidence"]["T"]),
        "evidence_A": "|".join(star_result["evidence"]["A"]),
        "evidence_R": "|".join(star_result["evidence"]["R"]),
    }


def run_train_batch_scoring(answer_rows):
    TRAIN_SCORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row_no",
        "url",
        "company",
        "season",
        "question_no",
        "question",
        "core_claim_display",
        "core_claim_score",
        "core_claim_reason",
        "experience_display",
        "experience_score",
        "experience_feedback",
        "S",
        "T",
        "A",
        "R",
        "evidence_S",
        "evidence_T",
        "evidence_A",
        "evidence_R",
    ]

    print(f"\n[train 1000개 단위 처리]")
    print(f"batch size: {BATCH_SIZE}")
    print(f"대상 답변 수: {len(answer_rows)}")
    print(f"저장 경로: {TRAIN_SCORE_PATH.relative_to(PROJECT_ROOT)}")

    with TRAIN_SCORE_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for batch_start in range(0, len(answer_rows), BATCH_SIZE):
            batch = answer_rows[batch_start:batch_start + BATCH_SIZE]
            for offset, row in enumerate(batch, start=1):
                writer.writerow(score_answer_row(row, batch_start + offset))

            processed = min(batch_start + len(batch), len(answer_rows))
            print(f"  {processed}/{len(answer_rows)} 완료")

    print("train scoring 완료")


run_train_batch_scoring(answer_rows)
