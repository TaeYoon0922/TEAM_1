from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.role02_star.head_first_classifier import classify_head_first
from models.role02_star.star_label_model import (
    STAR_CODES,
    STAR_MODEL_PATH,
    load_star_model,
    score_star_with_model,
)
from models.role02_star.star_scorer import score_star_completeness


SCHEMA_VERSION = "role02_to_role04.v1"
ROLE02_WEIGHTS = {
    "핵심 주장 명확성": 0.4,
    "경험 구체성": 0.6,
}


def build_role04_payload(
    answer: str,
    question: str = "",
    metadata: dict[str, Any] | None = None,
    star_model: Any | None = None,
) -> dict:
    """Return ROLE02 output in a ROLE04-compatible integration format."""
    metadata = metadata or {}
    head = classify_head_first(answer)
    star = score_experience(answer, star_model=star_model)

    core_metric = build_core_claim_metric(head)
    experience_metric = build_experience_metric(star)
    role02_score = calc_role02_score(core_metric["score"], experience_metric["score"])

    return {
        "schema_version": SCHEMA_VERSION,
        "source_role": "ROLE02",
        "target_role": "ROLE04",
        "question": question,
        "answer_id": metadata.get("answer_id") or metadata.get("row_no") or "",
        "metadata": {
            "url": metadata.get("url", ""),
            "company": metadata.get("company", ""),
            "season": metadata.get("season", ""),
            "question_no": metadata.get("question_no", ""),
        },
        "role02_summary": {
            "score": role02_score,
            "level": level_from_score(role02_score),
            "weights": ROLE02_WEIGHTS,
            "integration_rule": "0.4 * 핵심 주장 명확성 + 0.6 * 경험 구체성",
        },
        "role04_metrics": {
            "핵심 주장 명확성": core_metric,
            "경험 구체성": experience_metric,
        },
        "raw_outputs": {
            "core_claim": head,
            "experience_specificity": star,
        },
    }


def score_experience(answer: str, star_model: Any | None = None) -> dict:
    if star_model is not None:
        return score_star_with_model(answer, star_model)
    if STAR_MODEL_PATH.exists():
        return score_star_with_model(answer, load_star_model(STAR_MODEL_PATH))
    fallback = score_star_completeness(answer)
    fallback["scoring_note"] = "학습 모델 artifact가 없어 규칙 기반 STAR scorer를 사용했습니다."
    return fallback


def build_core_claim_metric(head: dict) -> dict:
    score = int(head.get("head_first_score", 0))
    return {
        "label": "핵심 주장 명확성",
        "score": score,
        "level": level_from_score(score),
        "feedback": head.get("reason", ""),
        "applicable": True,
        "details": {
            "display": head.get("display", "❌"),
            "is_clear": bool(head.get("is_head_first", 0)),
            "first_sentence": head.get("first_sentence", ""),
            "matched_keywords": head.get("matched_keywords", []),
            "bad_start_keywords": head.get("bad_start_keywords", []),
            "model_type": "rule_based_binary",
        },
    }


def build_experience_metric(star: dict) -> dict:
    checklist = {
        code: bool(star.get("checklist", {}).get(code, False))
        for code in STAR_CODES
    }
    score = int(star.get("score", 0))
    return {
        "label": "경험 구체성",
        "score": score,
        "level": level_from_score(score),
        "feedback": star.get("feedback", ""),
        "applicable": True,
        "details": {
            "display": star.get("display", ""),
            "checklist": checklist,
            "component_scores": star.get("component_scores", {}),
            "missing": star.get("missing", []),
            "evidence": star.get("evidence", {}),
            "model_type": "trained_multilabel_sentence_classifier",
            "scoring_note": star.get("scoring_note", ""),
        },
    }


def calc_role02_score(core_claim_score: int, experience_score: int) -> int:
    score = (
        core_claim_score * ROLE02_WEIGHTS["핵심 주장 명확성"]
        + experience_score * ROLE02_WEIGHTS["경험 구체성"]
    )
    return clamp(round(score))


def level_from_score(score: int) -> str:
    if score >= 80:
        return "우수"
    if score >= 60:
        return "보통"
    return "보완 필요"


def clamp(value: int, minimum: int = 0, maximum: int = 100) -> int:
    return max(minimum, min(maximum, value))


def main() -> None:
    train_path = PROJECT_ROOT / "data" / "processed" / "jobkorea_train.csv"
    with train_path.open(encoding="utf-8-sig", newline="") as f:
        row = next(row for row in csv.DictReader(f) if row.get("answer", "").strip())

    payload = build_role04_payload(
        answer=row["answer"],
        question=row.get("question", ""),
        metadata={
            "row_no": 1,
            "url": row.get("url", ""),
            "company": row.get("company", ""),
            "season": row.get("season", ""),
            "question_no": row.get("question_no", ""),
        },
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
