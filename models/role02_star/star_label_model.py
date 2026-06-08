from __future__ import annotations

import csv
import re
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline


STAR_CODES = ["S", "T", "A", "R"]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LABELED_SENTENCE_PATH = PROJECT_ROOT / "data" / "labeled" / "jobkorea_star_full_labeled_sentence_dataset.csv"
ARTIFACT_DIR = PROJECT_ROOT / "models" / "role02_star" / "artifacts"
STAR_MODEL_PATH = ARTIFACT_DIR / "star_sentence_classifier.joblib"
STAR_DECISION_THRESHOLD = 0.92


def load_labeled_sentence_rows(path: Path = LABELED_SENTENCE_PATH) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [
            row
            for row in csv.DictReader(f)
            if row.get("sentence", "").strip() and row.get("star_label", "").strip()
        ]


def split_by_url(rows: list[dict], test_ratio: float = 0.2, seed: int = 42) -> tuple[list[dict], list[dict]]:
    urls = sorted({row["url"] for row in rows if row.get("url")})
    rng = np.random.default_rng(seed)
    shuffled_urls = urls[:]
    rng.shuffle(shuffled_urls)
    test_size = max(1, int(len(shuffled_urls) * test_ratio))
    test_urls = set(shuffled_urls[:test_size])

    train_rows = [row for row in rows if row.get("url") not in test_urls]
    test_rows = [row for row in rows if row.get("url") in test_urls]
    return train_rows, test_rows


def parse_star_label(label: str) -> dict[str, int]:
    parts = {part.strip() for part in str(label).split("+") if part.strip()}
    return {code: int(code in parts) for code in STAR_CODES}


def rows_to_xy(rows: list[dict]) -> tuple[list[str], np.ndarray]:
    texts = [normalize_text(row.get("sentence", "")) for row in rows]
    y = np.array(
        [[parse_star_label(row.get("star_label", "")).get(code, 0) for code in STAR_CODES] for row in rows],
        dtype=int,
    )
    return texts, y


def build_star_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 5),
                    min_df=3,
                    max_features=80000,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                OneVsRestClassifier(
                    SGDClassifier(
                        loss="log_loss",
                        class_weight="balanced",
                        alpha=1e-5,
                        max_iter=30,
                        tol=1e-3,
                        random_state=42,
                    )
                ),
            ),
        ]
    )


def train_star_sentence_model(rows: list[dict] | None = None) -> Pipeline:
    rows = rows if rows is not None else load_labeled_sentence_rows()
    texts, labels = rows_to_xy(rows)
    model = build_star_pipeline()
    model.fit(texts, labels)
    return model


def save_star_model(model: Pipeline, path: Path = STAR_MODEL_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_star_model(path: Path = STAR_MODEL_PATH) -> Pipeline:
    return joblib.load(path)


def predict_sentence_labels(model: Pipeline, sentences: list[str], threshold: float = STAR_DECISION_THRESHOLD) -> list[dict]:
    texts = [normalize_text(sentence) for sentence in sentences]
    if not texts:
        return []

    probabilities = model.predict_proba(texts)
    predictions = (probabilities >= threshold).astype(int)
    results = []
    for sentence, pred_row, prob_row in zip(sentences, predictions, probabilities):
        checklist = {code: bool(pred_row[index]) for index, code in enumerate(STAR_CODES)}
        confidence = {code: round(float(prob_row[index]), 4) for index, code in enumerate(STAR_CODES)}
        results.append(
            {
                "sentence": sentence,
                "checklist": checklist,
                "confidence": confidence,
                "decision_threshold": threshold,
                "display": format_checklist(checklist),
            }
        )
    return results


def score_star_with_model(text: str, model: Pipeline, threshold: float = STAR_DECISION_THRESHOLD) -> dict:
    sentence_predictions = predict_sentence_labels(model, split_sentences(text), threshold=threshold)
    checklist = {
        code: any(prediction["checklist"].get(code, False) for prediction in sentence_predictions)
        for code in STAR_CODES
    }
    evidence = {
        code: [
            prediction["sentence"]
            for prediction in sentence_predictions
            if prediction["checklist"].get(code, False)
        ][:3]
        for code in STAR_CODES
    }
    component_scores = {code: 25 if checklist[code] else 0 for code in STAR_CODES}
    score = sum(component_scores.values())
    missing = [code for code, exists in checklist.items() if not exists]

    return {
        "metric": "경험 구체성",
        "score": score,
        "grade": grade_from_score(score),
        "checklist": checklist,
        "display": format_checklist(checklist),
        "missing": missing,
        "evidence": evidence,
        "component_scores": component_scores,
        "max_score": 100,
        "decision_threshold": threshold,
        "sentence_predictions": sentence_predictions,
        "scoring_note": f"라벨링된 STAR 문장 데이터셋으로 학습한 multi-label 모델 예측입니다. threshold={threshold}",
        "feedback": build_feedback(checklist, missing),
    }


def split_sentences(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    sentences = re.split(r"(?<=[.!?。！？])\s+|\n+", normalized)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def format_checklist(checklist: dict) -> str:
    return " ".join(f"{code}{'✅' if checklist.get(code) else '❌'}" for code in STAR_CODES)


def grade_from_score(score: int) -> str:
    if score >= 100:
        return "우수"
    if score >= 75:
        return "양호"
    if score >= 50:
        return "보통"
    return "보완 필요"


def build_feedback(checklist: dict, missing: list[str]) -> str:
    display = format_checklist(checklist)
    if not missing:
        return f"{display} S/T/A/R 요소가 모두 확인됩니다."
    return f"{display} {', '.join(missing)} 요소가 약합니다. 빠진 요소를 한두 문장으로 보강하세요."
