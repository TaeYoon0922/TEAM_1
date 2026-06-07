from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np


STAR_CODES = ["S", "T", "A", "R"]


def plot_star_radar(star_result: dict, output_path: str | Path) -> Path:
    """Save a radar chart for S/T/A/R checklist coverage."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    has_korean_font = set_korean_font()

    values = [1 if star_result["checklist"].get(code) else 0 for code in STAR_CODES]
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(STAR_CODES), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"polar": True})
    ax.plot(angles, values, color="#4C72B0", linewidth=2)
    ax.fill(angles, values, color="#4C72B0", alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(STAR_CODES)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["No", "Yes"] if not has_korean_font else ["미충족", "충족"])
    ax.set_ylim(0, 1)
    title = f"STAR coverage ({star_result['score']} pts)"
    if has_korean_font:
        title = f"STAR 충족도 ({star_result['score']}점)"
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def plot_role02_summary(head_binary: dict, star_result: dict, numbers: dict, output_path: str | Path) -> Path:
    """Save a compact role02 summary bar chart."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    has_korean_font = set_korean_font()

    labels = ["Core claim", "Experience"]
    if has_korean_font:
        labels = ["핵심 주장 명확성", "경험 구체성"]
    scores = [
        int(head_binary.get("head_first_score", 0)),
        int(star_result.get("score", 0)),
    ]
    colors = ["#4C72B0", "#55A868"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, scores, color=colors, width=0.48)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, score + 3, str(score), ha="center", fontsize=10)
    ax.set_ylim(0, 110)
    ax.set_ylabel("score")
    ax.set_title("ROLE02 summary" if not has_korean_font else "ROLE02 분석 요약")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def plot_role02_requirement_status(
    head_binary: dict,
    star_result: dict,
    numbers: dict,
    output_path: str | Path,
) -> Path:
    """Save a pass/fail chart for ROLE02 model requirements."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    has_korean_font = set_korean_font()

    labels = ["Core claim", "S", "T", "A", "R"]
    if has_korean_font:
        labels = ["핵심 주장", "S", "T", "A", "R"]

    statuses = [
        int(head_binary.get("is_head_first", 0)),
        int(bool(star_result["checklist"].get("S"))),
        int(bool(star_result["checklist"].get("T"))),
        int(bool(star_result["checklist"].get("A"))),
        int(bool(star_result["checklist"].get("R"))),
    ]
    colors = ["#55A868" if status else "#C44E52" for status in statuses]
    status_text = ["PASS" if status else "MISS" for status in statuses]
    if has_korean_font:
        status_text = ["충족" if status else "미충족" for status in statuses]

    fig, ax = plt.subplots(figsize=(9, 2.8))
    y = np.arange(len(labels))
    bars = ax.barh(y, [1] * len(labels), color=colors, height=0.6)
    for bar, text in zip(bars, status_text):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_y() + bar.get_height() / 2,
            text,
            ha="center",
            va="center",
            color="white",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xticks([])
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set_title("ROLE02 requirement status" if not has_korean_font else "ROLE02 모델 충족 여부")
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


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
