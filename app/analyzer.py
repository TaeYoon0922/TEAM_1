from __future__ import annotations

import re
import sys
import os
from dataclasses import dataclass, field
from typing import Dict, List

# ROLE03 hedge_detector
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models', 'role03_hedge'))
    from hedge_detector import detect_hedge_expressions
except ImportError:
    detect_hedge_expressions = None

# ROLE04 self_detector
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models', 'role04_self'))
    from self_detector import detect_self_language
except ImportError:
    detect_self_language = None

# ROLE05 relevance_detector
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from models.role05_match.relevance_detector import detect_answer_relevance
except ImportError:
    detect_answer_relevance = None


@dataclass
class MetricResult:
    label: str
    score: int
    level: str
    feedback: str
    details: dict = field(default_factory=dict)


@dataclass
class AnalysisResult:
    metrics: Dict[str, MetricResult]
    overall_score: int
    summary: str
    strengths: List[str]
    improvements: List[str]
    sentence_feedback: List[Dict[str, str]]


# ROLE02 키워드 (두괄식·STAR — ROLE02 모델 연결 전 규칙 기반 유지)
FIRST_SENTENCE_SIGNALS = [
    "결과", "성과", "배웠", "기여", "해결", "개선", "달성",
    "줄였", "높였", "만들", "완성", "입사",
]

STAR_SIGNALS = {
    "situation": ["상황", "당시", "문제", "어려움", "과제", "프로젝트", "현장"],
    "task":      ["목표", "역할", "담당", "책임", "해야", "필요", "요구"],
    "action":    ["분석", "제안", "실행", "설계", "개선", "협업", "조정", "도입", "시도"],
    "result":    ["결과", "성과", "달성", "증가", "감소", "개선", "완료", "해결", "배웠"],
}

# ROLE03 모호어 (hedge_detector 미연결 시 폴백용)
VAGUE_WORDS = [
    "열심히", "최선을", "많이", "다양한", "여러", "좋은",
    "성장", "노력", "책임감", "소통", "도전", "꼼꼼",
    "성실", "적극", "최대한",
]

NUMBER_PATTERN = r"\d|%|명|회|개월|년|배"


def analyze_cover_letter(text: str, question: str = "") -> AnalysisResult:
    cleaned = normalize_text(text)
    cleaned_question = normalize_text(question)
    sentences = split_sentences(cleaned)

    metrics = {
        "두괄식":    score_headline_structure(sentences, cleaned),
        "STAR":     score_star_structure(cleaned),
        "모호도":    score_ambiguity(cleaned),
        "기여중심성": score_contribution_focus(cleaned),
        "문장완성도": score_sentence_completeness(sentences),
    }

    if cleaned_question:
        metrics["질문적합도"] = score_question_relevance(cleaned_question, cleaned)
        overall_score = round(
            metrics["질문적합도"].score * 0.25
            + metrics["두괄식"].score   * 0.20
            + metrics["STAR"].score     * 0.22
            + metrics["모호도"].score   * 0.13
            + metrics["기여중심성"].score * 0.12
            + metrics["문장완성도"].score * 0.08
        )
    else:
        overall_score = round(
            metrics["두괄식"].score     * 0.25
            + metrics["STAR"].score    * 0.30
            + metrics["모호도"].score  * 0.20
            + metrics["기여중심성"].score * 0.15
            + metrics["문장완성도"].score * 0.10
        )

    return AnalysisResult(
        metrics=metrics,
        overall_score=overall_score,
        summary=build_summary(overall_score, metrics),
        strengths=build_strengths(metrics),
        improvements=build_improvements(metrics),
        sentence_feedback=build_sentence_feedback(sentences),
    )


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    protected = re.sub(r"([.!?。])\s+", r"\1\n", text)
    return [s.strip() for s in protected.splitlines() if s.strip()]


def clamp(value: int, minimum: int = 0, maximum: int = 100) -> int:
    return max(minimum, min(maximum, value))


# ── ROLE02 영역 ──────────────────────────────────────────────

def score_headline_structure(sentences: List[str], text: str) -> MetricResult:
    if not sentences:
        return MetricResult("두괄식", 0, "분석 불가", "분석할 문장이 없습니다.")

    first = sentences[0]
    first_ratio = len(first) / max(len(text), 1)
    signal_count = count_keywords(first, FIRST_SENTENCE_SIGNALS)
    has_number = bool(re.search(NUMBER_PATTERN, first))

    score = 35
    score += min(signal_count * 18, 36)
    score += 14 if has_number else 0
    score += 15 if first_ratio <= 0.35 else -10

    feedback = "첫 문장에 핵심 성과와 결론이 비교적 잘 드러납니다."
    if score < 60:
        feedback = "첫 문장에 지원자의 결론, 성과, 직무 적합성을 더 먼저 제시하면 좋습니다."

    return MetricResult("두괄식", clamp(score), level_from_score(score), feedback)


def score_star_structure(text: str) -> MetricResult:
    covered = [part for part, kws in STAR_SIGNALS.items() if count_keywords(text, kws) > 0]
    score = len(covered) * 20
    if re.search(NUMBER_PATTERN, text):
        score += 15
    if len(text) >= 250:
        score += 5

    missing = [p.upper() for p in STAR_SIGNALS if p not in covered]
    feedback = "상황, 과제, 행동, 결과의 흐름이 확인됩니다."
    if missing:
        feedback = f"{', '.join(missing)} 요소가 약합니다. 해당 내용을 한두 문장으로 보강하세요."

    return MetricResult("STAR", clamp(score), level_from_score(score), feedback)


# ── ROLE03 영역 ──────────────────────────────────────────────

def score_ambiguity(text: str) -> MetricResult:
    """높을수록 표현이 명확함 (0~100)."""
    if detect_hedge_expressions is not None:
        result = detect_hedge_expressions(text)
        score     = result["score"]
        hit_count = result["hit_count"]
        feedback  = result["summary"]
        if result["feedback_items"]:
            top = result["feedback_items"][0]
            feedback += f" (예: '{top['original']}' {top['suggestion']})"
        label = f"표현 명료성 — 모호 표현 {hit_count}개 발견"
        return MetricResult(label=label, score=score, level=level_from_score(score), feedback=feedback)

    # hedge_detector 미연결 시 폴백 — 점수 반전해 higher=better 유지
    vague_count = count_keywords(text, VAGUE_WORDS)
    number_count = len(re.findall(NUMBER_PATTERN, text))
    sentence_count = max(len(split_sentences(text)), 1)
    raw = clamp(vague_count * 9 - number_count * 5 + (18 if vague_count / sentence_count >= 1 else 0))
    score = clamp(100 - raw)
    feedback = ("추상 표현이 적고 구체성이 비교적 좋습니다." if score >= 55
                else "추상적인 표현이 많습니다. 수치, 기간, 대상, 행동 단위로 바꾸면 설득력이 올라갑니다.")
    return MetricResult("표현 명료성", score, level_from_score(score), feedback)


# ── ROLE04 영역 ──────────────────────────────────────────────

def score_contribution_focus(text: str) -> MetricResult:
    """R04-05·06·10: self_detector 연동 — 기여중심 표현 비율(0~100)."""
    if detect_self_language is None:
        return MetricResult("기여중심성", 50, "보통", "self_detector 연결 실패.")

    raw = detect_self_language(text)
    contrib_score = raw["contribution_score"]
    grade_feedback = {
        "A": "기여·성과 중심 표현이 뚜렷합니다. 결과가 잘 드러납니다.",
        "B": "기여 표현이 우세하지만 자기중심 표현이 일부 섞여 있습니다.",
        "C": "자기중심·기여 표현이 혼재합니다. 수치와 결과 중심으로 보완하세요.",
        "D": "자기중심 표현이 지배적입니다. 결과·수치·외부 주어로 재작성하세요.",
    }
    return MetricResult(
        label="기여중심성",
        score=contrib_score,
        level=level_from_score(contrib_score),
        feedback=grade_feedback.get(raw["grade"], raw["summary"]),
        details={
            "self_score":             raw["self_score"],
            "contribution_score":     contrib_score,
            "grade":                  raw["grade"],
            "self_hit_count":         raw["self_hit_count"],
            "contribution_hit_count": raw["contribution_hit_count"],
            "number_count":           raw["number_count"],
            "highlighted_html":       raw["highlighted_html"],
            "feedback_items":         raw["feedback_items"],
        },
    )


def score_sentence_completeness(sentences: List[str]) -> MetricResult:
    """R04-11: 문장 완성도 — 문장 길이·종결어·단어 반복 종합 (0~100, 높을수록 좋음)."""
    if not sentences:
        return MetricResult("문장완성도", 50, "보통", "분석할 문장이 없습니다.")

    score = 100
    issues: List[str] = []

    short_count = sum(1 for s in sentences if len(s) < 15)
    if short_count:
        score -= min(short_count * 8, 24)
        issues.append(f"짧은 문장 {short_count}개")

    ending_freq: Dict[str, int] = {}
    for s in sentences:
        for pattern in ["했습니다", "입니다", "습니다", "였습니다", "됩니다"]:
            if s.endswith(pattern):
                ending_freq[pattern] = ending_freq.get(pattern, 0) + 1
                break
    repeated_endings = {k: v for k, v in ending_freq.items() if v >= 3}
    if repeated_endings:
        score -= min(len(repeated_endings) * 12, 24)
        top = max(repeated_endings, key=repeated_endings.get)
        issues.append(f"'{top}' 종결 {repeated_endings[top]}회 반복")

    words = re.findall(r"[가-힣]{4,}", " ".join(sentences))
    word_freq: Dict[str, int] = {}
    for w in words:
        word_freq[w] = word_freq.get(w, 0) + 1
    repeated_words = {w: c for w, c in word_freq.items() if c >= 4}
    if repeated_words:
        score -= min(len(repeated_words) * 8, 16)
        top_w = max(repeated_words, key=repeated_words.get)
        issues.append(f"'{top_w}' {repeated_words[top_w]}회 반복")

    score = clamp(score)
    feedback = ("개선 포인트: " + ", ".join(issues) + ". 문장 다양성을 높이세요."
                if issues else "문장 길이와 표현이 고르게 구성되어 있습니다.")
    return MetricResult("문장완성도", score, level_from_score(score), feedback)


# ── ROLE05 영역 ──────────────────────────────────────────────

def score_question_relevance(question: str, answer: str) -> MetricResult:
    if detect_answer_relevance is None:
        return MetricResult("질문적합도", 0, "분석 불가", "질문-답변 적합도 모델을 불러오지 못했습니다.")
    result = detect_answer_relevance(question, answer)
    feedback = result["summary"]
    if result["feedback_items"]:
        feedback = result["feedback_items"][0]
    return MetricResult("질문적합도", result["score"], result["grade"], feedback)


# ── 공통 유틸 ────────────────────────────────────────────────

def count_keywords(text: str, keywords: List[str]) -> int:
    return sum(text.count(kw) for kw in keywords)


def level_from_score(score: int) -> str:
    if score >= 80:
        return "우수"
    if score >= 60:
        return "보통"
    return "보완 필요"


def risk_level_from_score(score: int) -> str:
    if score >= 70:
        return "높음"
    if score >= 40:
        return "주의"
    return "낮음"


def build_summary(overall_score: int, metrics: Dict[str, MetricResult]) -> str:
    structural = ["두괄식", "STAR"]
    if "질문적합도" in metrics:
        structural.insert(0, "질문적합도")
    weak_struct = min(structural, key=lambda k: metrics[k].score)
    need_improve = ("모호도" if metrics["모호도"].score <= metrics["기여중심성"].score
                    else "기여중심성")

    if "질문적합도" in metrics and metrics["질문적합도"].score < 50:
        return "답변의 표현 품질과 별개로 질문에 직접 답하는 힘이 약합니다. 질문 핵심어와 요구 조건을 먼저 맞춘 뒤 구조를 다듬으세요."
    if overall_score >= 80:
        return "핵심 메시지와 근거가 잘 연결된 자소서입니다. 문장 단위의 구체성만 더 다듬으면 완성도가 높아집니다."
    if overall_score >= 60:
        return f"기본 구조는 잡혀 있습니다. 특히 {weak_struct} 보강과 {need_improve} 개선이 우선입니다."
    return f"현재는 설득 근거가 약하게 보입니다. {weak_struct} 구조를 다시 잡고 {need_improve}을 먼저 개선하세요."


def build_strengths(metrics: Dict[str, MetricResult]) -> List[str]:
    strengths = []
    if metrics.get("질문적합도") and metrics["질문적합도"].score >= 70:
        strengths.append("질문 의도를 비교적 잘 받아서 답변하고 있습니다.")
    if metrics["두괄식"].score >= 70:
        strengths.append("첫 부분에서 핵심 메시지를 비교적 빠르게 제시합니다.")
    if metrics["STAR"].score >= 70:
        strengths.append("경험을 상황, 행동, 결과 흐름으로 설명하려는 구조가 보입니다.")
    if metrics["모호도"].score >= 70:
        strengths.append("추상 표현이 과하지 않아 문장이 비교적 명확합니다.")
    if metrics["기여중심성"].score >= 60:
        strengths.append("기여와 성과 중심 표현이 잘 드러납니다.")
    if metrics["문장완성도"].score >= 80:
        strengths.append("문장 길이와 표현 다양성이 균형 잡혀 있습니다.")
    return strengths or ["강점이 드러나기 시작했지만, 아직 수치와 결과 표현을 더 보강할 여지가 큽니다."]


def build_improvements(metrics: Dict[str, MetricResult]) -> List[str]:
    improvements = []
    if metrics.get("질문적합도") and metrics["질문적합도"].score < 70:
        improvements.append("질문에서 요구한 핵심어와 조건을 첫 문단에 직접 반영하세요.")
    if metrics["두괄식"].score < 70:
        improvements.append("첫 문장을 '무엇을 개선했고 어떤 결과를 냈는지'로 다시 시작하세요.")
    if metrics["STAR"].score < 70:
        improvements.append("상황-S, 과제-T, 행동-A, 결과-R가 각각 보이도록 문단을 재배치하세요.")
    if metrics["모호도"].score < 70:
        improvements.append("'열심히', '다양한', '성장' 같은 표현을 숫자, 기간, 대상, 행동으로 바꾸세요.")
    if metrics["기여중심성"].score < 50:
        improvements.append("'제가 했다' 다음에 팀, 고객, 조직에 생긴 변화를 한 문장 추가하세요.")
    if metrics["문장완성도"].score < 70:
        improvements.append(metrics["문장완성도"].feedback)
    return improvements


def build_sentence_feedback(sentences: List[str]) -> List[Dict[str, str]]:
    feedback = []
    for idx, sentence in enumerate(sentences[:8], start=1):
        vague_hits = [w for w in VAGUE_WORDS if w in sentence]
        has_number = bool(re.search(NUMBER_PATTERN, sentence))

        if vague_hits and not has_number:
            comment = f"'{', '.join(vague_hits[:3])}' 표현이 추상적입니다. 수치나 실제 행동으로 바꿔보세요."
        elif idx == 1 and not any(sig in sentence for sig in FIRST_SENTENCE_SIGNALS):
            comment = "첫 문장에는 결론이나 성과를 더 직접적으로 배치하는 편이 좋습니다."
        elif has_number:
            comment = "구체적인 수치가 있어 설득력에 도움이 됩니다."
        else:
            comment = "문장의 역할은 좋지만, 결과나 영향이 더 드러나면 좋습니다."

        feedback.append({"sentence": sentence, "comment": comment})
    return feedback
