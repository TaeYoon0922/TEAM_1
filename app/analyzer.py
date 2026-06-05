from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

try:
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models', 'role03_hedge'))
    from hedge_detector import detect_hedge_expressions
except ImportError:
    detect_hedge_expressions = None

try:
    from models.role05_match.relevance_detector import detect_answer_relevance
except ImportError:
    detect_answer_relevance = None

@dataclass
class MetricResult:
    label: str
    score: int
    level: str
    feedback: str


@dataclass
class AnalysisResult:
    metrics: Dict[str, MetricResult]
    overall_score: int
    summary: str
    strengths: List[str]
    improvements: List[str]
    sentence_feedback: List[Dict[str, str]]



MOCK_FIRST_SENTENCE_SIGNALS = [
    "결과",
    "성과",
    "배웠",
    "기여",
    "해결",
    "개선",
    "달성",
    "줄였",
    "높였",
    "만들",
    "완성",
    "입사",
]

MOCK_STAR_SIGNALS = {
    "situation": ["상황", "당시", "문제", "어려움", "과제", "프로젝트", "현장"],
    "task": ["목표", "역할", "담당", "책임", "해야", "필요", "요구"],
    "action": ["분석", "제안", "실행", "설계", "개선", "협업", "조정", "도입", "시도"],
    "result": ["결과", "성과", "달성", "증가", "감소", "개선", "완료", "해결", "배웠"],
}

MOCK_SELF_WORDS = ["저는", "제가", "저의", "나는", "내가", "나의", "나를", "저를"]
MOCK_OTHER_WORDS = ["팀", "고객", "사용자", "동료", "조직", "회사", "파트너", "구성원", "멘토"]
NUMBER_PATTERN = r"\d|%|명|회|개월|년|배"


def analyze_cover_letter(text: str, question: str = "") -> AnalysisResult:
    cleaned = normalize_text(text)
    cleaned_question = normalize_text(question)
    sentences = split_sentences(cleaned)

    metrics = {
        "두괄식": score_headline_structure(sentences, cleaned),
        "STAR": score_star_structure(cleaned),
        "모호도": score_ambiguity(cleaned),
        "자기중심": score_self_centered(cleaned),
    }

    clarity_score = metrics["모호도"].score
    balance_score = 100 - metrics["자기중심"].score

    if cleaned_question:
        metrics["질문적합도"] = score_question_relevance(cleaned_question, cleaned)
        overall_score = round(
            (
                metrics["질문적합도"].score * 0.30
                + metrics["두괄식"].score * 0.20
                + metrics["STAR"].score * 0.25
                + clarity_score * 0.15
                + balance_score * 0.10
            )
        )
    else:
        overall_score = round(
            (
                metrics["두괄식"].score * 0.28
                + metrics["STAR"].score * 0.32
                + clarity_score * 0.22
                + balance_score * 0.18
            )
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
    return [sentence.strip() for sentence in protected.splitlines() if sentence.strip()]


def clamp(value: int, minimum: int = 0, maximum: int = 100) -> int:
    return max(minimum, min(maximum, value))


def score_headline_structure(sentences: List[str], text: str) -> MetricResult:
    if not sentences:
        return MetricResult("두괄식", 0, "분석 불가", "분석할 문장이 없습니다.")

    first_sentence = sentences[0]
    first_ratio = len(first_sentence) / max(len(text), 1)
    signal_count = count_keywords(first_sentence, MOCK_FIRST_SENTENCE_SIGNALS)
    has_number = bool(re.search(NUMBER_PATTERN, first_sentence))

    score = 35
    score += min(signal_count * 18, 36)
    score += 14 if has_number else 0
    score += 15 if first_ratio <= 0.35 else -10

    feedback = "첫 문장에 핵심 성과와 결론이 비교적 잘 드러납니다."
    if score < 60:
        feedback = "첫 문장에 지원자의 결론, 성과, 직무 적합성을 더 먼저 제시하면 좋습니다."

    return MetricResult("두괄식", clamp(score), level_from_score(score), feedback)


def score_star_structure(text: str) -> MetricResult:
    covered = []
    for part, keywords in MOCK_STAR_SIGNALS.items():
        if count_keywords(text, keywords) > 0:
            covered.append(part)

    score = len(covered) * 20
    if re.search(NUMBER_PATTERN, text):
        score += 15
    if len(text) >= 250:
        score += 5

    missing = [part.upper() for part in MOCK_STAR_SIGNALS if part not in covered]
    feedback = "상황, 과제, 행동, 결과의 흐름이 확인됩니다."
    if missing:
        feedback = f"{', '.join(missing)} 요소가 약합니다. 해당 내용을 한두 문장으로 보강하세요."

    return MetricResult("STAR", clamp(score), level_from_score(score), feedback)


def score_ambiguity(text: str) -> MetricResult:
    if detect_hedge_expressions is None:
        return MetricResult("표현 명료성", 50, "보통", "hedge_detector 연결 실패.")

    result = detect_hedge_expressions(text)
    score     = result["score"]
    hit_count = result["hit_count"]
    grade     = result["grade"]
    feedback  = result["summary"]

    if result["feedback_items"]:
        top = result["feedback_items"][0]
        feedback += f" (예: '{top['original']}' {top['suggestion']})"

    return MetricResult(
        label    = f"표현 명료성 — 모호 표현 {hit_count}개 발견",
        score    = score,
        level    = grade,
        feedback = feedback,
    )

def score_self_centered(text: str) -> MetricResult:
    self_count = count_keywords(text, MOCK_SELF_WORDS)
    other_count = count_keywords(text, MOCK_OTHER_WORDS)

    score = self_count * 13 - other_count * 7
    if self_count >= 5 and other_count == 0:
        score += 20

    score = clamp(score)
    feedback = "개인 경험과 팀, 고객, 조직 관점의 균형이 괜찮습니다."
    if score >= 45:
        feedback = "자기 서술 비중이 높습니다. 팀, 고객, 조직에 준 영향을 함께 적어 균형을 맞추세요."

    return MetricResult("자기중심", score, risk_level_from_score(score), feedback)


def score_question_relevance(question: str, answer: str) -> MetricResult:
    if detect_answer_relevance is None:
        return MetricResult(
            "질문적합도",
            0,
            "분석 불가",
            "질문-답변 적합도 모델을 불러오지 못했습니다.",
        )

    result = detect_answer_relevance(question, answer)
    feedback = result["summary"]
    if result["feedback_items"]:
        feedback = result["feedback_items"][0]
    return MetricResult("질문적합도", result["score"], result["grade"], feedback)


def count_keywords(text: str, keywords: List[str]) -> int:
    return sum(text.count(keyword) for keyword in keywords)


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
    structural_metrics = ["두괄식", "STAR"]
    if "질문적합도" in metrics:
        structural_metrics.insert(0, "질문적합도")

    weak_metric = min(structural_metrics, key=lambda key: metrics[key].score)
    risk_metric = max(["모호도", "자기중심"], key=lambda key: metrics[key].score)

    if "질문적합도" in metrics and metrics["질문적합도"].score < 50:
        return "답변의 표현 품질과 별개로 질문에 직접 답하는 힘이 약합니다. 질문 핵심어와 요구 조건을 먼저 맞춘 뒤 구조를 다듬으세요."
    if overall_score >= 80:
        return "핵심 메시지와 근거가 잘 연결된 자소서입니다. 문장 단위의 구체성만 더 다듬으면 완성도가 높아집니다."
    if overall_score >= 60:
        return f"기본 구조는 잡혀 있습니다. 특히 {weak_metric} 보강과 {risk_metric} 완화가 우선입니다."
    return f"현재는 설득 근거가 약하게 보입니다. {weak_metric} 구조를 다시 잡고 {risk_metric} 표현을 먼저 줄이세요."


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
    if metrics["자기중심"].score <= 35:
        strengths.append("개인 중심 서술과 외부 영향 서술의 균형이 좋습니다.")
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
    if metrics["자기중심"].score > 35:
        improvements.append("'제가 했다' 다음에 팀, 고객, 조직에 생긴 변화를 한 문장 추가하세요.")
    return improvements


def build_sentence_feedback(sentences: List[str]) -> List[Dict[str, str]]:
    feedback = []
    for index, sentence in enumerate(sentences[:8], start=1):
        fallback_words = ["열심히", "최선을", "다양한", "여러", "노력", "책임감", "소통", "성장"]
        vague_hits = [word for word in fallback_words if word in sentence]
        has_number = bool(re.search(NUMBER_PATTERN, sentence))

        if vague_hits and not has_number:
            comment = f"'{', '.join(vague_hits[:3])}' 표현이 추상적입니다. 수치나 실제 행동으로 바꿔보세요."
        elif index == 1 and not any(signal in sentence for signal in MOCK_FIRST_SENTENCE_SIGNALS):
            comment = "첫 문장에는 결론이나 성과를 더 직접적으로 배치하는 편이 좋습니다."
        elif has_number:
            comment = "구체적인 수치가 있어 설득력에 도움이 됩니다."
        else:
            comment = "문장의 역할은 좋지만, 결과나 영향이 더 드러나면 좋습니다."

        feedback.append({"sentence": sentence, "comment": comment})
    return feedback
