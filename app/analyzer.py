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
    applicable: bool = True   # 군집별 게이팅용. False면 '해당 없음'
    details: dict = field(default_factory=dict)


@dataclass
class AnalysisResult:
    metrics: Dict[str, MetricResult]
    summary: str
    strengths: List[str]
    improvements: List[str]
    sentence_feedback: List[Dict[str, str]]


# 최종 표시 지표 7개 (순서 = 화면 노출 순서)
INDICATORS = [
    "문항 적합성",       # 질문에 맞는 답변인지        ← relevance_detector
    "핵심 주장 명확성",  # 두괄식, 핵심 메시지          ← KoSimCSE(폴백: 규칙)
    "경험 구체성",       # STAR, 실제 행동과 결과       ← 규칙(STAR)
    "지원 적합성",       # 직무/학교/전공 연결성        ← (임시 휴리스틱)
    "표현 명료성",       # 모호어·추상어·상투어         ← hedge_detector
    "자기표현 차별성",   # 본인만의 경험과 관점         ← self_detector
    "문장 완성도",       # 맞춤법·문장 구조·가독성       ← (임시 휴리스틱)
]

# 문항 적합성 점수가 이 미만이면 나머지 분석 중단
GATE_THRESHOLD = 20

# ROLE02 키워드 (두괄식·STAR 폴백)
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

# ROLE03 모호어 (hedge_detector 미연결 시 폴백)
VAGUE_WORDS = [
    "열심히", "최선을", "많이", "다양한", "여러", "좋은",
    "성장", "노력", "책임감", "소통", "도전", "꼼꼼",
    "성실", "적극", "최대한",
]

NUMBER_PATTERN = r"\d|%|명|회|개월|년|배"


def get_applicable_indicators(question: str):
    """질문 → 군집 → 적용 지표 집합. 실패/질문 없으면 None(=전체 적용)."""
    if not question:
        return None
    try:
        from models.role05_match.question_clusterer import predict_cluster
        res = predict_cluster(question)
        inds = res.get("cluster_info", {}).get("indicators")
        return set(inds) if inds else None
    except Exception:
        return None


def analyze_cover_letter(text: str, question: str = "") -> AnalysisResult:
    cleaned = normalize_text(text)
    cleaned_question = normalize_text(question)
    sentences = split_sentences(cleaned)

    metrics: Dict[str, MetricResult] = {}

    # 1단계 게이트: 문항 적합성
    if cleaned_question:
        relevance = score_question_relevance(cleaned_question, cleaned)
    else:
        relevance = MetricResult(
            "문항 적합성", 0, "분석 보류",
            "질문을 함께 입력하면 답변이 문항에 맞는지 분석합니다.", applicable=False,
        )
    metrics["문항 적합성"] = relevance

    # 게이트 미달 → 나머지 지표 '분석 보류'
    if relevance.applicable and relevance.score < GATE_THRESHOLD:
        for label in INDICATORS[1:]:
            metrics[label] = MetricResult(
                label, 0, "분석 보류",
                "문항 적합성이 낮아 분석을 멈췄습니다. 질문에 맞는 답변으로 고친 뒤 다시 시도하세요.",
                applicable=False,
            )
        return AnalysisResult(
            metrics=metrics,
            summary="답변이 질문에서 벗어나 있어 세부 분석을 멈췄습니다. 질문 핵심어와 요구 조건부터 맞춰 주세요.",
            strengths=[],
            improvements=["질문에서 요구한 핵심어와 조건을 먼저 답변에 직접 반영하세요."],
            sentence_feedback=build_sentence_feedback(sentences),
        )

    # 2단계 군집 기반 지표 게이팅
    applicable = get_applicable_indicators(cleaned_question)

    def gated(label: str, compute) -> MetricResult:
        if applicable is not None and label not in applicable:
            return MetricResult(
                label, 0, "해당 없음",
                "이 질문 유형에는 해당하지 않는 지표입니다.", applicable=False,
            )
        return compute()

    metrics["핵심 주장 명확성"] = gated("핵심 주장 명확성", lambda: score_core_claim_clarity(cleaned, sentences))
    metrics["경험 구체성"]     = gated("경험 구체성",     lambda: score_star_structure(cleaned))
    metrics["지원 적합성"]     = gated("지원 적합성",     lambda: score_job_fit(cleaned, cleaned_question))
    metrics["표현 명료성"]     = gated("표현 명료성",     lambda: score_ambiguity(cleaned))
    metrics["자기표현 차별성"] = gated("자기표현 차별성", lambda: score_self_centered(cleaned))
    metrics["문장 완성도"]     = gated("문장 완성도",     lambda: score_sentence_quality(sentences, cleaned))

    return AnalysisResult(
        metrics=metrics,
        summary=build_summary(metrics),
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
        return MetricResult("핵심 주장 명확성", 0, "분석 불가", "분석할 문장이 없습니다.")
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
    return MetricResult("핵심 주장 명확성", clamp(score), level_from_score(score), feedback)


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
    return MetricResult("경험 구체성", clamp(score), level_from_score(score), feedback)


def score_core_claim_clarity(text: str, sentences: List[str]) -> MetricResult:
    """핵심 주장 명확성. KoSimCSE 우선, 실패 시 규칙 기반 폴백."""
    try:
        from KoSimCSE import analyze_paragraph
        res = analyze_paragraph(text)
        if res["structure_type"] == "too_short":
            raise ValueError("too_short")
        score = clamp(round((res["dugoalsik_score"] or 0) * 100))
        level = {"dugoalsik": "우수", "migugoalsik": "보완 필요", "list_type": "보완 필요"}.get(
            res["structure_type"], "보통"
        )
        feedback = re.sub(r"\s*\(점수[^)]*\)|\s*\(균일도[^)]*\)", "", res["feedback"].split("\n")[0].strip())
        return MetricResult("핵심 주장 명확성", score, level, feedback)
    except Exception:
        fallback = score_headline_structure(sentences, text)
        return MetricResult("핵심 주장 명확성", fallback.score, fallback.level, fallback.feedback)


def score_job_fit(text: str, question: str = "") -> MetricResult:
    """지원 적합성 — ROLE05 job_fit_detector(직무 키워드 사전) 연결. 실패 시 휴리스틱 폴백."""
    try:
        from models.role05_match.job_fit_detector import detect_job_fit
        r = detect_job_fit(text, question)
        return MetricResult(
            "지원 적합성", clamp(r["score"]), r["grade"], r["feedback_items"][0],
            details={"matched_keywords": r["matched_keywords"], "link_hits": r["link_hits"]},
        )
    except Exception:
        return MetricResult(
            "지원 적합성", 0, "미연결",
            "지원 적합성 분석 모듈(job_fit_detector)을 불러오지 못했습니다.",
            applicable=False,
        )


def score_sentence_quality(sentences: List[str], text: str) -> MetricResult:
    """문장 완성도(맞춤법·문장 구조·가독성) — 전용 분석 모듈 미연결."""
    return MetricResult(
        "문장 완성도", 0, "미연결",
        "문장 완성도(맞춤법·가독성) 분석 모듈이 아직 없습니다.",
        applicable=False,
    )


# ── ROLE03 영역 ──────────────────────────────────────────────

def score_ambiguity(text: str) -> MetricResult:
    """표현 명료성 — 높을수록 명확 (0~100)."""
    if detect_hedge_expressions is not None:
        result = detect_hedge_expressions(text)
        score     = result["score"]
        hit_count = result["hit_count"]
        feedback  = result["summary"]
        if result["feedback_items"]:
            top = result["feedback_items"][0]
            feedback += f" (예: '{top['original']}' {top['suggestion']})"
        return MetricResult(
            label=f"표현 명료성 — 모호 표현 {hit_count}개 발견",
            score=score, level=level_from_score(score), feedback=feedback,
        )
    # 폴백: higher=better 유지를 위해 반전
    vague_count = count_keywords(text, VAGUE_WORDS)
    number_count = len(re.findall(NUMBER_PATTERN, text))
    sentence_count = max(len(split_sentences(text)), 1)
    raw = clamp(vague_count * 9 - number_count * 5 + (18 if vague_count / sentence_count >= 1 else 0))
    score = clamp(100 - raw)
    feedback = ("추상 표현이 적고 구체성이 비교적 좋습니다." if score >= 55
                else "추상적인 표현이 많습니다. 수치, 기간, 대상, 행동 단위로 바꾸면 설득력이 올라갑니다.")
    return MetricResult("표현 명료성", score, level_from_score(score), feedback)


# ── ROLE04 영역 ──────────────────────────────────────────────

def score_self_centered(text: str) -> MetricResult:
    """자기표현 차별성 — self_detector 연동. self_score가 낮을수록(기여 중심) 좋음."""
    if detect_self_language is None:
        return MetricResult("자기표현 차별성", 50, "주의", "self_detector 연결 실패.")

    result = detect_self_language(text)
    score = clamp(result["self_score"])  # 높을수록 자기중심(나 중심) → 위험

    feedback = result["summary"]
    if result["feedback_items"]:
        top = result["feedback_items"][0]
        feedback += f" (예: '{top['original']}' {top['suggestion']})"

    return MetricResult(
        label="자기표현 차별성",
        score=score,
        level=risk_level_from_score(score),
        feedback=feedback,
        details={
            "self_score":             result["self_score"],
            "contribution_score":     result["contribution_score"],
            "grade":                  result["grade"],
            "self_hit_count":         result["self_hit_count"],
            "contribution_hit_count": result["contribution_hit_count"],
            "highlighted_html":       result["highlighted_html"],
            "feedback_items":         result["feedback_items"],
        },
    )


# ── ROLE05 영역 ──────────────────────────────────────────────

def score_question_relevance(question: str, answer: str) -> MetricResult:
    if detect_answer_relevance is None:
        return MetricResult(
            "문항 적합성", 0, "분석 불가",
            "질문-답변 적합도 모델을 불러오지 못했습니다.",
        )
    result = detect_answer_relevance(question, answer)
    feedback = result["summary"]
    if result["feedback_items"]:
        feedback = result["feedback_items"][0]
    return MetricResult("문항 적합성", result["score"], result["grade"], feedback)


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


def _applicable(metrics: Dict[str, MetricResult], key: str) -> bool:
    m = metrics.get(key)
    return bool(m and m.applicable)

_on = _applicable


def build_summary(metrics: Dict[str, MetricResult]) -> str:
    candidates = [k for k in INDICATORS if _on(metrics, k)]
    if not candidates:
        return "분석할 내용이 부족합니다. 답변을 더 작성해 주세요."
    weak = min(candidates, key=lambda k: metrics[k].score)
    if _on(metrics, "문항 적합성") and metrics["문항 적합성"].score < 50:
        return "표현 품질과 별개로 질문에 직접 답하는 힘이 약합니다. 질문 핵심어와 요구 조건을 먼저 맞추세요."
    return f"전반적으로 '{weak}'이(가) 가장 약합니다. 아래 피드백에서 이 부분부터 보완해 보세요."


def build_strengths(metrics: Dict[str, MetricResult]) -> List[str]:
    strengths = []
    if _on(metrics, "문항 적합성") and metrics["문항 적합성"].score >= 70:
        strengths.append("질문 의도를 비교적 잘 받아서 답변하고 있습니다.")
    if _on(metrics, "핵심 주장 명확성") and metrics["핵심 주장 명확성"].score >= 70:
        strengths.append("첫 부분에서 핵심 메시지를 비교적 빠르게 제시합니다.")
    if _on(metrics, "경험 구체성") and metrics["경험 구체성"].score >= 70:
        strengths.append("경험을 상황, 행동, 결과 흐름으로 설명하려는 구조가 보입니다.")
    if _on(metrics, "표현 명료성") and metrics["표현 명료성"].score >= 70:
        strengths.append("추상 표현이 과하지 않아 문장이 비교적 명확합니다.")
    if _on(metrics, "자기표현 차별성") and metrics["자기표현 차별성"].score <= 35:
        strengths.append("나 중심 서술과 외부 영향 서술의 균형이 좋습니다.")
    if _on(metrics, "문장 완성도") and metrics["문장 완성도"].score >= 80:
        strengths.append("문장 길이와 구조가 읽기 좋습니다.")
    return strengths or ["강점이 드러나기 시작했지만, 아직 수치와 결과 표현을 더 보강할 여지가 큽니다."]


def build_improvements(metrics: Dict[str, MetricResult]) -> List[str]:
    improvements = []
    if _on(metrics, "문항 적합성") and metrics["문항 적합성"].score < 70:
        improvements.append("질문에서 요구한 핵심어와 조건을 첫 문단에 직접 반영하세요.")
    if _on(metrics, "핵심 주장 명확성") and metrics["핵심 주장 명확성"].score < 70:
        improvements.append("첫 문장을 '무엇을 개선했고 어떤 결과를 냈는지'로 다시 시작하세요.")
    if _on(metrics, "경험 구체성") and metrics["경험 구체성"].score < 70:
        improvements.append("상황-S, 과제-T, 행동-A, 결과-R가 각각 보이도록 문단을 재배치하세요.")
    if _on(metrics, "지원 적합성") and metrics["지원 적합성"].score < 70:
        improvements.append("지원 직무/전공/회사와의 연결을 '○○ 직무에 필요한 ○○ 역량'처럼 명시하세요.")
    if _on(metrics, "표현 명료성") and metrics["표현 명료성"].score < 70:
        improvements.append("'열심히', '다양한', '성장' 같은 표현을 숫자, 기간, 대상, 행동으로 바꾸세요.")
    if _on(metrics, "자기표현 차별성") and metrics["자기표현 차별성"].score > 35:
        improvements.append("'제가 했다' 다음에 팀, 고객, 조직에 생긴 변화를 한 문장 추가하세요.")
    if _on(metrics, "문장 완성도") and metrics["문장 완성도"].score < 70:
        improvements.append("긴 문장은 한 문장에 한 메시지만 담아 짧게 끊으세요.")
    return improvements


def build_sentence_feedback(sentences: List[str]) -> List[Dict[str, str]]:
    feedback = []
    for index, sentence in enumerate(sentences[:8], start=1):
        has_number = bool(re.search(NUMBER_PATTERN, sentence))

        if detect_hedge_expressions is not None:
            result = detect_hedge_expressions(sentence)
            feedback_items = result["feedback_items"]
            if feedback_items:
                suggestions = [f"'{fb['original']}' {fb['suggestion']}" for fb in feedback_items]
                comment = " / ".join(suggestions)
            elif has_number:
                comment = "구체적인 수치가 있어 설득력에 도움이 됩니다."
            elif index == 1 and not any(s in sentence for s in FIRST_SENTENCE_SIGNALS):
                comment = "첫 문장에는 결론이나 성과를 더 직접적으로 배치하는 편이 좋습니다."
            else:
                comment = "문장의 역할은 좋지만, 결과나 영향이 더 드러나면 좋습니다."
        else:
            vague_hits = [w for w in VAGUE_WORDS if w in sentence]
            if vague_hits and not has_number:
                comment = f"'{', '.join(vague_hits[:3])}' 표현이 추상적입니다. 수치나 실제 행동으로 바꿔보세요."
            elif has_number:
                comment = "구체적인 수치가 있어 설득력에 도움이 됩니다."
            else:
                comment = "문장의 역할은 좋지만, 결과나 영향이 더 드러나면 좋습니다."

        feedback.append({"sentence": sentence, "comment": comment})
    return feedback
