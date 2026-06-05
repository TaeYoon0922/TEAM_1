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

try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models', 'role04_self'))
    from self_detector import detect_self_language
except ImportError:
    detect_self_language = None

@dataclass
class MetricResult:
    label: str
    score: int
    level: str
    feedback: str
    applicable: bool = True   # 군집별 게이팅용(추후 cluster_map 연동). False면 '해당 없음'


@dataclass
class AnalysisResult:
    metrics: Dict[str, MetricResult]
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

NUMBER_PATTERN = r"\d|%|명|회|개월|년|배"


# 최종 표시 지표 7개 (순서 = 화면 노출 순서)
INDICATORS = [
    "문항 적합성",      # 질문에 맞는 답변인지        ← relevance_detector
    "핵심 주장 명확성",  # 두괄식, 핵심 메시지          ← KoSimCSE(폴백: 규칙)
    "경험 구체성",      # STAR, 실제 행동과 결과       ← 규칙(STAR)
    "지원 적합성",      # 직무/학교/전공 연결성        ← (임시 휴리스틱)
    "표현 명료성",      # 모호어·추상어·상투어         ← hedge_detector
    "자기표현 차별성",   # 본인만의 경험과 관점         ← self_detector
    "문장 완성도",      # 맞춤법·문장 구조·가독성       ← (임시 휴리스틱)
]


# 문항 적합성 점수가 이 미만이면 나머지 분석을 멈춘다(하드 게이트).
# relevance_detector 점수가 박한 편(온토픽도 30~50)이라, 명백한 동문서답(0 근처)만
# 걸러지도록 보수적으로 20으로 설정. 모델 점수 분포가 바뀌면 재조정 필요.
GATE_THRESHOLD = 20


def get_applicable_indicators(question: str):
    """질문 → 군집(predict_cluster) → 적용 지표 집합. 군집 판별 실패/질문 없음이면 None(=전체 적용)."""
    if not question:
        return None
    try:
        from models.role05_match.question_clusterer import predict_cluster  # 무거우니 지연 import
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

    # ── 1단계 게이트: 문항 적합성 ──────────────────────────────────────────────
    if cleaned_question:
        relevance = score_question_relevance(cleaned_question, cleaned)
    else:
        relevance = MetricResult(
            "문항 적합성", 0, "분석 보류",
            "질문을 함께 입력하면 답변이 문항에 맞는지 분석합니다.", applicable=False,
        )
    metrics["문항 적합성"] = relevance

    # 게이트 미달 → 나머지 6지표는 돌리지 않고 '분석 보류'로 표시
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

    # ── 2단계 군집 기반 지표 게이팅 ────────────────────────────────────────────
    applicable = get_applicable_indicators(cleaned_question)  # set 또는 None(=전체 적용)

    def gated(label: str, compute) -> MetricResult:
        if applicable is not None and label not in applicable:
            return MetricResult(
                label, 0, "해당 없음",
                "이 질문 유형에는 해당하지 않는 지표입니다.", applicable=False,
            )
        return compute()

    metrics["핵심 주장 명확성"] = gated("핵심 주장 명확성", lambda: score_core_claim_clarity(cleaned, sentences))
    metrics["경험 구체성"] = gated("경험 구체성", lambda: score_star_structure(cleaned))
    metrics["지원 적합성"] = gated("지원 적합성", lambda: score_job_fit(cleaned, cleaned_question))
    metrics["표현 명료성"] = gated("표현 명료성", lambda: score_ambiguity(cleaned))
    metrics["자기표현 차별성"] = gated("자기표현 차별성", lambda: score_self_centered(cleaned))
    metrics["문장 완성도"] = gated("문장 완성도", lambda: score_sentence_quality(sentences, cleaned))

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

    return MetricResult("경험 구체성", clamp(score), level_from_score(score), feedback)


def score_core_claim_clarity(text: str, sentences: List[str]) -> MetricResult:
    """핵심 주장 명확성(두괄식). KoSimCSE(role04) 우선, 실패 시 규칙 기반 폴백."""
    try:
        from KoSimCSE import analyze_paragraph  # role04_self (sys.path 등록됨), 무거우니 지연 import
        res = analyze_paragraph(text)
        if res["structure_type"] == "too_short":
            raise ValueError("too_short")
        score = clamp(round((res["dugoalsik_score"] or 0) * 100))
        level = {
            "dugoalsik":   "우수",
            "migugoalsik": "보완 필요",
            "list_type":   "보완 필요",
        }.get(res["structure_type"], "보통")
        feedback = res["feedback"].split("\n")[0].strip()
        feedback = re.sub(r"\s*\(점수[^)]*\)|\s*\(균일도[^)]*\)", "", feedback)  # 점수 노출 제거
        return MetricResult("핵심 주장 명확성", score, level, feedback)
    except Exception:
        # 폴백: 기존 규칙 기반 두괄식 판정
        fallback = score_headline_structure(sentences, text)
        return MetricResult("핵심 주장 명확성", fallback.score, fallback.level, fallback.feedback)


def score_job_fit(text: str, question: str = "") -> MetricResult:
    """지원 적합성(직무/학교/전공 연결성) — 임시 휴리스틱. 전용 모델 연결 전까지 참고용."""
    link_words = ["직무", "전공", "학과", "직무경험", "지원분야", "회사", "당사", "귀사", "조직", "현장"]
    hits = count_keywords(text, link_words)
    score = clamp(30 + hits * 12)
    if score >= 70:
        feedback = "직무·전공·회사와의 연결 표현이 보입니다. 구체적 직무명·전공 지식과 연결하면 더 강해집니다."
    else:
        feedback = "지원 직무/전공/회사와의 연결이 약합니다. '○○ 직무에 필요한 ○○ 역량'처럼 명시적으로 연결하세요."
    return MetricResult("지원 적합성", score, level_from_score(score), feedback)


def score_sentence_quality(sentences: List[str], text: str) -> MetricResult:
    """문장 완성도(문장 구조·가독성) — 임시 휴리스틱. 맞춤법 검사 모델 연결 전까지 참고용."""
    if not sentences:
        return MetricResult("문장 완성도", 0, "분석 불가", "분석할 문장이 없습니다.")
    lengths = [len(s) for s in sentences]
    avg_len = sum(lengths) / len(lengths)
    long_ratio = sum(1 for n in lengths if n > 120) / len(lengths)
    score = clamp(round(85 - long_ratio * 60 - max(0, avg_len - 90) * 0.5))
    if long_ratio >= 0.3 or avg_len > 100:
        feedback = "한 문장이 깁니다. 한 문장에 한 메시지만 담아 짧게 끊으면 가독성이 올라갑니다."
    else:
        feedback = "문장 길이와 구조는 대체로 읽기 좋습니다. 맞춤법은 별도 검수를 권장합니다."
    return MetricResult("문장 완성도", score, level_from_score(score), feedback)


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
    if detect_self_language is None:
        return MetricResult("자기표현 차별성", 50, "주의", "self_detector 연결 실패.")

    result = detect_self_language(text)
    # self_score: 높을수록 자기중심(나 중심) 표현이 많음
    score = clamp(result["self_score"])

    feedback = result["summary"]
    if result["feedback_items"]:
        top = result["feedback_items"][0]
        feedback += f" (예: '{top['original']}' {top['suggestion']})"

    return MetricResult("자기표현 차별성", score, risk_level_from_score(score), feedback)


def score_question_relevance(question: str, answer: str) -> MetricResult:
    if detect_answer_relevance is None:
        return MetricResult(
            "문항 적합성",
            0,
            "분석 불가",
            "질문-답변 적합도 모델을 불러오지 못했습니다.",
        )

    result = detect_answer_relevance(question, answer)
    feedback = result["summary"]
    if result["feedback_items"]:
        feedback = result["feedback_items"][0]
    return MetricResult("문항 적합성", result["score"], result["grade"], feedback)


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


def _applicable(metrics: Dict[str, MetricResult], key: str) -> bool:
    m = metrics.get(key)
    return bool(m and m.applicable)


def build_summary(metrics: Dict[str, MetricResult]) -> str:
    """점수 없는 정성 요약 — 가장 약한 지표를 짚어준다."""
    candidates = [k for k in INDICATORS if _applicable(metrics, k)]
    if not candidates:
        return "분석할 내용이 부족합니다. 답변을 더 작성해 주세요."

    weak = min(candidates, key=lambda k: metrics[k].score)
    if _applicable(metrics, "문항 적합성") and metrics["문항 적합성"].score < 50:
        return "표현 품질과 별개로 질문에 직접 답하는 힘이 약합니다. 질문 핵심어와 요구 조건을 먼저 맞추세요."
    return f"전반적으로 '{weak}'이(가) 가장 약합니다. 아래 피드백에서 이 부분부터 보완해 보세요."


def _on(metrics: Dict[str, MetricResult], key: str) -> bool:
    """해당 지표가 적용 대상(applicable)인지."""
    return _applicable(metrics, key)


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

        # hedge_detector로 문장별 분석
        if detect_hedge_expressions is not None:
            result = detect_hedge_expressions(sentence)
            hits = result["hits"]
            feedback_items = result["feedback_items"]

            if feedback_items:
                # 표현별 개선 제안 합치기
                suggestions = []
                for fb in feedback_items:
                    suggestions.append(f"'{fb['original']}' {fb['suggestion']}")
                comment = " / ".join(suggestions)

            elif has_number:
                comment = "구체적인 수치가 있어 설득력에 도움이 됩니다."
            elif index == 1 and not any(s in sentence for s in MOCK_FIRST_SENTENCE_SIGNALS):
                comment = "첫 문장에는 결론이나 성과를 더 직접적으로 배치하는 편이 좋습니다."
            else:
                comment = "문장의 역할은 좋지만, 결과나 영향이 더 드러나면 좋습니다."

        else:
            # fallback
            fallback_words = ["열심히", "최선을", "다양한", "여러", "노력", "책임감", "소통", "성장"]
            vague_hits = [w for w in fallback_words if w in sentence]
            if vague_hits and not has_number:
                comment = f"'{', '.join(vague_hits[:3])}' 표현이 추상적입니다. 수치나 실제 행동으로 바꿔보세요."
            elif has_number:
                comment = "구체적인 수치가 있어 설득력에 도움이 됩니다."
            else:
                comment = "문장의 역할은 좋지만, 결과나 영향이 더 드러나면 좋습니다."

        feedback.append({"sentence": sentence, "comment": comment})
    return feedback
