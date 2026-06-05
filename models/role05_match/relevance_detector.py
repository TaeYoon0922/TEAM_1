"""
질문 의도 탐지 모듈 (하위 호환 버전)
- 기본: question_clusterer.py의 SBERT + K-Means 군집화 사용
- 폴백: K-Means 모델 미학습 시 규칙 기반 키워드 매칭 사용
- 출력 형식: 기존 detect_question_intents() 인터페이스 유지
"""

from pathlib import Path

# ── 규칙 기반 폴백: 7개 유형 키워드 사전 ──────────────────────────────────────
QUESTION_INTENTS = {
    "motivation": [
        "지원 동기", "지원동기", "지원한 이유", "선택한 이유", "선택하게 된 이유",
        "왜 지원", "이 회사를 선택", "관심을 갖게 된", "이 직무를 선택",
    ],
    "experience": [
        "경험", "경력", "사례", "활동", "프로젝트", "업무", "인턴",
        "해결한", "극복한", "도전한", "성취한",
    ],
    "competency": [
        "역량", "강점", "능력", "기술", "스킬", "전문성", "장점",
        "잘하는", "뛰어난",
    ],
    "weakness": [
        "약점", "단점", "부족한", "개선", "보완", "한계", "아쉬운",
        "실패", "극복해야 할",
    ],
    "collaboration": [
        "협업", "팀워크", "팀 프로젝트", "팀원", "협력", "갈등", "조율",
        "소통", "의사소통", "리더십", "팔로워십",
    ],
    "growth": [
        "성장", "배움", "깨달음", "변화", "발전", "노력", "학습",
        "자기계발", "도전 정신",
    ],
    "plan": [
        "포부", "계획", "목표", "비전", "입사 후", "앞으로", "미래",
        "5년 후", "10년 후", "이루고 싶은",
    ],
}

# 군집 id → 기존 intent 레이블 매핑 (cluster_map.json 이름 확정 전 임시 대응)
_CLUSTER_TO_INTENT = {
    # 사용자가 cluster_map.json에 이름 붙인 뒤 아래를 채워 넣으면 됩니다.
    # 예) 0: "experience", 1: "motivation", 2: "plan"
}


def _rule_based_intent(question: str) -> str:
    """키워드 매칭으로 7개 유형 중 하나 반환. 매칭 없으면 'unknown'."""
    q = question
    best_intent = "unknown"
    best_count  = 0
    for intent, keywords in QUESTION_INTENTS.items():
        count = sum(1 for kw in keywords if kw in q)
        if count > best_count:
            best_count  = count
            best_intent = intent
    return best_intent


def _cluster_based_intent(question: str) -> dict:
    """
    군집화 기반 분류. 반환값:
        {"intent": str, "cluster_id": int, "cluster_info": dict}
    """
    try:
        from models.role05_match.question_clusterer import (
            predict_cluster, load_cluster_map, _load_km,
        )
        km   = _load_km()
        cmap = load_cluster_map()
        res  = predict_cluster(question, km=km, cluster_map=cmap)

        cid    = res["cluster_id"]
        cinfo  = res["cluster_info"]
        # 군집 이름을 기존 intent 레이블로 변환 (매핑 없으면 name 그대로)
        intent = _CLUSTER_TO_INTENT.get(cid, cinfo.get("name", f"cluster_{cid}"))

        return {"intent": intent, "cluster_id": cid, "cluster_info": cinfo}

    except RuntimeError:
        # K-Means 미학습 → 폴백
        return None
    except Exception:
        return None


# ── 공개 API ──────────────────────────────────────────────────────────────────

def detect_question_intents(question: str, use_clustering: bool = True) -> dict:
    """
    질문 유형 탐지 (기존 인터페이스 유지).

    Args:
        question:        분석할 질문 텍스트
        use_clustering:  True면 군집화 우선, 실패 시 규칙 기반 폴백
                         False면 규칙 기반만 사용

    Returns:
        {
          "intent":       str,          # 질문 유형 레이블
          "method":       str,          # "clustering" | "rule_based"
          "cluster_id":   int | None,
          "cluster_info": dict | None,  # models, star_required 등
          "models":       list[str],    # 필요한 평가 모델 목록
          "star_required": bool,
        }
    """
    if use_clustering:
        cluster_result = _cluster_based_intent(question)
        if cluster_result is not None:
            cinfo = cluster_result["cluster_info"]
            return {
                "intent":       cluster_result["intent"],
                "method":       "clustering",
                "cluster_id":   cluster_result["cluster_id"],
                "cluster_info": cinfo,
                "models":       cinfo.get("models", ["star", "hedge"]),
                "star_required": cinfo.get("star_required", True),
            }

    # 규칙 기반 폴백
    intent = _rule_based_intent(question)
    models, star_required = _intent_to_models(intent)
    return {
        "intent":       intent,
        "method":       "rule_based",
        "cluster_id":   None,
        "cluster_info": None,
        "models":       models,
        "star_required": star_required,
    }


def detect_question_intents_multi(question: str,
                                   use_clustering: bool = True) -> dict:
    """
    혼합 질문 처리 버전. split_subquestions로 분리 후 각각 분류, 모델 합집합 반환.

    Returns:
        {
          "original_question": str,
          "subquestions":      list[str],
          "intents":           list[str],
          "models":            list[str],
          "star_required":     bool,
          "method":            str,
        }
    """
    if use_clustering:
        try:
            from models.role05_match.question_clusterer import (
                predict_cluster_multi, load_cluster_map, _load_km,
            )
            km  = _load_km()
            res = predict_cluster_multi(question, km=km,
                                        cluster_map=load_cluster_map())
            return {
                "original_question": res["original_question"],
                "subquestions":      res["subquestions"],
                "intents":           [str(cid) for cid in res["cluster_ids"]],
                "models":            res["merged_models"],
                "star_required":     res["star_required"],
                "method":            "clustering",
            }
        except Exception:
            pass

    # 규칙 기반 폴백
    intent = _rule_based_intent(question)
    models, star_required = _intent_to_models(intent)
    return {
        "original_question": question,
        "subquestions":      [question],
        "intents":           [intent],
        "models":            models,
        "star_required":     star_required,
        "method":            "rule_based",
    }


def _intent_to_models(intent: str) -> tuple:
    """규칙 기반 intent → (models, star_required)"""
    mapping = {
        "experience":    (["star", "hedge", "self"],  True),
        "competency":    (["star", "hedge", "self"],  True),
        "weakness":      (["star", "hedge"],           True),
        "collaboration": (["star", "hedge", "self"],  True),
        "growth":        (["star", "hedge"],           True),
        "motivation":    (["clarity", "hedge"],        False),
        "plan":          (["clarity", "hedge"],        False),
        "unknown":       (["star", "hedge"],           True),
    }
    return mapping.get(intent, (["star", "hedge"], True))


# ── 테스트 ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    samples = [
        "팀 갈등 해결 경험에 대해 서술하시오",
        "협업 중 어려움을 극복한 사례를 작성하시오",
        "지원 동기를 기술하시오",
        "입사 후 포부를 서술하시오",
        "경험을 바탕으로 입사 후 포부를 서술하시오",
    ]

    print("detect_question_intents() 테스트\n" + "=" * 55)
    for q in samples:
        res = detect_question_intents(q)
        print(f"Q: {q}")
        print(f"  intent={res['intent']}  method={res['method']}")
        print(f"  models={res['models']}  star={res['star_required']}\n")
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set


TOKEN_PATTERN = re.compile(r"[가-힣A-Za-z0-9]+")

QUESTION_STOPWORDS = {
    "귀하",
    "본인",
    "자신",
    "대한",
    "대해",
    "대하여",
    "관련",
    "기술",
    "기술해",
    "서술",
    "서술해",
    "설명",
    "설명해",
    "작성",
    "작성해",
    "하시오",
    "하십시오",
    "주십시오",
    "주세요",
    "주시기",
    "바랍니다",
    "무엇",
    "있는지",
    "어떤",
    "어떻게",
    "왜",
    "방식",
    "및",
    "또는",
    "그리고",
    "있는",
    "관련된",
    "구체적",
    "구체적으로",
    "실제",
    "바탕",
    "과정",
    "본인이",
    "본인의",
    "맡은",
    "발휘",
    "일하며",
    "사람들",
    "부족했던",
    "쓰고",
    "위해",
    "했는지",
    "했던",
    "하여",
    "에서",
    "으로",
    "이를",
    "그에",
    "맞춰",
    "이내",
    "아래",
    "항목",
    "문항",
    "질문",
    "답변",
}

QUESTION_INTENTS: Dict[str, Dict[str, Sequence[str]]] = {
    "motivation": {
        "question": ("지원동기", "지원 동기", "지원한 이유", "왜 지원", "입사", "관심", "선택한 이유"),
        "answer": ("지원", "입사", "관심", "매력", "비전", "가치", "사업", "직무", "회사"),
    },
    "experience": {
        "question": ("경험", "사례", "프로젝트", "수행", "해결", "어려움", "갈등", "문제", "도전"),
        "answer": ("경험", "프로젝트", "당시", "문제", "어려움", "해결", "개선", "수행", "진행"),
    },
    "competency": {
        "question": ("강점", "역량", "능력", "전문성", "장점", "직무역량", "적합"),
        "answer": ("강점", "역량", "능력", "전문성", "장점", "활용", "발휘", "적합"),
    },
    "weakness": {
        "question": ("단점", "약점", "부족", "보완", "개선점", "실패"),
        "answer": ("단점", "약점", "부족", "보완", "개선", "극복", "실패", "배웠"),
    },
    "collaboration": {
        "question": ("협업", "팀워크", "조직", "갈등", "소통", "공동", "함께"),
        "answer": ("협업", "팀", "조직", "갈등", "소통", "조율", "함께", "동료"),
    },
    "growth": {
        "question": ("성장과정", "가치관", "성격", "인생관", "생활신조"),
        "answer": ("성장", "가치관", "성격", "신조", "배경", "어릴", "학창"),
    },
    "plan": {
        "question": ("입사 후", "포부", "계획", "목표", "기여", "비전", "어떻게 활용"),
        "answer": ("입사 후", "계획", "목표", "기여", "포부", "비전", "활용", "성장하겠습니다"),
    },
}

GENERIC_ANSWER_TERMS = {
    "열심히",
    "최선을",
    "성실",
    "노력",
    "책임감",
    "긍정",
    "적극",
    "소통",
    "성장",
}


@dataclass
class MatchResult:
    role: str
    metric: str
    score: int
    grade: str
    summary: str
    question_intents: List[str]
    matched_keywords: List[str]
    missing_keywords: List[str]
    subquestion_coverage: float
    feedback_items: List[str]

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "metric": self.metric,
            "score": self.score,
            "grade": self.grade,
            "summary": self.summary,
            "question_intents": self.question_intents,
            "matched_keywords": self.matched_keywords,
            "missing_keywords": self.missing_keywords,
            "subquestion_coverage": self.subquestion_coverage,
            "feedback_items": self.feedback_items,
        }


def detect_answer_relevance(question: str, answer: str) -> dict:
    """Evaluate whether an answer addresses the given self-introduction question.

    The score is high only when the answer overlaps with the question's key terms,
    satisfies the detected question intent, and covers multiple sub-questions.
    """
    question = normalize_text(question)
    answer = normalize_text(answer)

    if not question or not answer:
        return MatchResult(
            role="ROLE_05_MATCH",
            metric="question_answer_relevance",
            score=0,
            grade="분석 불가",
            summary="질문과 답변을 모두 입력해야 질문 적합도를 평가할 수 있습니다.",
            question_intents=[],
            matched_keywords=[],
            missing_keywords=[],
            subquestion_coverage=0.0,
            feedback_items=["질문 원문과 자소서 답변을 함께 넣어 주세요."],
        ).to_dict()

    question_keywords = extract_question_keywords(question)
    answer_tokens = set(tokenize(answer))
    matched_keywords = sorted(find_keyword_matches(question_keywords, answer))
    missing_keywords = sorted(question_keywords - set(matched_keywords))

    intents = detect_question_intents(question)
    intent_score = score_intent_alignment(intents, answer)
    keyword_score = score_keyword_overlap(question_keywords, matched_keywords)
    coverage = score_subquestion_coverage(question, answer, answer_tokens)
    penalty = calc_off_topic_penalty(question_keywords, matched_keywords, intents, answer)

    raw_score = round(keyword_score * 0.45 + intent_score * 0.35 + coverage * 0.20 - penalty)
    score = clamp(raw_score)
    grade, summary = grade_match(score)

    return MatchResult(
        role="ROLE_05_MATCH",
        metric="question_answer_relevance",
        score=score,
        grade=grade,
        summary=summary,
        question_intents=intents,
        matched_keywords=matched_keywords[:12],
        missing_keywords=missing_keywords[:12],
        subquestion_coverage=round(coverage / 100, 2),
        feedback_items=build_feedback(score, intents, missing_keywords, coverage),
    ).to_dict()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def tokenize(text: str) -> List[str]:
    tokens = []
    for raw in TOKEN_PATTERN.findall(text.lower()):
        token = strip_korean_particle(raw)
        if len(token) >= 2 and token not in QUESTION_STOPWORDS:
            tokens.append(token)
    return tokens


def strip_korean_particle(token: str) -> str:
    for suffix in ("으로써", "으로서", "에게서", "까지", "부터", "에게", "에서", "으로", "하고", "이며", "이나", "처럼", "에는", "에도", "만을", "들을", "적인", "하다", "하며", "했고", "했습니다", "하는지", "했는지", "하기", "하는", "했던", "하고", "에는", "한", "할", "은", "는", "이", "가", "을", "를", "의", "에", "와", "과", "도", "만", "로"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 2:
            return token[: -len(suffix)]
    return token


def extract_question_keywords(question: str) -> Set[str]:
    keywords = set(tokenize(question))
    for intent_info in QUESTION_INTENTS.values():
        for phrase in intent_info["question"]:
            if phrase in question:
                keywords.update(tokenize(phrase))
    keywords.update(expand_compound_keywords(keywords))
    return keywords


def expand_compound_keywords(keywords: Set[str]) -> Set[str]:
    expanded = set()
    compound_map = {
        "지원동기": ("지원", "동기"),
        "직무역량": ("직무", "역량"),
        "성장과정": ("성장", "과정"),
        "입사후": ("입사", "계획"),
    }
    for keyword in keywords:
        expanded.update(compound_map.get(keyword, ()))
    return expanded


def find_keyword_matches(question_keywords: Set[str], answer: str) -> Set[str]:
    answer_tokens = set(tokenize(answer))
    matches = set()
    for keyword in question_keywords:
        if keyword in answer_tokens or keyword in answer:
            matches.add(keyword)
            continue
        if len(keyword) >= 3 and any(keyword in token or token in keyword for token in answer_tokens):
            matches.add(keyword)
    return matches


def detect_question_intents(question: str) -> List[str]:
    intents = []
    for intent, intent_info in QUESTION_INTENTS.items():
        if any(signal in question for signal in intent_info["question"]):
            intents.append(intent)
    return intents or ["general"]


def score_keyword_overlap(question_keywords: Set[str], matched_keywords: Sequence[str]) -> int:
    if not question_keywords:
        return 50
    return clamp(round(len(matched_keywords) / len(question_keywords) * 100))


def score_intent_alignment(intents: Sequence[str], answer: str) -> int:
    if not intents or intents == ["general"]:
        return 60

    scores = []
    for intent in intents:
        answer_signals = QUESTION_INTENTS[intent]["answer"]
        hit_count = sum(1 for signal in answer_signals if signal in answer)
        scores.append(clamp(hit_count * 25))
    return round(sum(scores) / len(scores))


def score_subquestion_coverage(question: str, answer: str, answer_tokens: Set[str]) -> int:
    parts = split_subquestions(question)
    if len(parts) <= 1:
        return 70

    covered = 0
    for part in parts:
        part_keywords = extract_question_keywords(part)
        if not part_keywords:
            covered += 1
            continue
        part_matches = find_keyword_matches(part_keywords, answer)
        part_intents = detect_question_intents(part)
        has_intent = score_intent_alignment(part_intents, answer) >= 45
        if len(part_matches) / max(len(part_keywords), 1) >= 0.25 or has_intent:
            covered += 1

    return clamp(round(covered / len(parts) * 100))


def split_subquestions(question: str) -> List[str]:
    cleaned = re.sub(r"(\([가-힣A-Za-z0-9]+\)|[가-힣A-Za-z0-9]+[).])", "\n", question)
    parts = [part.strip(" ,;:/") for part in re.split(r"\n|[?]|(?:\s+[및]\s+)", cleaned) if part.strip(" ,;:/")]
    return [part for part in parts if len(part) >= 8] or [question]


def calc_off_topic_penalty(
    question_keywords: Set[str],
    matched_keywords: Sequence[str],
    intents: Sequence[str],
    answer: str,
) -> int:
    penalty = 0
    if question_keywords and not matched_keywords:
        penalty += 35
    elif question_keywords and len(matched_keywords) / len(question_keywords) < 0.15:
        penalty += 18

    generic_hits = sum(1 for term in GENERIC_ANSWER_TERMS if term in answer)
    if generic_hits >= 4 and len(matched_keywords) <= 2:
        penalty += 12

    if "weakness" in intents and not any(term in answer for term in QUESTION_INTENTS["weakness"]["answer"]):
        penalty += 18
    if "plan" in intents and not any(term in answer for term in QUESTION_INTENTS["plan"]["answer"]):
        penalty += 18

    return penalty


def build_feedback(score: int, intents: Sequence[str], missing_keywords: Sequence[str], coverage: int) -> List[str]:
    feedback = []
    if score < 60:
        feedback.append("답변 첫 문단에 질문의 핵심어를 직접 다시 받아서 답을 시작하세요.")
    if missing_keywords:
        feedback.append(f"질문 핵심어 중 '{', '.join(missing_keywords[:5])}'가 답변에서 약합니다.")
    if coverage < 80:
        feedback.append("질문이 여러 조건을 요구합니다. 각 조건마다 한 문장 이상 답변을 배치하세요.")
    if "motivation" in intents:
        feedback.append("지원동기 문항은 회사/직무 선택 이유와 본인 경험을 연결해야 합니다.")
    if "plan" in intents:
        feedback.append("입사 후 계획 문항은 실행 계획과 회사에 줄 기여를 함께 써야 합니다.")
    if "weakness" in intents:
        feedback.append("단점 문항은 단점 자체보다 보완 행동과 변화 결과까지 포함해야 합니다.")
    return feedback or ["질문 의도와 답변 내용이 비교적 잘 맞습니다."]


def grade_match(score: int) -> tuple[str, str]:
    if score >= 80:
        return "우수", "질문 의도와 답변 내용이 잘 맞습니다."
    if score >= 60:
        return "보통", "큰 방향은 맞지만 질문의 일부 조건이 약하게 반영되었습니다."
    if score >= 40:
        return "주의", "답변 품질과 별개로 질문에 직접 답하는 힘이 부족합니다."
    return "위험", "질문과 무관한 답변으로 평가될 가능성이 큽니다."


def clamp(value: int, minimum: int = 0, maximum: int = 100) -> int:
    return max(minimum, min(maximum, value))
