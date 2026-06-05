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
