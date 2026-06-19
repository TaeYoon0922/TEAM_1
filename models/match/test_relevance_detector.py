from relevance_detector import detect_answer_relevance


def test_relevant_answer_scores_higher_than_irrelevant_answer():
    question = "지원동기와 입사 후 회사에 어떻게 기여할 수 있는지 기술해 주세요."
    relevant = (
        "데이터 분석 직무에 지원한 이유는 고객 행동 데이터를 통해 실제 매출 개선에 기여하고 싶기 때문입니다. "
        "입사 후에는 이탈 고객 분석 모델을 만들고 캠페인 성과를 개선하겠습니다."
    )
    irrelevant = (
        "대학교 축제에서 부스를 운영하며 팀원들과 협업했고 3일 동안 방문객 500명을 응대했습니다. "
        "그 결과 운영 경험을 쌓았습니다."
    )

    relevant_result = detect_answer_relevance(question, relevant)
    irrelevant_result = detect_answer_relevance(question, irrelevant)

    assert relevant_result["score"] > irrelevant_result["score"]
    assert relevant_result["score"] >= 60
    assert irrelevant_result["score"] < 60


def test_subquestion_feedback_for_partial_answer():
    question = "조직 내에서 변화를 주도했던 경험을 쓰고, 그 변화가 회사에서 어떻게 활용될 수 있는지 기술해 주세요."
    answer = "동아리에서 회의 방식을 개선해 회의 시간을 30% 줄인 경험이 있습니다."

    result = detect_answer_relevance(question, answer)

    assert result["subquestion_coverage"] < 1.0
    assert any("조건" in item for item in result["feedback_items"])


if __name__ == "__main__":
    test_relevant_answer_scores_higher_than_irrelevant_answer()
    test_subquestion_feedback_for_partial_answer()
    print("match relevance tests passed")
