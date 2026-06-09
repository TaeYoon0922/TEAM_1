# ROLE 04 성능 평가 스크립트
# 방식: 사람이 정답 라벨을 붙인 테스트셋 → 모델 출력과 비교 → F1 계산
#
# 라벨 정의
#   self         : 자기중심 표현이 지배적 (모델 grade D)
#   contribution : 기여중심 표현이 지배적 (모델 grade A 또는 B)
#   mixed        : 혼재 (모델 grade C)

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from self_detector import detect_self_language

# ── 테스트셋 (정답 라벨 포함) ────────────────────────────────
# 각 항목: (텍스트, 정답_라벨)
# 정답은 문장을 읽은 사람이 상식적으로 판단한 값
TEST_SET = [
    # ── 자기중심 (self) ──────────────────────────────────────
    ("저는 이 경험을 통해 정말 많은 것을 느꼈습니다.", "self"),
    ("저는 이 일을 하면서 진정한 책임감을 배웠습니다.", "self"),
    ("저는 힘들었지만 성장할 수 있었습니다.", "self"),
    ("저는 열정을 가지고 매 순간 최선을 다했습니다.", "self"),
    ("나는 이 과정에서 깨달았습니다. 포기하지 않는 것이 중요하다는 것을.", "self"),
    ("저는 뿌듯함을 느꼈고, 이 경험이 저를 성장시켰습니다.", "self"),
    ("저의 끈기와 도전정신이 이 일을 가능하게 했습니다.", "self"),
    ("본인은 이 프로젝트를 통해 내면의 가치관을 정립하게 됐습니다.", "self"),
    ("저는 매일 자기계발을 위해 노력해왔습니다.", "self"),
    ("저는 이 경험이 저의 의지를 더욱 단단하게 만들었다고 생각합니다.", "self"),

    # ── 기여중심 (contribution) ──────────────────────────────
    ("팀 성과를 30% 향상시켰습니다.", "contribution"),
    ("API 응답 속도를 기존 대비 2배 단축했습니다.", "contribution"),
    ("고객 불만 건수를 3개월 만에 40% 감소시켰습니다.", "contribution"),
    ("시스템 오류율을 0.5%에서 0.1%로 개선했습니다.", "contribution"),
    ("팀원 5명과 협업하여 프로젝트를 2주 앞당겨 완료했습니다.", "contribution"),
    ("데이터 처리 파이프라인을 구현해 분석 시간을 20시간에서 3시간으로 단축했습니다.", "contribution"),
    ("고객 재구매율이 15% 증가했습니다.", "contribution"),
    ("서비스가 안정적으로 배포되어 월간 이용자 수가 1만 명을 달성했습니다.", "contribution"),
    ("코드 리팩토링을 주도하여 유지보수 비용을 연간 500만 원 절감했습니다.", "contribution"),
    ("부서 전체 업무 효율이 25% 향상됐습니다.", "contribution"),

    # ── 혼합 (mixed) ─────────────────────────────────────────
    ("저는 열심히 노력한 결과 팀 성과를 20% 개선했습니다.", "mixed"),
    ("저는 이 경험을 통해 성장했고, 그 결과 프로젝트를 성공적으로 완료했습니다.", "mixed"),
    ("저는 책임감을 갖고 임한 덕분에 오류율을 30% 낮출 수 있었습니다.", "mixed"),
    ("저는 어려움을 느꼈지만 팀과 협력하여 기한 내 배포를 달성했습니다.", "mixed"),
    ("본인의 끈기 덕분에 시스템 응답 시간을 2배 단축할 수 있었습니다.", "mixed"),
    ("저는 이 프로젝트를 주도했고, 팀 만족도를 4.8점으로 끌어올렸습니다.", "mixed"),
    ("저는 배우면서 성장했고, 그 과정에서 서비스 품질을 개선했습니다.", "mixed"),
    ("저의 도전정신으로 새 기능을 구현해 고객 이탈률을 10% 낮췄습니다.", "mixed"),
    ("저는 힘든 순간도 있었지만 결국 3개 팀의 협업을 성공적으로 조율했습니다.", "mixed"),
    ("본인은 이 경험에서 많은 것을 느꼈으며, 결과적으로 매출 15% 증가에 기여했습니다.", "mixed"),
]


# ── 모델 예측 ────────────────────────────────────────────────

def predict_label(text: str) -> str:
    """grade → 3분류 라벨로 변환"""
    r = detect_self_language(text)
    grade = r["grade"]
    if grade == "D":
        return "self"
    elif grade in ("A", "B"):
        return "contribution"
    else:  # C
        return "mixed"


# ── 지표 계산 ────────────────────────────────────────────────

def compute_metrics(true_labels, pred_labels, target_class):
    """이진 분류 기준 Precision / Recall / F1"""
    tp = sum(t == target_class and p == target_class for t, p in zip(true_labels, pred_labels))
    fp = sum(t != target_class and p == target_class for t, p in zip(true_labels, pred_labels))
    fn = sum(t == target_class and p != target_class for t, p in zip(true_labels, pred_labels))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def macro_f1(true_labels, pred_labels, classes):
    f1s = []
    for cls in classes:
        _, _, f1 = compute_metrics(true_labels, pred_labels, cls)
        f1s.append(f1)
    return sum(f1s) / len(f1s)


# ── 메인 ─────────────────────────────────────────────────────

def main():
    true_labels, pred_labels = [], []
    errors = []

    for text, true in TEST_SET:
        pred = predict_label(text)
        true_labels.append(true)
        pred_labels.append(pred)
        if pred != true:
            errors.append((true, pred, text))

    classes = ["self", "contribution", "mixed"]
    total = len(TEST_SET)
    correct = sum(t == p for t, p in zip(true_labels, pred_labels))

    print("=" * 60)
    print("ROLE 04  자기중심 탐지기  성능 평가")
    print(f"테스트셋: {total}개 (각 라벨 10개)")
    print("=" * 60)

    print(f"\n[전체 정확도]  {correct}/{total} = {correct/total:.1%}\n")

    print(f"{'클래스':<14} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 46)
    for cls in classes:
        p, r, f1 = compute_metrics(true_labels, pred_labels, cls)
        cnt = sum(t == cls for t in true_labels)
        print(f"{cls:<14} {p:>10.3f} {r:>10.3f} {f1:>10.3f}   (N={cnt})")

    mf1 = macro_f1(true_labels, pred_labels, classes)
    print("-" * 46)
    print(f"{'Macro F1':<14} {'':>10} {'':>10} {mf1:>10.3f}")

    print(f"\n[오분류 케이스] {len(errors)}건")
    for true, pred, text in errors:
        print(f"  정답={true:<14} 예측={pred:<14} | {text[:40]}...")

    print("\n[평가 방법 주석]")
    print("  · 테스트셋: 사람이 직접 판단한 30개 문장 (자기중심 10 / 기여중심 10 / 혼합 10)")
    print("  · grade D -> self, grade A/B -> contribution, grade C -> mixed 로 매핑")
    print("  · 합성 테스트셋이므로 실제 자소서 분포를 완전히 반영하지 않음")


if __name__ == "__main__":
    main()
