LABEL_DIST = {
    "A (기여 중심 뚜렷)": 1,
    "B (기여 우세)":      29,
    "C (혼재)":           25,
    "D (자기중심 지배)":  45,
}




RESULTS = {
    "A": {"precision": 0.000, "recall": 0.000, "f1": 0.000, "n": 1},
    "B": {"precision": 0.614, "recall": 0.931, "f1": 0.740, "n": 29},
    "C": {"precision": 0.263, "recall": 0.200, "f1": 0.227, "n": 25},
    "D": {"precision": 0.757, "recall": 0.622, "f1": 0.683, "n": 45},
}

MACRO_F1 = 0.412
ACCURACY = 0.60




IMPROVEMENT = {
    "초기 (all-C 예측)": {"macro_f1": 0.100, "accuracy": 0.24},
    "최종":              {"macro_f1": 0.412, "accuracy": 0.60},
    "개선 배율":         "Macro F1 4.1×, 정확도 2.5×",
}




LIMITATIONS = [

    "A 클래스 샘플 1개 → 임계값 튜닝 불가, F1=0.000 고정",
    "100개 소규모 평가셋 → 통계적 신뢰도 제한",
    "단일 라벨러(1인) → 주관적 판단 개입 가능",


    "주어 추적 없음 → 회사·산업 통계 수치를 개인 성과로 오탐 (idx 73 사례)",
    "어미 변이 커버리지 한계 → regex fallback이 모든 활용형을 포착 못함",
    "B↔C 경계 모호 → C F1=0.227로 낮음 (기여 표현이 적은 B 텍스트 오분류)",


    "주어 추적 NLP 적용 시 3인칭 성과 구별 가능",
    "더 많은 A 샘플 확보 시 A 임계값 정밀 조정 가능",
    "BERT 기반 시퀀스 분류로 문맥·주어를 함께 고려하면 정확도 추가 향상 기대",
]





def print_summary():
    print("=" * 55)
    print("  SELF 자기중심 탐지기 - 성능 평가 요약")
    print("=" * 55)
    print(f"\n  정확도(Accuracy) : {ACCURACY:.0%}")
    print(f"  Macro F1         : {MACRO_F1:.3f}  (초기 0.100 대비 4.1×)\n")
    print(f"  {'등급':<4}  {'P':>6}  {'R':>6}  {'F1':>6}  {'n':>4}")
    print("  " + "-" * 33)
    for g, m in RESULTS.items():
        print(f"  {g:<4}  {m['precision']:>6.3f}  {m['recall']:>6.3f}  {m['f1']:>6.3f}  {m['n']:>4}")
    print()
    print("  [한계점]")
    for lim in LIMITATIONS:
        prefix = "  →" if "기대" in lim or "가능" in lim else "  ·"
        print(f"{prefix} {lim}")
    print("=" * 55)


if __name__ == "__main__":
    print_summary()
