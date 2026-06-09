"""
ROLE 04 성능 평가 - sampled_100.csv 기반
사람 라벨(A/B/C/D) vs 모델 예측(A/B/C/D) 비교 후 P/R/F1 산출
"""
import sys, os, csv
sys.path.insert(0, os.path.dirname(__file__))
from self_detector import detect_self_language

# ── 사람이 직접 판단한 Ground Truth 라벨 ──────────────────────
# 기준: A=기여중심 뚜렷(수치·외부성과 지배), B=기여우세+자기표현 일부
#       C=혼재, D=자기중심 지배
GROUND_TRUTH = {
    0: "C",  # 책임감·도움 행동 혼재
    1: "C",  # 생산효율 2% 언급 있으나 자기주어 다수
    2: "D",  # 개인 성장 서술, 수치 없음
    3: "D",  # 열망·의견 위주
    4: "D",  # 뿌듯함·개인 경험 위주
    5: "C",  # 기술 문제해결 + 개인 학습
    6: "D",  # 미래 목표 나열
    7: "D",  # 자신감 획득 위주
    8: "D",  # 개인 포부
    9: "C",  # 아이들 반응 + 개인 서술 혼재
    10: "B", # 설문 방식 개선→실적 달성
    11: "C", # 공연 성공 + 개인 감정 혼재
    12: "D", # 자기 특성 나열
    13: "D", # 미래 계획
    14: "B", # 우수마트 사례 전달·취업추천서 획득
    15: "D", # 미래 계획
    16: "C", # 학부 활동 성과 + 자기주어 다수
    17: "C", # 웹사이트 리뉴얼 제안 + 개인 서술
    18: "D", # 개인 공부
    19: "B", # 색띠 아이디어→혼선 감소·업무 회전율 향상
    20: "B", # 팀플 2위·CSR 본선 등 구체 성과 다수
    21: "D", # 개인 관찰
    22: "B", # 8개국 13명 지원 완료·수혜자 감사 피드백
    23: "D", # 추상적 투자 철학
    24: "D", # 개인 성장 서술
    25: "C", # 헌혈 20회(개인 목표) + 타인 기여 혼재
    26: "B", # 업무 방법 팀 전체 채택·성실 평가
    27: "B", # 기사 실력 향상 평 획득
    28: "B", # LINC 아이디어 상 수상
    29: "C", # 감사 전달 소규모 결과
    30: "C", # 업무 경험 서술 혼재
    31: "D", # 한 문장 자기 소개
    32: "D", # 기사 분석·의견 위주
    33: "D", # 개인 프로젝트, 끈기 강조
    34: "C", # 갈등 해결 서술
    35: "C", # IT 역량 + QA 인턴 학습
    36: "B", # 과장 교육자료·R&D 검토 제안 채택
    37: "C", # 최우수 단원 선정 + 자기중심 서술
    38: "D", # 기술 경험 공유 위주
    39: "B", # 소논문 최고점·총학생회 개선안 채택
    40: "B", # 성과보수 수령·고객 문제 해결
    41: "C", # 개인 학점 달성 + 자기성찰
    42: "C", # VOC 없음·칭찬 글 수령
    43: "C", # 야구 타격 수치 + 선발 등재
    44: "D", # 미래 목표
    45: "C", # 지원금 150만원 + 인간관계 이야기 혼재
    46: "B", # 경진대회 장려상 수상
    47: "D", # 미래 목표
    48: "B", # 해커톤 6팀 중 2등
    49: "D", # 개인 동기 서술
    50: "B", # 전투력 측정 1등
    51: "D", # 개인 변화 서술
    52: "D", # 자격증 준비 과정
    53: "D", # 개인 학습 서술
    54: "C", # 게임 완성·공연 성공 + 개인 서술
    55: "D", # 회사 소개 위주
    56: "C", # 인턴·복수전공 경험 서술
    57: "D", # 제품 시장 분석
    58: "B", # 프로젝트 일정 1주 단축
    59: "D", # 자격증 취득 위주
    60: "D", # 회사 시장 분석
    61: "D", # 자격증 취득 나열
    62: "D", # IoT 산업 분석
    63: "D", # 윤리적 판단 서술
    64: "D", # 어학 학습 서술
    65: "B", # 야구 분석으로 팀 조 1위 + 자격증
    66: "B", # 마케팅 활동 전국 2위·개인 14위
    67: "B", # 팀 과제 1등·상금 500위안
    68: "B", # 방위사업청장상 동상·논문 등재
    69: "C", # 일정 지연 해결 + 팀워크 학습
    70: "D", # 미래 포부
    71: "B", # 시간당 8달러 팁·동료 부러움
    72: "D", # 개인 학습·미래 포부
    73: "D", # 반도체 산업 분석
    74: "D", # 회사 선택 기준 서술
    75: "C", # 서비스 사례 서술
    76: "D", # 개인 반성
    77: "C", # 영어 만점 + C→A 재수강 개인 성취
    78: "B", # 기획 실제 업무 적용
    79: "B", # 에너지 해커톤 대상(1등)
    80: "D", # 회사 정보 습득 위주
    81: "A", # 지게차 동선 1m 확보·작업 1시간 단축 등 구체 수치
    82: "C", # 회사 서비스 개선 제안 서술
    83: "D", # 철학적 에세이
    84: "C", # 무역 실무 경력 + 개인 성장
    85: "B", # 굿즈 판매 61만원 달성
    86: "D", # 인간관계 서술
    87: "B", # 스터디 부원 대다수 목표 달성
    88: "D", # 교육 정책 분석
    89: "D", # 개인 철학 서술
    90: "B", # 교내 대회 동상·특허출원
    91: "D", # 자기 묘사
    92: "D", # 자기 묘사
    93: "D", # 미래 계획
    94: "D", # 짧은 갈등 해결(구체 결과 없음)
    95: "B", # 수거비용 절감·A학점 획득
    96: "C", # 개인 미니 창업(도덕적 문제로 중단)
    97: "B", # 팀 프로젝트 1등·A+ 학점
    98: "B", # 전년대비 120% 매출 달성
    99: "B", # 유가화보 수주 구체 결과
}

# ── CSV 로드 ──────────────────────────────────────────────────
DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "labeled", "role04_self", "sampled_100.csv"
)

def load_data():
    rows = {}
    with open(DATA_PATH, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["idx"])
            rows[idx] = row["answer"]
    return rows

# ── 지표 계산 ─────────────────────────────────────────────────
def compute_metrics(true_labels, pred_labels, target_class):
    tp = sum(t == target_class and p == target_class for t, p in zip(true_labels, pred_labels))
    fp = sum(t != target_class and p == target_class for t, p in zip(true_labels, pred_labels))
    fn = sum(t == target_class and p != target_class for t, p in zip(true_labels, pred_labels))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1, tp, fp, fn

# ── 메인 ─────────────────────────────────────────────────────
def main():
    data = load_data()
    true_labels, pred_labels, details = [], [], []

    print("샘플 예측 중...")
    for idx in sorted(GROUND_TRUTH.keys()):
        text = data.get(idx, "")
        if not text:
            print(f"  [WARN] idx={idx} 텍스트 없음")
            continue
        result = detect_self_language(text)
        pred = result["grade"]
        true = GROUND_TRUTH[idx]
        true_labels.append(true)
        pred_labels.append(pred)
        details.append((idx, true, pred, result["self_score"], result["contribution_score"]))

    total   = len(true_labels)
    correct = sum(t == p for t, p in zip(true_labels, pred_labels))

    print("\n" + "=" * 66)
    print("ROLE 04  자기중심 탐지기  성능 평가  (sampled_100.csv)")
    print(f"테스트셋: {total}개 / 정답: 사람이 직접 라벨링 (A/B/C/D)")
    print("=" * 66)

    print(f"\n[전체 정확도]  {correct}/{total} = {correct/total:.1%}\n")

    classes = ["A", "B", "C", "D"]
    class_label = {"A": "기여 뚜렷", "B": "기여 우세", "C": "혼  재", "D": "자기중심"}

    print(f"{'클래스':<6} {'의미':<10} {'N(정답)':>8} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("-" * 56)
    macro_f1_sum = 0
    for cls in classes:
        p, r, f1, tp, fp, fn = compute_metrics(true_labels, pred_labels, cls)
        n = sum(t == cls for t in true_labels)
        macro_f1_sum += f1
        print(f"  {cls}    {class_label[cls]:<10} {n:>8}   {p:>9.3f}  {r:>7.3f}  {f1:>7.3f}   TP={tp} FP={fp} FN={fn}")

    macro_f1 = macro_f1_sum / len(classes)
    print("-" * 56)
    print(f"  Macro F1 (A/B/C/D)                              {macro_f1:>7.3f}")

    # 정답 분포
    print("\n[정답 라벨 분포]")
    for cls in classes:
        n = sum(t == cls for t in true_labels)
        print(f"  {cls}: {n}개")

    # 예측 분포
    print("\n[예측 라벨 분포]")
    for cls in classes:
        n = sum(p == cls for p in pred_labels)
        print(f"  {cls}: {n}개")

    # 오분류 상세
    errors = [(idx, t, p, ss, cs) for idx, t, p, ss, cs in details if t != p]
    print(f"\n[오분류 케이스] {len(errors)}건")
    print(f"{'idx':>4}  {'정답':>4}  {'예측':>4}  {'self':>5}  {'contrib':>7}")
    print("-" * 32)
    for idx, t, p, ss, cs in errors:
        print(f"{idx:>4}  {t:>4}  {p:>4}  {ss:>5}  {cs:>7}")

if __name__ == "__main__":
    main()
