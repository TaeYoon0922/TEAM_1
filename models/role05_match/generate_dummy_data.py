from __future__ import annotations

import csv
import random
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from xml.sax.saxutils import escape


SEED = 20260527
ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "data" / "labeled" / "role05_match"
OUTPUT_CSV = OUTPUT_DIR / "role05_match_dummy_300.csv"
OUTPUT_XLSX = OUTPUT_DIR / "role05_match_dummy_300.xlsx"


QUESTION_TEMPLATES = [
    {
        "question_type": "motivation_plan",
        "question": "{company} {position} 직무에 지원한 동기와 입사 후 어떤 방식으로 기여할 수 있는지 기술해 주십시오.",
        "requirements": ["지원동기", "직무 이해", "입사 후 기여"],
        "keywords": ["지원", "직무", "기여", "입사 후"],
    },
    {
        "question_type": "collaboration",
        "question": "다른 사람들과 함께 일하며 팀워크를 발휘했던 경험과 그 과정에서 본인이 맡은 역할을 구체적으로 서술해 주십시오.",
        "requirements": ["협업 상황", "본인 역할", "팀 성과"],
        "keywords": ["팀", "협업", "역할", "성과"],
    },
    {
        "question_type": "problem_solving",
        "question": "예상하지 못한 문제를 발견하고 해결했던 경험을 상황, 행동, 결과 중심으로 기술해 주십시오.",
        "requirements": ["문제 상황", "해결 행동", "결과"],
        "keywords": ["문제", "해결", "행동", "결과"],
    },
    {
        "question_type": "competency",
        "question": "{position} 직무와 관련된 본인의 강점 또는 역량을 실제 경험을 바탕으로 설명해 주십시오.",
        "requirements": ["직무 역량", "근거 경험", "활용 가능성"],
        "keywords": ["강점", "역량", "경험", "직무"],
    },
    {
        "question_type": "weakness",
        "question": "본인의 단점 또는 부족했던 점을 쓰고, 이를 보완하기 위해 어떤 노력을 했는지 기술해 주십시오.",
        "requirements": ["단점 인식", "보완 행동", "변화"],
        "keywords": ["단점", "부족", "보완", "개선"],
    },
    {
        "question_type": "growth",
        "question": "본인의 성장과정 또는 가치관이 현재의 진로 선택에 어떤 영향을 주었는지 작성해 주십시오.",
        "requirements": ["성장 배경", "가치관", "진로 연결"],
        "keywords": ["성장", "가치관", "진로", "영향"],
    },
    {
        "question_type": "customer",
        "question": "고객 또는 사용자의 불편을 파악하고 개선했던 경험을 구체적인 결과와 함께 설명해 주십시오.",
        "requirements": ["고객 문제", "개선 행동", "구체적 결과"],
        "keywords": ["고객", "사용자", "개선", "결과"],
    },
    {
        "question_type": "challenge",
        "question": "목표 달성을 위해 도전적으로 시도했던 경험과 그 과정에서 배운 점을 기술해 주십시오.",
        "requirements": ["목표", "도전 행동", "배운 점"],
        "keywords": ["목표", "도전", "시도", "배운"],
    },
    {
        "question_type": "ethics",
        "question": "원칙을 지키기 위해 어려운 선택을 했던 경험과 그 결과를 서술해 주십시오.",
        "requirements": ["원칙", "어려운 선택", "결과"],
        "keywords": ["원칙", "선택", "결과", "신뢰"],
    },
    {
        "question_type": "learning",
        "question": "지원 분야와 관련된 전문성을 키우기 위해 학습하거나 실습했던 경험을 구체적으로 기술해 주십시오.",
        "requirements": ["전문성", "학습 과정", "적용 결과"],
        "keywords": ["전문성", "학습", "실습", "적용"],
    },
]

COMPANIES = [
    "한국수자원공사",
    "롯데월드",
    "한국남동발전",
    "CJ올리브네트웍스",
    "현대자동차",
    "신한은행",
    "네이버",
    "LG전자",
    "한화솔루션",
    "카카오페이",
]

POSITIONS = [
    "데이터 분석",
    "시설관리",
    "전기·전자엔지니어",
    "서비스 기획",
    "품질관리",
    "영업관리",
    "백엔드 개발",
    "환경안전",
    "마케팅",
    "인사",
]

PROJECTS = [
    "민원 처리 프로세스 개선",
    "동아리 예산 관리 자동화",
    "센서 불량 원인 분석",
    "고객 상담 FAQ 재정비",
    "재고 데이터 대시보드 구축",
    "신입 교육 자료 표준화",
    "현장 안전 점검표 개편",
    "사용자 이탈 원인 분석",
    "캠페인 성과 리포트 개선",
    "스터디 운영 방식 개선",
]

METRICS = [
    "처리 시간을 28% 줄였습니다",
    "오류 건수를 월 15건에서 4건으로 낮췄습니다",
    "참여율을 62%에서 84%로 높였습니다",
    "재작업 시간을 주 6시간 절감했습니다",
    "고객 문의 응답 시간을 평균 2일에서 6시간으로 단축했습니다",
    "점검 누락률을 18%에서 3%로 낮췄습니다",
    "보고서 작성 시간을 40분에서 15분으로 줄였습니다",
    "만족도 점수를 3.8점에서 4.5점으로 높였습니다",
]

WEAKNESSES = [
    "초기에는 세부 일정까지 혼자 관리하려는 경향",
    "발표 전 긴장으로 핵심 메시지를 길게 설명하는 습관",
    "완성도를 높이려다 검토 요청이 늦어지는 점",
    "낯선 도구를 사용할 때 초반 학습 시간이 오래 걸리는 점",
]

OFF_TOPIC_TOPICS = [
    "어릴 때부터 운동을 좋아해 주말마다 축구를 했습니다. 체력을 기르며 성실함을 배웠고 앞으로도 밝은 태도로 생활하겠습니다.",
    "해외여행 중 길을 잃었지만 현지인에게 질문하며 목적지를 찾았습니다. 이 경험으로 낯선 환경에서도 당황하지 않는 법을 배웠습니다.",
    "저는 음악 감상을 좋아합니다. 다양한 장르를 들으며 스트레스를 관리했고 긍정적인 마음가짐을 유지할 수 있었습니다.",
    "학창 시절 반장을 맡아 친구들과 즐겁게 지냈습니다. 사람들과 친해지는 성격은 저의 가장 큰 장점이라고 생각합니다.",
    "독서를 통해 인문학적 소양을 키웠습니다. 책을 읽으며 세상을 넓게 보는 태도를 배웠고 꾸준히 성장하고 있습니다.",
]


def build_excellent_answer(q: dict, company: str, position: str, project: str, metric: str) -> str:
    req = q["requirements"]
    question_type = q["question_type"]
    if question_type == "motivation_plan":
        return (
            f"저는 {position} 직무에서 실제 문제를 데이터와 현장 관찰로 연결해 해결하는 점에 끌려 {company}에 지원했습니다. "
            f"이전에 {project}를 진행하며 문제 원인을 정리하고, 이해관계자 인터뷰와 로그 분석을 통해 실행 우선순위를 세웠습니다. "
            f"이후 제가 맡은 역할은 실행안을 설계하고 주 단위로 결과를 검증하는 것이었으며, 그 과정에서 {metric}. "
            f"입사 후에도 이 경험을 바탕으로 {company}의 {position} 업무에서 고객과 조직에 측정 가능한 기여를 만들겠습니다."
        )
    if question_type == "collaboration":
        return (
            f"{project}를 진행할 때 일정 지연으로 팀원 간 역할이 겹치는 문제가 있었습니다. "
            f"저는 회의록을 기준으로 업무를 다시 나누고, 매일 10분씩 진행 상황을 공유하는 방식을 제안했습니다. "
            f"그 결과 제 역할이었던 자료 정리와 일정 조율이 안정화되었고 {metric}. "
            f"이 경험을 통해 팀워크는 좋은 분위기보다 역할과 책임을 명확히 하는 데서 시작된다는 점을 배웠습니다."
        )
    if question_type == "problem_solving":
        return (
            f"{project} 과정에서 예상하지 못한 문제를 발견한 상황이 있었습니다. "
            f"저는 해결 행동으로 오류 발생 시간, 담당 단계, 입력 데이터를 표로 정리해 병목 지점을 찾았습니다. "
            f"이후 체크리스트와 검수 기준을 새로 적용했고, 결과적으로 {metric}. "
            f"문제를 감으로 판단하지 않고 데이터로 좁힌 점이 가장 큰 성과였습니다."
        )
    if question_type == "competency":
        return (
            f"제 강점은 {position} 직무에 필요한 문제 구조화와 실행력입니다. "
            f"{project}에서 저는 요구사항을 기능, 일정, 리스크로 나누어 우선순위를 정했습니다. "
            f"그 뒤 핵심 과제부터 실행하고 결과를 매주 검토해 {metric}. "
            f"이 역량은 {company}에서도 복잡한 업무를 실행 가능한 단위로 바꾸는 데 활용될 수 있습니다."
        )
    if question_type == "weakness":
        return (
            f"제 단점은 중요한 일을 맡으면 초기에 혼자 해결하려는 경향이 있었다는 점입니다. "
            f"{project}를 하며 이 방식이 검토 지연으로 이어진다는 것을 깨닫고, 중간 공유 시점을 일정표에 고정했습니다. "
            f"이후 팀원 피드백을 먼저 반영하면서 {metric}. "
            f"지금은 완성도보다 빠른 공유와 보완을 우선하는 방식으로 개선하고 있습니다."
        )
    if question_type == "growth":
        return (
            f"저는 성장과정에서 맡은 일을 끝까지 책임지는 태도를 중요하게 배웠습니다. "
            f"그 가치관은 {project}를 수행할 때도 이어져, 어려운 상황에서도 원인을 기록하고 해결책을 끝까지 검증하게 했습니다. "
            f"그 결과 {metric}. "
            f"이 경험은 현재의 진로 선택에 영향을 주었고, {position} 직무에서도 책임 있는 실행으로 이어가겠습니다."
        )
    if question_type == "customer":
        return (
            f"{project} 중 고객 또는 사용자가 같은 내용을 반복 문의하는 불편을 발견했습니다. "
            f"저는 고객 문제를 파악하기 위해 문의 유형을 분류하고, 가장 많은 질문부터 답변 흐름과 안내 문구를 개선했습니다. "
            f"구체적 결과로 {metric}. "
            f"사용자의 불편을 실제 행동 데이터로 확인하고 바꿨다는 점에서 {position} 직무와도 맞닿아 있습니다."
        )
    if question_type == "challenge":
        return (
            f"저는 목표 달성을 위해 {project}에서 기존 방식보다 높은 기준을 세우고 도전적으로 시도했습니다. "
            f"처음에는 반대 의견이 있었지만 작은 범위로 먼저 실험하고 결과를 공유했습니다. "
            f"그 결과 {metric}. "
            f"이 과정에서 배운 점은 도전이 무작정 밀어붙이는 것이 아니라 근거와 검증 단계를 함께 설계하는 일이라는 것입니다."
        )
    if question_type == "ethics":
        return (
            f"{project}를 맡았을 때 일정 단축을 위해 검수 단계를 생략하자는 의견이 있었습니다. "
            f"저는 단기 성과보다 원칙과 신뢰가 중요하다고 판단해 최소 검수 항목을 남기는 대안을 제시했습니다. "
            f"그 결과 일정은 유지하면서도 {metric}. "
            f"이 경험을 통해 원칙을 지키는 선택이 장기적으로 조직의 신뢰를 높인다는 것을 배웠습니다."
        )
    if question_type == "learning":
        return (
            f"{position} 분야의 전문성을 키우기 위해 {project}를 직접 실습 주제로 삼았습니다. "
            f"관련 자료를 학습한 뒤 작은 기능부터 구현하고, 결과를 기록하며 부족한 개념을 다시 보완했습니다. "
            f"그 결과 {metric}. "
            f"학습을 결과물로 연결한 경험을 바탕으로 {company}에서도 빠르게 실무 지식을 흡수하겠습니다."
        )
    return (
        f"저는 {position} 직무에서 실제 문제를 데이터와 현장 관찰로 연결해 해결하는 점에 끌려 {company}에 지원했습니다. "
        f"이전에 {project}를 진행하며 {req[0]}을 먼저 정리하고, 이해관계자 인터뷰와 로그 분석을 통해 핵심 원인을 좁혔습니다. "
        f"이후 제가 맡은 역할은 실행안을 설계하고 주 단위로 결과를 검증하는 것이었으며, 그 과정에서 {metric}. "
        f"입사 후에도 이 경험을 바탕으로 {company}의 {position} 업무에서 고객과 조직에 측정 가능한 기여를 만들겠습니다."
    )


def build_medium_answer(q: dict, company: str, position: str, project: str, metric: str) -> str:
    req = q["requirements"]
    return (
        f"{position} 직무에 관심이 있어 {company}에 지원했습니다. "
        f"대학 시절 {project}를 하면서 {req[0]}의 중요성을 느꼈고 팀원들과 함께 문제를 해결하려고 노력했습니다. "
        f"구체적인 과정에서 여러 의견을 듣고 자료를 정리했으며 일부 개선 효과도 있었습니다. "
        f"다만 결과는 {metric} 정도로 정리할 수 있고, 앞으로 관련 역량을 더 키워 회사에 도움이 되고 싶습니다."
    )


def build_off_topic_answer(q: dict, company: str, position: str, project: str, metric: str, topic: str) -> str:
    return (
        f"{topic} "
        f"또한 저는 맡은 일을 끝까지 하려는 책임감이 있습니다. "
        f"앞으로 {company}에서도 누구보다 성실하게 일하며 좋은 구성원이 되겠습니다."
    )


def score_for(label: str, row_index: int) -> int:
    rng = random.Random(SEED + row_index)
    if label == "EXCELLENT":
        return rng.randint(86, 98)
    if label == "MEDIUM":
        return rng.randint(52, 74)
    return rng.randint(5, 34)


def generate_rows() -> list[dict]:
    random.seed(SEED)
    rows = []
    buckets = [
        ("EXCELLENT", "엄청 잘 답변", 100),
        ("MEDIUM", "중간 정도 답변", 100),
        ("OFF_TOPIC", "질문과 어긋난 답변", 100),
    ]

    row_no = 1
    for label, bucket_name, count in buckets:
        for i in range(count):
            q = QUESTION_TEMPLATES[(row_no - 1) % len(QUESTION_TEMPLATES)]
            company = COMPANIES[(row_no + i * 3) % len(COMPANIES)]
            position = POSITIONS[(row_no * 2 + i) % len(POSITIONS)]
            project = PROJECTS[(row_no + i * 5) % len(PROJECTS)]
            metric = METRICS[(row_no + i * 7) % len(METRICS)]
            weakness = WEAKNESSES[(row_no + i) % len(WEAKNESSES)]
            question = q["question"].format(company=company, position=position)

            if label == "EXCELLENT":
                answer = build_excellent_answer(q, company, position, project, metric)
                mismatch_reason = "질문 핵심 조건을 모두 반영하고 직무/회사 연결과 구체적 결과가 있음"
                covered = q["requirements"]
            elif label == "MEDIUM":
                answer = build_medium_answer(q, company, position, project, metric)
                if q["question_type"] == "weakness":
                    answer = (
                        f"저의 보완점은 {weakness}입니다. 이를 줄이기 위해 체크리스트를 만들고 주변 피드백을 받았습니다. "
                        "아직 결과를 수치로 설명하기는 어렵지만 같은 실수를 반복하지 않으려 노력하고 있습니다."
                    )
                mismatch_reason = "문항 방향은 맞지만 요구 조건 일부가 추상적이거나 결과/직무 연결이 약함"
                covered = q["requirements"][:2]
            else:
                answer = build_off_topic_answer(
                    q,
                    company,
                    position,
                    project,
                    metric,
                    OFF_TOPIC_TOPICS[(row_no + i) % len(OFF_TOPIC_TOPICS)],
                )
                mismatch_reason = "문장은 자연스럽지만 질문 핵심어와 요구 조건을 거의 반영하지 않음"
                covered = []

            rows.append(
                {
                    "sample_id": f"MATCH{row_no:03d}",
                    "source": "synthetic_from_direct_labeling_200_style",
                    "company": company,
                    "position": position,
                    "question_no": (i % 5) + 1,
                    "question_type": q["question_type"],
                    "question": question,
                    "answer": answer,
                    "quality_bucket": bucket_name,
                    "manual_label": label,
                    "match_score_0_100": score_for(label, row_no),
                    "question_requirements": " | ".join(q["requirements"]),
                    "covered_requirements": " | ".join(covered),
                    "expected_keywords": " | ".join(q["keywords"]),
                    "mismatch_reason": mismatch_reason,
                    "notes": "ROLE05_MATCH 질문-답변 적합도 모델용 더미데이터",
                }
            )
            row_no += 1
    return rows


def write_csv(rows: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_xlsx(rows: list[dict]) -> None:
    """Write a simple single-sheet XLSX without optional Excel dependencies."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    matrix = [fieldnames] + [[row[name] for name in fieldnames] for row in rows]

    sheet_xml = build_sheet_xml(matrix)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    files = {
        "[Content_Types].xml": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>""",
        "_rels/.rels": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>""",
        "xl/_rels/workbook.xml.rels": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>""",
        "xl/workbook.xml": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="role05_match_dummy_300" sheetId="1" r:id="rId1"/>
  </sheets>
</workbook>""",
        "xl/styles.xml": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="2">
    <font><sz val="11"/><color theme="1"/><name val="Calibri"/><family val="2"/></font>
    <font><b/><sz val="11"/><color rgb="FFFFFFFF"/><name val="Calibri"/><family val="2"/></font>
  </fonts>
  <fills count="3">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF0F766E"/><bgColor indexed="64"/></patternFill></fill>
  </fills>
  <borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="2">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1"/>
  </cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>""",
        "xl/worksheets/sheet1.xml": sheet_xml,
        "docProps/core.xml": f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:creator>Codex</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>""",
        "docProps/app.xml": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex</Application>
  <DocSecurity>0</DocSecurity>
  <ScaleCrop>false</ScaleCrop>
  <HeadingPairs><vt:vector size="2" baseType="variant"><vt:variant><vt:lpstr>Worksheets</vt:lpstr></vt:variant><vt:variant><vt:i4>1</vt:i4></vt:variant></vt:vector></HeadingPairs>
  <TitlesOfParts><vt:vector size="1" baseType="lpstr"><vt:lpstr>role05_match_dummy_300</vt:lpstr></vt:vector></TitlesOfParts>
</Properties>""",
    }

    with ZipFile(OUTPUT_XLSX, "w", ZIP_DEFLATED) as z:
        for name, content in files.items():
            z.writestr(name, content)


def build_sheet_xml(matrix: list[list[object]]) -> str:
    col_widths = {
        1: 14,
        2: 34,
        3: 18,
        4: 22,
        5: 12,
        6: 20,
        7: 70,
        8: 100,
        9: 18,
        10: 14,
        11: 16,
        12: 36,
        13: 36,
        14: 28,
        15: 52,
        16: 36,
    }
    cols = "".join(
        f'<col min="{idx}" max="{idx}" width="{width}" customWidth="1"/>'
        for idx, width in col_widths.items()
    )
    rows_xml = []
    for row_idx, row in enumerate(matrix, start=1):
        cells = []
        for col_idx, value in enumerate(row, start=1):
            ref = f"{column_name(col_idx)}{row_idx}"
            style = ' s="1"' if row_idx == 1 else ""
            if isinstance(value, int):
                cells.append(f'<c r="{ref}"{style}><v>{value}</v></c>')
            else:
                text = escape(str(value), {'"': "&quot;"})
                cells.append(f'<c r="{ref}" t="inlineStr"{style}><is><t>{text}</t></is></c>')
        rows_xml.append(f'<row r="{row_idx}">{"".join(cells)}</row>')
    last_ref = f"{column_name(len(matrix[0]))}{len(matrix)}"
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <dimension ref="A1:{last_ref}"/>
  <sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>
  <sheetFormatPr defaultRowHeight="15"/>
  <cols>{cols}</cols>
  <sheetData>{"".join(rows_xml)}</sheetData>
  <autoFilter ref="A1:{last_ref}"/>
</worksheet>"""


def column_name(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def main() -> None:
    rows = generate_rows()
    write_csv(rows)
    write_xlsx(rows)
    print(f"wrote {len(rows)} rows to {OUTPUT_CSV}")
    print(f"wrote {len(rows)} rows to {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()
