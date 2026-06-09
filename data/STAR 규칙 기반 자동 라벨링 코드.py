# -*- coding: utf-8 -*-
"""
JobKorea STAR rule-based auto-labeling pipeline

Input:
  1) 원본 잡코리아 CSV
  2) STAR_최종_통합_라벨링.xlsx: 200개 수동/확정 라벨 파일

Output:
  문장 단위 STAR 자동 라벨링 CSV

주의:
  - 수동 200개와 정확히 매칭되는 문장은 수동 라벨을 우선 적용합니다.
  - 나머지는 규칙 기반으로 S/T/A/R/X 및 멀티라벨을 부여합니다.
  - XLSX 읽기는 artifact_tool을 사용합니다.
"""

import argparse
import csv
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from artifact_tool import Blob, SpreadsheetFile

LABEL_ORDER = ["S", "T", "A", "R"]


def normalize_label(label):
    if label is None:
        return "X"
    s = str(label).strip().upper().replace(" ", "")
    if not s or s in {"NONE", "NAN", "NULL"}:
        return "X"
    s = s.replace("/", "+").replace(",", "+")
    if "+" not in s and len(s) > 1 and all(ch in "STARX" for ch in s):
        s = "+".join([ch for ch in LABEL_ORDER if ch in s])
    parts = []
    for p in s.split("+"):
        if p in LABEL_ORDER and p not in parts:
            parts.append(p)
    return "+".join(parts) if parts else "X"


def set_to_label(labels):
    return "+".join([x for x in LABEL_ORDER if x in labels]) if labels else "X"


def norm_text(s):
    s = "" if s is None else str(s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[\"'“”‘’`]", "", s)
    s = re.sub(r"(좋은점|아쉬운점)\s*\d+", "", s)
    return re.sub(r"\s+", " ", s).strip()


def norm_key(s):
    s = norm_text(s).lower()
    return re.sub(r"[\W_]+", "", s)


def load_manual_rows(manual_xlsx, sheet_name="통합_최종", max_rows=201):
    wb = SpreadsheetFile.import_xlsx(Blob.load(str(manual_xlsx)))
    headers = None
    rows = []
    for start in range(1, max_rows + 1, 10):
        end = min(start + 9, max_rows)
        ndjson = wb.inspect({
            "kind": "table",
            "range": f"{sheet_name}!A{start}:O{end}",
            "include": "values",
            "table_max_rows": 20,
            "table_max_cols": 15,
        }).ndjson
        obj = json.loads(ndjson.strip().splitlines()[0])
        vals = obj.get("values", [])
        if not vals:
            continue
        if start == 1:
            headers = vals[0]
            data_rows = vals[1:]
        else:
            data_rows = vals
        for row in data_rows:
            row = row + [None] * (len(headers) - len(row))
            rows.append(dict(zip(headers, row)))
    return rows


def build_manual_maps(manual_rows):
    by_comp_q_sent = {}
    by_sent = defaultdict(list)
    for r in manual_rows:
        comp = norm_key(r.get("회사"))
        q = norm_key(r.get("질문"))
        sent = norm_key(r.get("문장"))
        label = normalize_label(r.get("최종_라벨"))
        mid = r.get("문장ID") or ""
        if sent:
            by_comp_q_sent[(comp, q, sent)] = (label, mid)
            by_sent[sent].append((label, mid, comp, q))

    by_sent_unique = {}
    for sent, vals in by_sent.items():
        if len(set(v[0] for v in vals)) == 1:
            by_sent_unique[sent] = (vals[0][0], vals[0][1])
    return by_comp_q_sent, by_sent_unique


SETTING = r"(당시|그때|처음|초기|과거|고등학생|대학생|대학교|대학|학부|학기|수업|강의|실습|인턴|아르바이트|동아리|대외활동|공모전|프로젝트|팀\s*프로젝트|현장실습|연구실|군대|봉사|대회|축제|캠프|교육|과정|활동|회사|부서|매장|고객|팀원|조원)"
PROBLEM = r"(상황|환경|배경|계기|경험|사례|문제|이슈|갈등|어려움|난관|위기|부족|미흡|한계|실패|오류|불편|민원|요청|잡음|차질|불화|혼란|원인|이유|관계)"
TASK = r"(목표|목적|과제|임무|미션|역할|직무|담당|책임|업무|분담|요구사항|제출|해결해야|수행해야|완수해야|해야\s*했|해야\s*하는|해야겠|맡게\s*되|맡았|맡아|배정|지시|계획|포부|입사\s*후|앞으로|향후)"
ACTION_N = r"(분석|조사|검색|확인|정리|작성|제작|개발|구현|설계|기획|제안|발표|준비|수집|학습|공부|연습|훈련|참여|진행|운영|관리|협의|소통|공유|설득|요청|도입|적용|활용|개선|보완|수정|변경|해결|극복|시도|실행|실천|노력|도전|계획|분담|피드백|테스트|검토|비교|측정|기록|방문|인터뷰|회의|교육|재정비|조율|협업|의견|자료|보고서|문서|코드|모델|시스템|프로그램|서비스|방안|방법|리뷰|처리|관찰|수행|파악|체험|도출|운용|홍보|판매|상담|평가|분류|예측|추정|제시|연구|실험)"
ACTION_V = r"(하였|했|했습니다|하였습니다|진행했|진행하였|제안했|제안하였|제작했|제작하였|개발했|개발하였|작성했|작성하였|조사했|조사하였|분석했|분석하였|수행했|수행하였|노력했|노력하였|참여했|참여하였|준비했|준비하였|맡았|활용했|활용하였|도입했|도입하였|적용했|적용하였|개선했|개선하였|해결했|해결하였|정리했|정리하였|확인했|확인하였|관리했|관리하였|계획했|계획하였|실행했|실행하였|바꾸|찾아|찾았|만들|완성했|완료했|배우기\s*위해|공부하였|공부했|설득|공유|리뷰했|처리했|처리하였|관찰해|수행하여|파악하고|도출|판매|상담|제시|연구|실험)"
RESULT_MARK = r"(결과|성과|성과로|그\s*결과|이를\s*통해|이를\s*계기로|덕분에|마침내|최종적으로|결국|이후|그\s*후|덕에)"
RESULT_N = r"(달성|성공|수상|입상|합격|선정|채택|인정|인정\s*받|완성|완료|해결|향상|상승|증가|감소|개선|절감|최소화|높였|낮췄|줄였|늘렸|얻을\s*수|배울\s*수|배웠|깨닫|깨달|느꼈|알게\s*되|발견|성장|기여|만족|호평|칭찬|신뢰|관심|효과|효율|획득|길렀|인기상|우수상|최우수상)"
NUM_RESULT = r"(\d+\s*(?:%|퍼센트|점|등|등급|위|명|회|건|개|원|만원|억원|시간|개월|년)|전년\s*대비|목표\s*대비|만점|최고|최우수|우수|1등|일등|대폭)"
NOISE_PATTERNS = [r"^(?:좋은점|아쉬운점|글자수|byte|bytes?)\s*\d*", r"^[\W_]+$"]


def has(pattern, text):
    return re.search(pattern, text) is not None


def is_noise_sentence(sentence):
    t = re.sub(r"\s+", " ", str(sentence or "")).strip()
    if len(t) < 5:
        return True
    return any(re.search(p, t, flags=re.I) for p in NOISE_PATTERNS)


def rule_label(sentence, before=None, after=None, question=None):
    s = re.sub(r"\s+", " ", str(sentence or "")).strip()
    if is_noise_sentence(s):
        return "X", 0.55, "noise_or_too_short"

    s2 = re.sub(r"(좋은점|아쉬운점)\s*\d+", " ", s)
    hits = []
    labels = set()

    generic = has(r"(라고|이라고)\s*생각|의미|중요|필요|기업|회사|산업|서비스|기술|사회|미래|비전", s2) and not has(
        r"(저는|제가|저희|우리|당시|프로젝트|경험|결과|성과|목표|역할|해결|수행|노력|분석|개발|제작)", s2
    )
    future = has(r"(입사\s*후|앞으로|향후|미래|하고\s*싶|기여하고\s*싶|성장하고\s*싶|하겠습니다|수행하겠습니다|되겠습니다|만들겠습니다|힘쓰겠습니다)", s2)

    if has(PROBLEM, s2) and (has(r"(있었|없었|발생|생기|느끼|처음|당시|때|경험|프로젝트|수업|활동)", s2) or len(s2) < 180):
        labels.add("S"); hits.append("S:배경/상황/문제")
    elif has(SETTING, s2) and has(r"(있었|했습니다|하였습니다|하던|했던|하게\s*되|때|당시|경험|사례|시작)", s2):
        labels.add("S"); hits.append("S:시점/장소/활동 배경")
    elif has(r"^\d{4}년", s2) and has(r"(시작|참여|입학|입사|진행)", s2):
        labels.add("S"); hits.append("S:연도 기반 배경")

    if has(TASK, s2) or has(r"(필요가\s*있|필요하다고|하기로\s*했|하고자\s*했|하고자\s*하|해야겠다고|제가\s*맡은|저의\s*역할)", s2):
        labels.add("T"); hits.append("T:목표/역할/과제")
    if future:
        labels.add("T"); hits.append("T:미래 계획/포부")

    action_strong = False
    if has(ACTION_N, s2) and (has(ACTION_V, s2) or has(r"(저는|제가|저희|우리).{0,30}", s2)):
        action_strong = True
    if has(r"(하기\s*위해|위하여|도록)\s*.{0,70}(했습니다|하였습니다|노력|진행|제안|준비|개발|분석|활용|적용|도입|수행|제작|작성)", s2):
        action_strong = True
    if action_strong:
        labels.add("A"); hits.append("A:구체 행동")

    result_strong = False
    if has(RESULT_MARK, s2) and (has(RESULT_N, s2) or has(NUM_RESULT, s2) or has(r"(수\s*있|되었|됐|었습니다|했습니다)", s2)):
        result_strong = True
    if has(NUM_RESULT, s2) and (has(RESULT_N, s2) or has(r"(상승|하락|증가|감소|향상|개선|달성|수상|입상)", s2)):
        result_strong = True
    if has(r"(배웠|깨달|느꼈|알게\s*되|발견하|이해하게|성장할\s*수|얻을\s*수|기를\s*수|길렀|인정\s*받|수상|입상|합격)", s2):
        result_strong = True
    if has(r"(만족도|매출|효율|성과|점수|실적|능력).{0,40}(향상|증가|개선|감소|대폭|높|길렀|키웠)", s2):
        result_strong = True
    if result_strong:
        labels.add("R"); hits.append("R:성과/결과/배움")

    # 후처리: 과잉 라벨 방지
    if "R" in labels and "A" in labels and has(r"^(그\s*결과|결과|이를\s*통해|덕분에|이후)", s2):
        if not has(r"(이용|활용|적용|도입|제작|개발|분석|제안|준비|진행|노력|시도|수행|작성|조사).{0,25}(했|하였|했습니다|하였습니다|방안|방법|결과)", s2):
            labels.discard("A"); hits.append("A제거:순수 결과문")
    if future:
        labels.discard("S")
        if "R" in labels and not has(r"(목표\s*달성|성과\s*창출|기여|도움)", s2):
            labels.discard("R")
        if "A" in labels and has(r"(하고\s*싶|기여|도움|힘쓰겠습니다)", s2):
            labels.discard("A")
    if generic and not ("T" in labels or "R" in labels):
        labels = set(); hits.append("X:일반 의견/기업 설명")
    if "A" in labels and not has(r"(저는|제가|저희|우리|하기\s*위해|도록|하였|했|했습니다|하였습니다|진행|제안|제작|개발|작성|조사|분석|수행|노력|참여|준비|맡|활용|도입|적용|개선|해결|정리|확인|관리|실행|바꾸|찾아|만들|리뷰|처리|관찰|파악|도출|판매|상담|연구|실험)", s2):
        labels.discard("A"); hits.append("A제거:행동 주체/서술 부족")
    if "S" in labels and len(labels) > 1 and not has(r"(당시|상황|문제|어려움|난관|위기|갈등|부족|한계|오류|불편|민원|잡음|처음|계기|배경)", s2):
        if has(r"(경험|프로젝트|팀|수업|활동)", s2) and ("A" in labels or "R" in labels or "T" in labels):
            labels.discard("S"); hits.append("S제거:약한 맥락어")
    if "T" in labels and len(labels) > 1 and has(r"과제", s2) and not has(r"(목표|역할|담당|맡|해야|임무|책임)", s2):
        labels.discard("T"); hits.append("T제거:약한 과제어")

    label = set_to_label(labels)
    conf = min(0.96, 0.5 + 0.12 * len(labels) + 0.06 * len(hits))
    if label == "X":
        conf = 0.58 if len(hits) == 0 else 0.62
    return label, round(conf, 3), "; ".join(hits) if hits else "규칙 미검출"


def clean_answer_text(text):
    if text is None:
        return ""
    t = str(text)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"[\r\t]+", " ", t)
    t = re.sub(r"(좋은점|아쉬운점)\s*\d+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def split_sentences_korean(text):
    t = clean_answer_text(text)
    if not t:
        return []
    t = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", t)
    t = re.sub(r"([.!?。？！])\s*", r"\1<SEP>", t)
    t = re.sub(r"(습니다|했습니다|하였습니다|였습니다|됩니다|입니다|합니다|하였다|했다|된다|한다)\s+(?=[가-힣A-Za-z0-9\"'(\[])", r"\1<SEP>", t)
    parts = [p.replace("<DOT>", ".").strip(" \n\t\"“”") for p in t.split("<SEP>")]
    out = []
    for p in parts:
        p = re.sub(r"\s+", " ", p).strip()
        if len(p) < 5 or re.fullmatch(r"[\W\d_]+", p):
            continue
        out.append(p)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--manual_xlsx", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()

    manual_rows = load_manual_rows(Path(args.manual_xlsx))
    manual_by_comp_q_sent, manual_by_sent_unique = build_manual_maps(manual_rows)

    full_fields = [
        "sample_id", "source_row_id", "sentence_no",
        "url", "company", "season", "specs", "question_no", "question",
        "context_before", "sentence", "context_after",
        "star_label", "label_source", "manual_id", "confidence", "rule_hits",
    ]
    compact_fields = [
        "sample_id", "source_row_id", "sentence_no",
        "company", "season", "question_no", "question",
        "context_before", "sentence", "context_after",
        "star_label", "label_source", "manual_id", "confidence",
    ]
    fields = compact_fields if args.compact else full_fields

    stats = Counter()
    label_counts = Counter()
    sent_counts = []

    with open(args.input_csv, "r", encoding="utf-8-sig", newline="") as f_in, \
         open(args.output_csv, "w", encoding="utf-8-sig", newline="") as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=fields)
        writer.writeheader()

        for row_idx, row in enumerate(reader, start=1):
            stats["original_rows"] += 1
            if not (row.get("answer") or "").strip():
                continue
            stats["nonempty_answer_rows"] += 1
            sents = split_sentences_korean(row.get("answer", ""))
            sent_counts.append(len(sents))

            for i, sent in enumerate(sents, start=1):
                before = sents[i - 2] if i >= 2 else ""
                after = sents[i] if i < len(sents) else ""
                key = (norm_key(row.get("company")), norm_key(row.get("question")), norm_key(sent))
                sent_key = norm_key(sent)

                if key in manual_by_comp_q_sent:
                    label, manual_id = manual_by_comp_q_sent[key]
                    source, confidence, hits = "manual_200", 1.0, "manual_200_exact_company_question_sentence_match"
                    stats["manual_200"] += 1
                elif sent_key in manual_by_sent_unique:
                    label, manual_id = manual_by_sent_unique[sent_key]
                    source, confidence, hits = "manual_200", 1.0, "manual_200_unique_sentence_match"
                    stats["manual_200"] += 1
                else:
                    label, confidence, hits = rule_label(sent, before, after, row.get("question"))
                    source, manual_id = "rule_auto", ""
                    stats["rule_auto"] += 1

                label_counts[label] += 1
                stats["sentence_rows_total"] += 1

                record = {
                    "sample_id": f"JK{row_idx:05d}_S{i:03d}",
                    "source_row_id": row_idx,
                    "sentence_no": i,
                    "url": row.get("url", ""),
                    "company": row.get("company", ""),
                    "season": row.get("season", ""),
                    "specs": row.get("specs", ""),
                    "question_no": row.get("question_no", ""),
                    "question": row.get("question", ""),
                    "context_before": before,
                    "sentence": sent,
                    "context_after": after,
                    "star_label": label,
                    "label_source": source,
                    "manual_id": manual_id,
                    "confidence": confidence,
                    "rule_hits": hits,
                }
                writer.writerow({k: record[k] for k in fields})

    print("Summary")
    print(dict(stats))
    print("avg_sentences_per_answer:", round(statistics.mean(sent_counts), 2) if sent_counts else 0)
    print("label_counts:", dict(label_counts))


if __name__ == "__main__":
    main()
