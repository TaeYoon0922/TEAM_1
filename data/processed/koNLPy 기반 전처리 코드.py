# -*- coding: utf-8 -*-
"""
KoNLPy 기반 잡코리아 자기소개서 데이터 전처리 파이프라인

입력 CSV 컬럼 예시:
- url, company, season, specs, question_no, question, answer

주요 기능:
1. CSV 로드 및 결측/중복 제거
2. 자기소개서 텍스트 노이즈 제거
3. KoNLPy 형태소 분석(Okt/Komoran/Kkma/Hannanum/Mecab 선택 가능)
4. 품사 필터링, 불용어 제거, 짧은 토큰 제거
5. 전처리 CSV, 토큰 말뭉치 txt, 빈도표 CSV 저장

실행 예시:
python konlpy_jobkorea_preprocess_pipeline.py \
  --input "잡코리아 데이터셋 원문 7000개(1).csv" \
  --output_csv "jobkorea_preprocessed.csv" \
  --tokens_txt "jobkorea_tokens.txt" \
  --freq_csv "jobkorea_token_freq.csv" \
  --analyzer okt \
  --text_col answer \
  --use_question
"""

from __future__ import annotations

import argparse
import html
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
from tqdm import tqdm


# ------------------------------------------------------------
# 1) 불용어 사전
# ------------------------------------------------------------
# 너무 강하게 지우면 직무/성과 정보가 사라질 수 있으므로 기본 불용어는 보수적으로 둔다.
BASE_STOPWORDS = {
    "저", "저희", "제가", "저는", "나", "우리", "본인", "자신",
    "것", "수", "등", "및", "때", "중", "위해", "통해", "대한", "관련",
    "그리고", "또한", "하지만", "그러나", "따라서", "그래서", "이후", "먼저",
    "있다", "없다", "하다", "되다", "이다", "같다", "보다", "받다", "주다",
    "있습니다", "했습니다", "합니다", "됩니다", "되었습니다", "입니다",
    "정도", "부분", "경우", "과정", "결과", "내용", "활동", "경험",
}

# 잡코리아 합격자소서에 자주 반복되지만 분석 목적에 따라 의미가 약할 수 있는 단어.
# 직무 분석에서는 일부가 의미 있을 수 있으니 옵션으로만 적용한다.
JOBKOREA_EXTRA_STOPWORDS = {
    "지원", "직무", "회사", "기업", "업무", "역량", "성장", "입사", "인재",
    "문제", "해결", "목표", "노력", "생각", "사람", "고객", "팀", "프로젝트",
}

# 한 글자여도 의미를 살릴 수 있는 토큰. 필요하면 과제 주제에 맞게 추가한다.
ONE_CHAR_WHITELIST = {"AI", "C", "R", "팀"}


# ------------------------------------------------------------
# 2) 텍스트 정제 함수
# ------------------------------------------------------------
def normalize_text(text: str) -> str:
    """유니코드/HTML/공백 기본 정규화."""
    if pd.isna(text):
        return ""

    text = str(text)
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u200b", " ").replace("\xa0", " ")
    return text


def clean_noise(text: str) -> str:
    """자기소개서 데이터에 맞춘 노이즈 제거."""
    text = normalize_text(text)

    # HTML 태그, URL, 이메일 제거
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", " ", text)

    # 개인정보성 패턴 일부 마스킹/제거: 전화번호, 긴 숫자열
    text = re.sub(r"\b\d{2,3}[- .]?\d{3,4}[- .]?\d{4}\b", " ", text)
    text = re.sub(r"\b\d{6,}\b", " ", text)

    # 잡코리아/자소서에서 자주 보이는 안내 문구성 기호 제거
    text = re.sub(r"[※◆■□▲△▶▷●○◎◇★☆]+", " ", text)

    # 따옴표 제목, 번호 목록의 형식 잡음 완화
    text = re.sub(r"[\"'“”‘’`]+", " ", text)
    text = re.sub(r"(?m)^\s*\d+[.)]\s*", " ", text)

    # 한글, 영어, 숫자, 일부 기술 토큰 기호만 남기기
    # C++, C#, node.js 같은 표현을 완전히 망가뜨리지 않도록 +#._는 임시 보존
    text = re.sub(r"[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9+#._\s-]", " ", text)

    # 자음/모음만 반복되는 잡음 제거
    text = re.sub(r"[ㄱ-ㅎㅏ-ㅣ]+", " ", text)

    # 반복 공백 정리
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ------------------------------------------------------------
# 3) KoNLPy 형태소 분석기 로더
# ------------------------------------------------------------
def load_analyzer(name: str = "okt"):
    """분석기 이름에 맞는 KoNLPy 객체 생성."""
    name = name.lower()

    if name == "okt":
        from konlpy.tag import Okt
        return Okt()
    if name == "komoran":
        from konlpy.tag import Komoran
        return Komoran()
    if name == "kkma":
        from konlpy.tag import Kkma
        return Kkma()
    if name == "hannanum":
        from konlpy.tag import Hannanum
        return Hannanum()
    if name == "mecab":
        from konlpy.tag import Mecab
        return Mecab()

    raise ValueError("analyzer는 okt, komoran, kkma, hannanum, mecab 중 하나여야 합니다.")


# 분석기별 품사 태그가 달라서 필터링 기준을 분리한다.
KEEP_POS = {
    # Okt 품사
    "okt": {"Noun", "Verb", "Adjective", "Alpha", "Number"},
    # 세종계열/분석기별 태그: 명사, 동사, 형용사, 어근, 영어/숫자 계열 중심
    "komoran": {"NNG", "NNP", "VV", "VA", "XR", "SL", "SN"},
    "kkma": {"NNG", "NNP", "NNB", "VV", "VA", "XR", "OL", "NR"},
    "hannanum": {"N", "P", "M", "F"},
    "mecab": {"NNG", "NNP", "VV", "VA", "XR", "SL", "SN"},
}


def pos_tag(analyzer, analyzer_name: str, text: str) -> List[Tuple[str, str]]:
    """분석기별 pos 호출 방식 통일."""
    if not text:
        return []

    if analyzer_name == "okt":
        # norm=True: 정규화, stem=True: 동사/형용사 원형화
        return analyzer.pos(text, norm=True, stem=True)

    return analyzer.pos(text)


def token_filter(
    tagged: Iterable[Tuple[str, str]],
    analyzer_name: str,
    stopwords: set[str],
    min_len: int = 2,
    keep_numbers: bool = False,
) -> List[str]:
    """품사/불용어/길이 기준으로 토큰 필터링."""
    keep_pos = KEEP_POS[analyzer_name]
    tokens: List[str] = []

    for token, pos in tagged:
        token = token.strip().lower()
        if not token:
            continue

        # 품사 필터링
        if pos not in keep_pos:
            continue

        # 숫자 제거 옵션
        if not keep_numbers and re.fullmatch(r"\d+", token):
            continue

        # 기술 토큰 보정: c++/c#/node.js 등은 일부 허용
        token = token.strip("._- ")
        if not token:
            continue

        # 불용어 제거
        if token in stopwords:
            continue

        # 한 글자 토큰 제거. 단, 화이트리스트는 보존
        if len(token) < min_len and token.upper() not in ONE_CHAR_WHITELIST:
            continue

        tokens.append(token)

    return tokens


def preprocess_document(
    text: str,
    analyzer,
    analyzer_name: str,
    stopwords: set[str],
    min_len: int = 2,
    keep_numbers: bool = False,
) -> tuple[str, list[str]]:
    clean = clean_noise(text)
    tagged = pos_tag(analyzer, analyzer_name, clean)
    tokens = token_filter(
        tagged,
        analyzer_name=analyzer_name,
        stopwords=stopwords,
        min_len=min_len,
        keep_numbers=keep_numbers,
    )
    return clean, tokens


# ------------------------------------------------------------
# 4) 전체 파이프라인
# ------------------------------------------------------------
def run_pipeline(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_path}")

    df = pd.read_csv(input_path, encoding=args.encoding)

    required_cols = {args.text_col}
    if args.use_question:
        required_cols.add("question")
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV에 필요한 컬럼이 없습니다: {sorted(missing_cols)}")

    # 결측 answer 제거
    df[args.text_col] = df[args.text_col].fillna("").astype(str)
    df = df[df[args.text_col].str.strip().ne("")].copy()

    # 중복 답변 제거 옵션
    if args.drop_duplicates:
        before = len(df)
        df = df.drop_duplicates(subset=[args.text_col]).copy()
        print(f"중복 제거: {before:,} -> {len(df):,}")

    # 질문 + 답변을 같이 넣으면 문항 맥락까지 반영 가능
    if args.use_question:
        df["raw_text"] = df["question"].fillna("").astype(str) + " " + df[args.text_col]
    else:
        df["raw_text"] = df[args.text_col]

    stopwords = set(BASE_STOPWORDS)
    if args.use_jobkorea_stopwords:
        stopwords |= JOBKOREA_EXTRA_STOPWORDS

    if args.stopwords_file:
        sw_path = Path(args.stopwords_file)
        if sw_path.exists():
            user_sw = {
                line.strip().lower()
                for line in sw_path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.startswith("#")
            }
            stopwords |= user_sw
        else:
            raise FileNotFoundError(f"불용어 파일을 찾을 수 없습니다: {sw_path}")

    analyzer_name = args.analyzer.lower()
    analyzer = load_analyzer(analyzer_name)

    tqdm.pandas(desc="KoNLPy preprocessing")

    results = []
    for text in tqdm(df["raw_text"].tolist(), desc="형태소 분석 중"):
        clean, tokens = preprocess_document(
            text=text,
            analyzer=analyzer,
            analyzer_name=analyzer_name,
            stopwords=stopwords,
            min_len=args.min_len,
            keep_numbers=args.keep_numbers,
        )
        results.append((clean, tokens))

    df["clean_text"] = [x[0] for x in results]
    df["tokens"] = [" ".join(x[1]) for x in results]
    df["token_count"] = [len(x[1]) for x in results]

    # 토큰이 너무 적은 문서 제거
    before = len(df)
    df = df[df["token_count"] >= args.min_tokens].copy()
    print(f"토큰 수 기준 필터링: {before:,} -> {len(df):,}")

    # 저장
    output_csv = Path(args.output_csv)
    tokens_txt = Path(args.tokens_txt)
    freq_csv = Path(args.freq_csv)

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    tokens_txt.write_text("\n".join(df["tokens"].tolist()), encoding="utf-8")

    counter = Counter()
    for doc in df["tokens"]:
        counter.update(doc.split())

    freq_df = pd.DataFrame(counter.most_common(), columns=["token", "freq"])
    freq_df.to_csv(freq_csv, index=False, encoding="utf-8-sig")

    print("\n완료")
    print(f"- 전처리 CSV: {output_csv.resolve()}")
    print(f"- 토큰 말뭉치 TXT: {tokens_txt.resolve()}")
    print(f"- 토큰 빈도표 CSV: {freq_csv.resolve()}")
    print("\n상위 토큰 30개")
    print(freq_df.head(30).to_string(index=False))


# ------------------------------------------------------------
# 5) CLI
# ------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="입력 CSV 경로")
    parser.add_argument("--encoding", default="utf-8-sig", help="CSV 인코딩")
    parser.add_argument("--text_col", default="answer", help="분석할 텍스트 컬럼")
    parser.add_argument("--use_question", action="store_true", help="question 컬럼을 answer 앞에 붙여 분석")
    parser.add_argument("--analyzer", default="okt", choices=["okt", "komoran", "kkma", "hannanum", "mecab"])
    parser.add_argument("--output_csv", default="jobkorea_preprocessed.csv")
    parser.add_argument("--tokens_txt", default="jobkorea_tokens.txt")
    parser.add_argument("--freq_csv", default="jobkorea_token_freq.csv")
    parser.add_argument("--stopwords_file", default=None, help="사용자 정의 불용어 txt 파일. 한 줄에 한 단어")
    parser.add_argument("--use_jobkorea_stopwords", action="store_true", help="자소서 특화 추가 불용어 적용")
    parser.add_argument("--drop_duplicates", action="store_true", help="동일 answer 중복 제거")
    parser.add_argument("--min_len", type=int, default=2, help="최소 토큰 길이")
    parser.add_argument("--min_tokens", type=int, default=3, help="최소 문서 토큰 수")
    parser.add_argument("--keep_numbers", action="store_true", help="숫자 토큰 유지")
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
