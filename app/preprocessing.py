import re
from typing import List, Tuple, Set

try:
    from konlpy.tag import Okt
except Exception:
    Okt = None


class Preprocessor:
    """KoNLPy 기반 형태소 분석 및 노이즈 제거 유틸리티.

    주요 기능:
    - URL/이메일/숫자/특수문자 제거
    - KoNLPy `Okt`를 이용한 형태소 분석(`morphs`, `pos`, `nouns`)
    - POS 필터링 및 불용어 제거
    """

    def __init__(self, pos_keep: List[str] = None, stopwords: Set[str] = None):
        if Okt is None:
            raise RuntimeError("KoNLPy Okt tagger is not available. Install konlpy and Java.")
        self.tagger = Okt()
        self.pos_keep = pos_keep or ["Noun", "Verb", "Adjective"]
        self.stopwords = set(stopwords or [])

    def normalize(self, text: str) -> str:
        """기본 노멀라이징: URL, 이메일, 숫자, 각종 특수문자 제거 및 공백 정리."""
        if not text:
            return ""
        text = text.strip()
        text = re.sub(r"https?://\S+", " ", text)
        text = re.sub(r"\S+@\S+", " ", text)
        text = re.sub(r"[\u2000-\u206F\u2E00-\u2E7F\!\"#\$%&\(\)\*\+,\-./:;<=>\?@\[\]^_`{\|}~]", " ", text)
        text = re.sub(r"\d+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def morphs(self, text: str) -> List[str]:
        """정규화 후 형태소 추출(어근화 및 정규화 옵션 적용)."""
        text = self.normalize(text)
        if not text:
            return []
        toks = self.tagger.morphs(text, norm=True, stem=True)
        return [t for t in toks if t and t not in self.stopwords]

    def pos(self, text: str) -> List[Tuple[str, str]]:
        """정규화 후 품사 태깅을 반환."""
        text = self.normalize(text)
        if not text:
            return []
        return self.tagger.pos(text, norm=True, stem=True)

    def filter_pos(self, pos_tags: List[Tuple[str, str]]) -> List[str]:
        """지정한 품사만 남기고 불용어 제거."""
        return [tok for tok, tag in pos_tags if tag in self.pos_keep and tok not in self.stopwords]

    def preprocess(self, text: str, as_morphs: bool = False) -> List[str]:
        """통합 전처리: 정규화 → 형태소 또는 POS 필터링 반환."""
        if as_morphs:
            return self.morphs(text)
        return self.filter_pos(self.pos(text))

    def extract_nouns(self, text: str) -> List[str]:
        """명사만 추출(정규화 적용)."""
        text = self.normalize(text)
        if not text:
            return []
        nouns = self.tagger.nouns(text)
        return [n for n in nouns if n and n not in self.stopwords]


if __name__ == "__main__":
    p = Preprocessor(stopwords={"그", "이", "있다", "없다", "하는"})
    s = "이 문장은 테스트입니다. 이메일 test@example.com, 링크 https://example.com, 숫자 1234 포함."
    print("Normalized:", p.normalize(s))
    print("Morphs:", p.morphs(s))
    print("Filtered POS:", p.preprocess(s))
    print("Nouns:", p.extract_nouns(s))
