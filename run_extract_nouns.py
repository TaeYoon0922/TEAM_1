"""단계 1: kiwipiepy로 명사만 추출해서 저장 (sklearn 미사용)"""
import sys, os, pickle, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(__file__))

from pathlib import Path
from kiwipiepy import Kiwi
from models.role05_match.question_clusterer import load_questions, _CACHE_DIR

NOUN_Q_CACHE = _CACHE_DIR / "noun_questions.pkl"
NOUN_TAGS    = {"NNG", "NNP", "NNB"}  # 일반명사, 고유명사, 의존명사

questions = load_questions()
print(f"질문 수: {len(questions)}")

if NOUN_Q_CACHE.exists():
    print(f"[캐시] 이미 존재: {NOUN_Q_CACHE}")
    with open(NOUN_Q_CACHE, "rb") as f:
        noun_docs = pickle.load(f)
else:
    print("kiwipiepy 명사 추출 중...")
    kiwi = Kiwi()
    noun_docs = []
    for i, q in enumerate(questions):
        tokens = kiwi.analyze(q, top_n=1)[0][0]
        nouns  = [t.form for t in tokens if t.tag in NOUN_TAGS and len(t.form) > 1]
        noun_docs.append(" ".join(nouns) if nouns else q[:20])
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(questions)} 완료")

    with open(NOUN_Q_CACHE, "wb") as f:
        pickle.dump(noun_docs, f)
    print(f"저장 완료: {NOUN_Q_CACHE}")

# 샘플 확인
print("\n[샘플 10개]")
for q, nd in list(zip(questions, noun_docs))[:10]:
    print(f"  원문: {q[:60]}")
    print(f"  명사: {nd[:60]}")
    print()
