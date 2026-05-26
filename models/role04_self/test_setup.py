from kiwipiepy import Kiwi
import spacy

kiwi = Kiwi()
nlp = spacy.load("ko_core_news_sm")

# 형태소 분석 테스트
result = kiwi.tokenize("테스트 문장입니다")
print(" Kiwi:", [token.form for token in result])

# spaCy 테스트
doc = nlp("테스트 문장입니다")
print(" spaCy:", [token.text for token in doc])

import spacy

nlp = spacy.load("ko_core_news_sm")

# 더미 문장 테스트
sentences = [
    "저는 팀 프로젝트에서 리더를 맡았습니다.",
    "팀의 매출을 30% 향상시켰습니다.",
    "저는 최선을 다했다고 생각합니다."
]

for sent in sentences:
    doc = nlp(sent)
    print(f"\n📝 문장: {sent}")
    for token in doc:
        print(f"  {token.text} → 품사: {token.pos_}, 의존관계: {token.dep_}, 머리어: {token.head.text}")