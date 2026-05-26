from kiwipiepy import Kiwi
import spacy

kiwi = Kiwi()
nlp = spacy.load("ko_core_news_sm")

# 형태소 분석 테스트
result = kiwi.tokenize("테스트 문장입니다")
print("✅ Kiwi:", [token.form for token in result])

# spaCy 테스트
doc = nlp("테스트 문장입니다")
print("✅ spaCy:", [token.text for token in doc])