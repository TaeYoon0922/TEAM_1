from kiwipiepy import Kiwi
import spacy

kiwi = Kiwi()
nlp = spacy.load("ko_core_news_sm")


print("=" * 50)
print("R04-01: 환경 세팅 확인")
print("=" * 50)

result = kiwi.tokenize("테스트 문장입니다")
print("[+] Kiwi:", [token.form for token in result])

doc = nlp("테스트 문장입니다")
print("[+] spaCy:", [token.text for token in doc])


print()
print("=" * 50)
print("R04-02: 주어-술어 관계 추출 테스트")
print("=" * 50)

sentences = [
    "저는 팀 프로젝트에서 리더를 맡았습니다.",
    "팀의 매출을 30% 향상시켰습니다.",
    "저는 최선을 다했다고 생각합니다.",
    "시스템 응답 속도를 2배 단축했습니다.",
    "저는 이 경험을 통해 많은 것을 느꼈습니다.",
]

for sent in sentences:
    doc = nlp(sent)
    print(f"\n[문장] {sent}")
    for token in doc:
        print(f"  {token.text:12} 품사: {token.pos_:6}  의존관계: {token.dep_:12}  머리어: {token.head.text}")
