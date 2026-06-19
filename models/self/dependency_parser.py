import spacy
from kiwipiepy import Kiwi
from self_dict_v1 import SELF_SUBJECT, SELF_EMOTION_VERB, SELF_COGNITION_VERB
from contribution_dict_v1 import CONTRIBUTION_RESULT_NOUN, CONTRIBUTION_EXTERNAL_SUBJECT

nlp = spacy.load("ko_core_news_sm")
kiwi = Kiwi()


FIRST_PERSON_TOKENS = {"저", "제", "나", "내", "본인"}


CONTRIBUTION_VERB_ROOTS = {
    "달성", "개선", "향상", "절감", "단축", "증가", "감소",
    "구현", "개발", "설계", "완료", "완성", "배포",
    "해결", "처리", "도입", "적용", "기여", "주도",
}


SELF_VERB_ROOTS = {
    "느끼", "깨닫", "알", "배우", "생각", "이해",
    "경험", "성장", "발전", "익히", "습득", "쌓",
    "뿌듯", "만족", "감동",
}


def parse_sentence(text: str) -> dict:
    doc = nlp(text)

    subject = None
    predicate = None
    subject_type = "unknown"
    verb_type = "neutral"

    dep_pairs = []
    for token in doc:
        dep_pairs.append({
            "token": token.text,
            "dep":   token.dep_,
            "head":  token.head.text,
            "pos":   token.pos_,
        })


        if token.dep_ == "ROOT":
            predicate = token.text




        if token.dep_ in ("nsubj", "nsubjpass", "dislocated") \
                and token.pos_ != "NUM" \
                and not token.text[0].isdigit():
            subject = token.text


    if subject:
        if any(fp in subject for fp in FIRST_PERSON_TOKENS):
            subject_type = "first_person"
        elif any(ext in subject for ext in CONTRIBUTION_EXTERNAL_SUBJECT):
            subject_type = "external"


    if predicate:
        morph_result = kiwi.tokenize(predicate)
        for token in morph_result:
            if token.form in CONTRIBUTION_VERB_ROOTS:
                verb_type = "contribution"
                break
            elif token.form in SELF_VERB_ROOTS:
                verb_type = "self"
                break


    if subject_type == "first_person" and verb_type == "self":
        sentence_type = "self_centered"
    elif verb_type == "contribution" or subject_type == "external":
        sentence_type = "contribution"
    else:
        sentence_type = "neutral"

    return {
        "text":          text,
        "subject":       subject,
        "predicate":     predicate,
        "subject_type":  subject_type,
        "verb_type":     verb_type,
        "sentence_type": sentence_type,
        "dep_pairs":     dep_pairs,
    }


def parse_paragraph(text: str) -> dict:

    split_result = kiwi.split_into_sents(text)
    sentences_text = [s.text for s in split_result]

    results = [parse_sentence(s) for s in sentences_text]

    self_count = sum(1 for r in results if r["sentence_type"] == "self_centered")
    contrib_count = sum(1 for r in results if r["sentence_type"] == "contribution")
    neutral_count = sum(1 for r in results if r["sentence_type"] == "neutral")
    total = max(len(results), 1)

    return {
        "sentences":          results,
        "total_count":        total,
        "self_count":         self_count,
        "contribution_count": contrib_count,
        "neutral_count":      neutral_count,
        "self_ratio":         round(self_count / total, 2),
        "contribution_ratio": round(contrib_count / total, 2),
    }
