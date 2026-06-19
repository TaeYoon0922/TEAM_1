from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import kss
import csv
from pathlib import Path


_model = None

def get_model():
    global _model
    if _model is None:
        print("KoSimCSE 로딩 중...")
        _model = SentenceTransformer("BM-K/KoSimCSE-roberta")
        print("완료")
    return _model




def analyze_paragraph(paragraph: str) -> dict:
    model = get_model()


    sentences = kss.split_sentences(paragraph.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]


    if len(sentences) < 2:
        return {
            "structure_type" : "too_short",
            "center_sentence": sentences[0] if sentences else "",
            "center_idx"     : 0,
            "dugoalsik_score": None,
            "uniformity"     : None,
            "feedback"       : "문장이 너무 짧아 분석이 어렵습니다.",
            "all_scores"     : [],
        }


    embeddings = model.encode(sentences)
    centroid   = embeddings.mean(axis=0, keepdims=True)


    scores = cosine_similarity(embeddings, centroid).flatten()


    center_idx      = int(scores.argmax())
    center_sentence = sentences[center_idx]
    center_score    = float(scores[center_idx])


    uniformity = float(scores.std())
    is_list_type = uniformity < 0.03 and len(sentences) >= 3



    first_score      = float(scores[0])
    is_dugoalsik     = (center_idx == 0) and (first_score >= 0.75)
    dugoalsik_score  = first_score


    if is_list_type:
        structure_type = "list_type"
    elif is_dugoalsik:
        structure_type = "dugoalsik"
    else:
        structure_type = "migugoalsik"


    feedback = _make_feedback(
        structure_type, center_sentence, center_idx,
        dugoalsik_score, uniformity, sentences
    )

    return {
        "structure_type" : structure_type,
        "center_sentence": center_sentence,
        "center_idx"     : center_idx,
        "dugoalsik_score": round(dugoalsik_score, 4),
        "uniformity"     : round(uniformity, 4),
        "feedback"       : feedback,
        "all_scores"     : [
            {"sentence": s, "score": round(float(sc), 4)}
            for s, sc in zip(sentences, scores)
        ],
    }


def analyze_essay(essay: str) -> dict:
    paragraphs = [p.strip() for p in essay.split("\n\n") if p.strip()]
    results    = []

    for i, para in enumerate(paragraphs):
        result = analyze_paragraph(para)
        result["paragraph_idx"] = i
        result["paragraph_preview"] = para[:40] + "..." if len(para) > 40 else para
        results.append(result)


    types       = [r["structure_type"] for r in results]
    dugo_count  = types.count("dugoalsik")
    migu_count  = types.count("migugoalsik")
    list_count  = types.count("list_type")
    total       = len(results)


    center_sentences = [
        r["center_sentence"]
        for r in results
        if r["structure_type"] != "too_short"
    ]

    return {
        "total_paragraphs"   : total,
        "dugoalsik_count"    : dugo_count,
        "migugoalsik_count"  : migu_count,
        "list_type_count"    : list_count,
        "dugoalsik_ratio"    : round(dugo_count / total, 2) if total else 0,
        "center_sentences"   : center_sentences,
        "paragraph_results"  : results,
        "overall_feedback"   : _make_overall_feedback(dugo_count, migu_count,
                                                       list_count, total),
    }




def _make_feedback(structure_type, center_sentence, center_idx,
                   dugoalsik_score, uniformity, sentences) -> str:
    if structure_type == "dugoalsik":
        return (
            f"✅ 두괄식 구조입니다. (점수: {dugoalsik_score:.2f})\n"
            f"   첫 문장이 핵심을 잘 요약하고 있습니다."
        )
    elif structure_type == "list_type":
        return (
            f"📋 나열형 구조입니다. (균일도: {uniformity:.3f})\n"
            f"   모든 문장이 비슷한 비중으로 나열되어 있습니다.\n"
            f"   전체를 아우르는 핵심 문장을 첫 줄에 추가해보세요."
        )
    else:
        return (
            f"⚠️  미괄식 또는 산만한 구조입니다. (점수: {dugoalsik_score:.2f})\n"
            f"   핵심 문장은 {center_idx + 1}번째 문장입니다:\n"
            f"   → \"{center_sentence}\"\n"
            f"   이 문장을 첫 번째로 올려보세요."
        )


def _make_overall_feedback(dugo, migu, lst, total) -> str:
    ratio = dugo / total if total else 0
    lines = [f"📊 전체 {total}개 문단 분석 결과"]
    lines.append(f"   두괄식: {dugo}개 | 미괄식: {migu}개 | 나열형: {lst}개")

    if ratio >= 0.7:
        lines.append("✅ 전체적으로 두괄식 구조가 잘 지켜지고 있습니다.")
    elif ratio >= 0.4:
        lines.append("⚠️  일부 문단에서 두괄식이 지켜지지 않고 있습니다.")
    else:
        lines.append("❌ 대부분의 문단이 미괄식입니다. 핵심 내용을 앞으로 가져오세요.")

    return "\n".join(lines)




if __name__ == "__main__":
    train_path = Path(__file__).resolve().parents[2] / "data" / "processed" / "jobkorea_train.csv"
    with train_path.open(encoding="utf-8-sig", newline="") as f:
        train_essay = next(
            row["answer"]
            for row in csv.DictReader(f)
            if row.get("answer", "").strip()
        )

    print("=" * 60)
    print(f"train 데이터 예시: {train_path.relative_to(Path(__file__).resolve().parents[2])}")
    result = analyze_essay(train_essay)

    print(result["overall_feedback"])
    print()

    for r in result["paragraph_results"]:
        print(f"[문단 {r['paragraph_idx'] + 1}] {r['paragraph_preview']}")
        print(r["feedback"])
        print()

    print("=" * 60)
    print("📝 자소서 핵심 요약 (중심문장 모음)")
    for i, s in enumerate(result["center_sentences"], 1):
        print(f"  {i}. {s}")
