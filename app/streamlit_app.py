from __future__ import annotations

from datetime import datetime
from html import escape

import streamlit as st

from analyzer import AnalysisResult, analyze_cover_letter


METRIC_DESCRIPTIONS = {
    "문항 적합성": {
        "summary": "답변이 질문의 핵심 의도와 조건에 맞는지 보는 지표입니다.",
        "detail": "답변이 아무리 구체적이어도 질문과 다른 내용을 쓰면 좋은 자소서로 평가되기 어렵습니다.",
        "check": "질문 핵심어, 질문 유형, 복합 조건이 답변에 반영됐는지 확인합니다.",
    },
    "핵심 주장 명확성": {
        "summary": "핵심 결론을 앞에서 먼저 제시하는 두괄식 구조입니다.",
        "detail": "첫 문장이나 문단 앞부분에서 무엇을 했고 어떤 성과를 냈는지 빠르게 보여주는지 봅니다.",
        "check": "첫 문장에 성과, 변화, 직무 적합성이 보이는지 확인합니다.",
    },
    "경험 구체성": {
        "summary": "상황, 과제, 행동, 결과(STAR)의 경험 서술 구조입니다.",
        "detail": "Situation, Task, Action, Result가 모두 보여야 경험이 단순 주장보다 설득력 있는 근거로 읽힙니다.",
        "check": "배경, 맡은 역할, 실제 행동, 결과가 빠지지 않았는지 확인합니다.",
    },
    "지원 적합성": {
        "summary": "직무·전공·회사와의 연결성을 보는 지표입니다.",
        "detail": "경험이 지원 직무/전공/회사와 연결될수록 '왜 이 사람인지'가 설득력 있게 드러납니다.",
        "check": "지원 직무명·전공 지식·회사와의 연결 표현이 있는지 확인합니다.",
    },
    "표현 명료성": {
        "summary": "모호어·추상어·상투어가 적을수록 좋은 지표입니다.",
        "detail": "헤지 표현·추상어가 적고 수치·행동 중심일수록 명료합니다.",
        "check": "열심히, 다양한, 성장 같은 추상 표현 대신 수치·행동 중심 표현이 쓰였는지 확인합니다.",
    },
    "자기표현 차별성": {
        "summary": "본인만의 경험과 관점이 드러나는지 보는 지표입니다.",
        "detail": "제가, 저는 중심의 서술만 반복되면 팀·고객·조직에 준 영향과 본인만의 차별점이 약해 보입니다.",
        "check": "내 행동 이후 팀, 고객, 조직에 어떤 변화가 있었는지 확인합니다.",
    },
    "문장 완성도": {
        "summary": "맞춤법·문장 구조·가독성을 보는 지표입니다.",
        "detail": "한 문장에 한 메시지를 담고 길이가 적절할수록 읽기 쉽고 완성도 있게 읽힙니다.",
        "check": "지나치게 긴 문장·복잡한 구조가 없는지 확인합니다.",
    },
}


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --page: #f7f7f8;
            --surface: #ffffff;
            --sidebar: #171717;
            --ink: #202123;
            --muted: #6b7280;
            --line: #e5e7eb;
            --soft: #f3f4f6;
            --green: #10a37f;
            --blue: #2563eb;
            --amber: #f59e0b;
            --red: #dc2626;
        }

        .stApp {
            background: var(--page);
            color: var(--ink);
        }

        .block-container {
            max-width: 980px;
            padding-top: 3.8rem;
            padding-bottom: 2.6rem;
        }

        [data-testid="stHeader"] {
            background: rgba(247, 247, 248, 0.92);
            backdrop-filter: blur(10px);
        }

        [data-testid="stSidebar"] {
            background: var(--sidebar);
        }

        [data-testid="stSidebar"] * {
            color: #f4f4f5;
        }

        [data-testid="stSidebar"] .stButton > button {
            background: #2d2d2d;
            border: 1px solid #3f3f46;
            color: #ffffff;
            justify-content: flex-start;
        }

        .topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            padding: 0.2rem 0 1rem;
            border-bottom: 1px solid var(--line);
        }

        .brand {
            font-size: 1.05rem;
            font-weight: 800;
            letter-spacing: 0;
        }

        .status-pill {
            border: 1px solid var(--line);
            border-radius: 999px;
            color: var(--muted);
            background: #ffffff;
            padding: 0.38rem 0.72rem;
            font-size: 0.82rem;
            font-weight: 700;
            white-space: nowrap;
        }

        .message {
            display: grid;
            grid-template-columns: 38px minmax(0, 1fr);
            gap: 0.9rem;
            padding: 1.35rem 0;
        }

        .avatar {
            width: 38px;
            height: 38px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 900;
        }

        .assistant-avatar {
            background: var(--green);
            color: #ffffff;
        }

        .user-avatar {
            background: #111827;
            color: #ffffff;
        }

        .bubble, .page-card {
            border: 1px solid var(--line);
            background: var(--surface);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            box-shadow: 0 8px 22px rgba(16, 24, 40, 0.04);
        }

        .bubble h1, .page-card h1 {
            font-size: 1.65rem;
            line-height: 1.22;
            margin: 0 0 0.6rem;
            letter-spacing: 0;
        }

        .bubble p, .page-card p {
            color: var(--muted);
            line-height: 1.7;
            margin: 0;
        }

        .prompt-shell {
            position: sticky;
            bottom: 0;
            z-index: 10;
            padding: 1rem 0 0.1rem;
            background: linear-gradient(180deg, rgba(247, 247, 248, 0), var(--page) 28%);
        }

        .stTextArea textarea {
            min-height: 150px;
            border-radius: 18px;
            border: 1px solid #d1d5db;
            background: #ffffff;
            box-shadow: 0 12px 36px rgba(16, 24, 40, 0.08);
            font-size: 0.98rem;
            line-height: 1.65;
        }

        .stButton > button {
            height: 2.9rem;
            border-radius: 12px;
            border: 1px solid var(--green);
            background: var(--green);
            color: #ffffff;
            font-weight: 850;
        }

        .score-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 1rem 0 1.1rem;
        }

        .score-card {
            border: 1px solid var(--line);
            border-radius: 14px;
            background: #ffffff;
            padding: 0.95rem;
        }

        .score-label {
            color: var(--muted);
            font-size: 0.8rem;
            font-weight: 800;
            margin-bottom: 0.3rem;
        }

        .score-value {
            font-size: 1.9rem;
            font-weight: 900;
            line-height: 1;
        }

        .score-level {
            display: inline-flex;
            margin-top: 0.58rem;
            border-radius: 999px;
            background: var(--soft);
            color: #374151;
            padding: 0.22rem 0.52rem;
            font-size: 0.75rem;
            font-weight: 800;
        }

        .meter {
            width: 100%;
            height: 7px;
            border-radius: 999px;
            background: #edf0f3;
            overflow: hidden;
            margin-top: 0.75rem;
        }

        .meter span {
            display: block;
            height: 100%;
            border-radius: 999px;
        }

        .ind-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 1rem 0 1.1rem;
        }

        .ind-card {
            border: 1px solid var(--line);
            border-radius: 14px;
            background: #ffffff;
            padding: 0.95rem 1rem;
        }

        .ind-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.5rem;
            font-weight: 800;
            font-size: 0.95rem;
            margin-bottom: 0.5rem;
        }

        .ind-level {
            border-radius: 999px;
            color: #ffffff;
            padding: 0.18rem 0.55rem;
            font-size: 0.72rem;
            font-weight: 800;
            white-space: nowrap;
        }

        .ind-level.ind-na {
            background: #e5e7eb;
            color: #6b7280;
        }

        .ind-feedback {
            color: #374151;
            font-size: 0.88rem;
            line-height: 1.5;
        }

        .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }

        .chip {
            border: 1px solid var(--line);
            border-radius: 999px;
            background: #ffffff;
            padding: 0.42rem 0.68rem;
            color: #374151;
            font-size: 0.85rem;
            line-height: 1.35;
        }

        .sentence-box, .history-card {
            border: 1px solid var(--line);
            border-radius: 12px;
            background: #ffffff;
            padding: 0.85rem;
            margin-bottom: 0.65rem;
        }

        .sentence-box b {
            color: var(--ink);
        }

        .sentence-box p {
            margin-top: 0.35rem;
            color: var(--muted);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.45rem;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            border: 1px solid var(--line);
            background: #ffffff;
            padding: 0.45rem 0.9rem;
        }

        @media (max-width: 760px) {
            .score-grid, .ind-grid {
                grid-template-columns: repeat(1, minmax(0, 1fr));
            }

            .topbar {
                align-items: flex-start;
                flex-direction: column;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="자소서 분석", page_icon="📝", layout="wide")
    inject_styles()
    init_state()
    render_sidebar()

    page = st.session_state.page
    if page == "자소서 분석":
        render_analysis_page()
    elif page == "저장된 분석":
        render_history_page()
    else:
        render_metric_page()


def init_state() -> None:
    if "page" not in st.session_state:
        st.session_state.page = "자소서 분석"
    if "cover_letter_text" not in st.session_state:
        st.session_state.cover_letter_text = ""
    if "question_text" not in st.session_state:
        st.session_state.question_text = ""
    if "analysis_history" not in st.session_state:
        st.session_state.analysis_history = []


def render_sidebar() -> None:
    with st.sidebar:
        st.header("자소서 분석")
        if st.button("새 분석 시작", use_container_width=True):
            st.session_state.cover_letter_text = ""
            st.session_state.question_text = ""
            st.session_state.page = "자소서 분석"
            st.rerun()
        if st.button("저장된 분석", use_container_width=True):
            st.session_state.page = "저장된 분석"
            st.rerun()
        if st.button("분석 지표", use_container_width=True):
            st.session_state.page = "분석 지표"
            st.rerun()


def render_topbar(title: str, status: str) -> None:
    st.markdown(
        f"""
        <div class="topbar">
            <div class="brand">{escape(title)}</div>
            <div class="status-pill">{escape(status)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_analysis_page() -> None:
    render_topbar("자소서 분석", "질문 적합도 포함 피드백")

    text = st.session_state.cover_letter_text.strip()
    question = st.session_state.question_text.strip()
    if text:
        result = analyze_cover_letter(text, question)
        render_user_message(text, question)
        render_assistant_message(result)
    else:
        render_intro_message()

    render_input_form()


def render_history_page() -> None:
    render_topbar("저장된 분석", "최근 분석 기록")
    history = st.session_state.analysis_history

    if not history:
        st.markdown(
            """
            <div class="page-card">
                <h1>저장된 분석이 없습니다.</h1>
                <p>자소서 분석 화면에서 문장을 입력하고 분석하면 여기에 자동으로 저장됩니다.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    for item in reversed(history):
        st.markdown(
            f"""
            <div class="history-card">
                <b>{escape(item["time"])}</b>
                <p>{escape(item["preview"])}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        cols = st.columns([1, 1, 4])
        with cols[0]:
            if st.button("열기", key=f'open_{item["id"]}'):
                st.session_state.cover_letter_text = item["text"]
                st.session_state.question_text = item.get("question", "")
                st.session_state.page = "자소서 분석"
                st.rerun()
        with cols[1]:
            if st.button("결과 보기", key=f'view_{item["id"]}'):
                render_assistant_message(
                    analyze_cover_letter(item["text"], item.get("question", ""))
                )

    st.divider()
    if st.button("기록 지우기", use_container_width=True):
        st.session_state.analysis_history = []
        st.rerun()


def render_metric_page() -> None:
    render_topbar("분석 지표", "지표 설명")

    st.write("")
    st.markdown(
        "자소서를 평가할 때 보는 일곱 가지 기준입니다. 각 지표는 문장의 구조와 표현을 다른 관점에서 확인합니다."
    )

    metric_items = list(METRIC_DESCRIPTIONS.items())
    for row_start in range(0, len(metric_items), 2):
        cols = st.columns(2, gap="large")
        for col, (metric, info) in zip(cols, metric_items[row_start : row_start + 2]):
            with col:
                with st.container():
                    st.subheader(metric)
                    st.markdown(f"**{info['summary']}**")
                    st.write(info["detail"])
                    st.divider()
                    st.caption(info["check"])

    with st.expander("결과는 어떻게 보나요?"):
        st.write("점수 대신, 각 지표별로 무엇을 어떻게 고치면 좋을지 피드백만 보여줍니다.")
        st.write("질문 유형(군집)에 따라 일부 지표는 '해당 없음'으로 빠질 수 있습니다.")


def render_intro_message() -> None:
    st.markdown(
        """
        <div class="message">
            <div class="avatar assistant-avatar">AI</div>
            <div class="bubble">
                <h1>자소서 문장을 붙여넣어 주세요.</h1>
                <p>
                    질문과 자소서 답변을 붙여넣으면 문항 적합성, 핵심 주장 명확성, 경험 구체성, 지원 적합성,
                    표현 명료성, 자기표현 차별성, 문장 완성도 7가지 관점에서 바로 고칠 수 있는 피드백으로 정리해드립니다.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_input_form() -> None:
    st.markdown('<div class="prompt-shell">', unsafe_allow_html=True)
    with st.form("analysis_form", clear_on_submit=False):
        question_text = st.text_area(
            "질문 입력",
            value=st.session_state.question_text,
            placeholder="자소서 질문을 붙여넣어 주세요. 예: 지원동기와 입사 후 기여 계획을 기술해 주세요.",
            height=92,
        )
        user_text = st.text_area(
            "자소서 입력",
            value=st.session_state.cover_letter_text,
            placeholder="자소서 문장을 붙여넣고 분석하기를 눌러주세요.",
            height=155,
        )
        submitted = st.form_submit_button("분석하기", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if submitted:
        cleaned = user_text.strip()
        cleaned_question = question_text.strip()
        st.session_state.cover_letter_text = cleaned
        st.session_state.question_text = cleaned_question
        if cleaned:
            save_history(cleaned, cleaned_question)
        st.rerun()


def save_history(text: str, question: str) -> None:
    preview = text.replace("\n", " ")[:40]
    if len(text) > 40:
        preview += "..."

    latest = st.session_state.analysis_history[-1] if st.session_state.analysis_history else None
    if latest and latest["text"] == text and latest.get("question", "") == question:
        latest["time"] = datetime.now().strftime("%H:%M")
        latest["preview"] = preview
        return

    st.session_state.analysis_history.append(
        {
            "id": datetime.now().strftime("%Y%m%d%H%M%S%f"),
            "time": datetime.now().strftime("%H:%M"),
            "question": question,
            "text": text,
            "preview": preview,
        }
    )


def render_user_message(text: str, question: str = "") -> None:
    preview = escape(text).replace("\n", "<br>")
    question_html = ""
    if question:
        question_html = f'<p><b>질문</b><br>{escape(question).replace(chr(10), "<br>")}</p><hr>'
    st.markdown(
        f"""
        <div class="message">
            <div class="avatar user-avatar">나</div>
            <div class="bubble">{question_html}{preview}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_assistant_message(result: AnalysisResult) -> None:
    st.markdown(
        f"""
        <div class="message">
            <div class="avatar assistant-avatar">AI</div>
            <div class="bubble">
                <h1>피드백</h1>
                <p>{escape(result.summary)}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(indicator_cards_html(result), unsafe_allow_html=True)

    feedback_tab, sentence_tab = st.tabs(["개선 포인트", "문장별 코멘트"])
    with feedback_tab:
        render_feedback(result)
    with sentence_tab:
        render_sentence_feedback(result)


def indicator_cards_html(result: AnalysisResult) -> str:
    """7지표 피드백 카드 — 점수 없이 지표명·등급·피드백만 노출."""
    cards = []
    for metric in result.metrics.values():
        if not metric.applicable:
            level_html = '<span class="ind-level ind-na">해당 없음</span>'
        else:
            level_html = f'<span class="ind-level" style="background:{level_color(metric.level)};">{escape(metric.level)}</span>'
        cards.append(
            f"""
            <div class="ind-card">
                <div class="ind-head">{escape(metric.label)}{level_html}</div>
                <div class="ind-feedback">{escape(metric.feedback)}</div>
            </div>
            """
        )
    return f'<div class="ind-grid">{"".join(cards)}</div>'


def level_color(level: str) -> str:
    if level in ("우수", "낮음"):
        return "#10a37f"
    if level in ("보통", "주의", "분석 보류"):
        return "#f59e0b"
    return "#dc2626"  # 보완 필요 / 높음 / 분석 불가 등


def render_feedback(result: AnalysisResult) -> None:
    strength_col, improve_col = st.columns(2, gap="large")
    with strength_col:
        st.markdown("### 강점")
        st.markdown(chips_html(result.strengths), unsafe_allow_html=True)
    with improve_col:
        st.markdown("### 우선 개선 포인트")
        st.markdown(chips_html(result.improvements), unsafe_allow_html=True)


def render_sentence_feedback(result: AnalysisResult) -> None:
    for index, item in enumerate(result.sentence_feedback, start=1):
        st.markdown(
            f"""
            <div class="sentence-box">
                <b>{index}. {escape(item["sentence"])}</b>
                <p>{escape(item["comment"])}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def chips_html(items: list[str]) -> str:
    chips = "".join(f'<span class="chip">{escape(item)}</span>' for item in items)
    return f'<div class="chip-row">{chips}</div>'


if __name__ == "__main__":
    main()
