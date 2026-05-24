from __future__ import annotations

from datetime import datetime
from html import escape

import altair as alt
import pandas as pd
import streamlit as st

from analyzer import AnalysisResult, analyze_cover_letter


GOOD_METRICS = {"두괄식", "STAR"}
METRIC_DESCRIPTIONS = {
    "두괄식": {
        "summary": "핵심 결론을 앞에서 먼저 제시하는 구조입니다.",
        "detail": "좋은 자소서는 첫 문장이나 문단 앞부분에서 내가 무엇을 했고 어떤 성과를 냈는지 빠르게 보여줍니다.",
        "check": "첫 문장에 성과, 변화, 직무 적합성이 보이는지 확인합니다.",
    },
    "STAR": {
        "summary": "상황, 과제, 행동, 결과의 경험 서술 구조입니다.",
        "detail": "Situation, Task, Action, Result가 모두 보여야 경험이 단순 주장보다 설득력 있는 근거로 읽힙니다.",
        "check": "배경, 맡은 역할, 실제 행동, 결과가 빠지지 않았는지 확인합니다.",
    },
    "모호도": {
        "summary": "추상적이고 두루뭉술한 표현의 정도입니다.",
        "detail": "열심히, 다양하게, 성장 같은 표현이 많고 수치, 기간, 대상, 행동이 부족하면 모호도가 높아집니다.",
        "check": "추상어를 숫자, 기간, 대상, 행동으로 바꿀 수 있는지 확인합니다.",
    },
    "자기중심": {
        "summary": "나 중심 표현이 과한지 보는 지표입니다.",
        "detail": "제가, 저는 중심의 서술만 반복되면 팀, 고객, 조직에 준 영향이 약해 보일 수 있습니다.",
        "check": "내 행동 이후 팀, 고객, 조직에 어떤 변화가 있었는지 확인합니다.",
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
            .score-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
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
    if "analysis_history" not in st.session_state:
        st.session_state.analysis_history = []


def render_sidebar() -> None:
    with st.sidebar:
        st.header("자소서 분석")
        if st.button("새 분석 시작", use_container_width=True):
            st.session_state.cover_letter_text = ""
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
    render_topbar("자소서 분석", "4개 지표 기반 피드백")

    text = st.session_state.cover_letter_text.strip()
    if text:
        result = analyze_cover_letter(text)
        render_user_message(text)
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
        result = analyze_cover_letter(item["text"])
        st.markdown(
            f"""
            <div class="history-card">
                <b>{escape(item["time"])} · {item["score"]}점</b>
                <p>{escape(item["preview"])}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        cols = st.columns([1, 1, 4])
        with cols[0]:
            if st.button("열기", key=f'open_{item["id"]}'):
                st.session_state.cover_letter_text = item["text"]
                st.session_state.page = "자소서 분석"
                st.rerun()
        with cols[1]:
            if st.button("결과 보기", key=f'view_{item["id"]}'):
                render_assistant_message(result)

    st.divider()
    if st.button("기록 지우기", use_container_width=True):
        st.session_state.analysis_history = []
        st.rerun()


def render_metric_page() -> None:
    render_topbar("분석 지표", "지표 설명")

    st.write("")
    st.markdown(
        "자소서를 평가할 때 보는 네 가지 기준입니다. 각 지표는 문장의 구조와 표현 위험을 다른 관점에서 확인합니다."
    )

    metric_items = list(METRIC_DESCRIPTIONS.items())
    for row_start in range(0, len(metric_items), 2):
        cols = st.columns(2, gap="large")
        for col, (metric, info) in zip(cols, metric_items[row_start : row_start + 2]):
            with col:
                with st.container(border=True):
                    st.subheader(metric)
                    st.markdown(f"**{info['summary']}**")
                    st.write(info["detail"])
                    st.divider()
                    st.caption(info["check"])

    with st.expander("점수는 어떻게 해석하나요?"):
        st.write("두괄식과 STAR는 높을수록 좋고, 모호도와 자기중심은 낮을수록 좋습니다.")
        st.write("현재 화면은 UI와 통합 흐름을 확인하기 위한 형태이며, 최종 모델 출력이 들어오면 같은 화면에 연결하면 됩니다.")


def render_intro_message() -> None:
    st.markdown(
        """
        <div class="message">
            <div class="avatar assistant-avatar">AI</div>
            <div class="bubble">
                <h1>자소서 문장을 붙여넣어 주세요.</h1>
                <p>
                    입력한 문장을 두괄식, STAR, 모호도, 자기중심 지표로 분석하고
                    바로 고칠 수 있는 피드백으로 정리해드립니다.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_input_form() -> None:
    st.markdown('<div class="prompt-shell">', unsafe_allow_html=True)
    with st.form("analysis_form", clear_on_submit=False):
        user_text = st.text_area(
            "자소서 입력",
            value=st.session_state.cover_letter_text,
            label_visibility="collapsed",
            placeholder="자소서 문장을 붙여넣고 분석하기를 눌러주세요.",
            height=155,
        )
        submitted = st.form_submit_button("분석하기", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if submitted:
        cleaned = user_text.strip()
        st.session_state.cover_letter_text = cleaned
        if cleaned:
            save_history(cleaned, analyze_cover_letter(cleaned))
        st.rerun()


def save_history(text: str, result: AnalysisResult) -> None:
    preview = text.replace("\n", " ")[:40]
    if len(text) > 40:
        preview += "..."

    latest = st.session_state.analysis_history[-1] if st.session_state.analysis_history else None
    if latest and latest["text"] == text:
        latest["score"] = result.overall_score
        latest["time"] = datetime.now().strftime("%H:%M")
        latest["preview"] = preview
        return

    st.session_state.analysis_history.append(
        {
            "id": datetime.now().strftime("%Y%m%d%H%M%S%f"),
            "time": datetime.now().strftime("%H:%M"),
            "text": text,
            "score": result.overall_score,
            "preview": preview,
        }
    )


def render_user_message(text: str) -> None:
    preview = escape(text).replace("\n", "<br>")
    st.markdown(
        f"""
        <div class="message">
            <div class="avatar user-avatar">나</div>
            <div class="bubble">{preview}</div>
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
                <h1>분석 결과: {result.overall_score}점</h1>
                <p>{escape(result.summary)}</p>
                {metric_cards_html(result)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    dashboard_tab, feedback_tab, sentence_tab = st.tabs(["대시보드", "개선 피드백", "문장별 코멘트"])
    with dashboard_tab:
        render_dashboard(result)
    with feedback_tab:
        render_feedback(result)
    with sentence_tab:
        render_sentence_feedback(result)


def metric_cards_html(result: AnalysisResult) -> str:
    cards = []
    for metric in result.metrics.values():
        cards.append(
            f"""
            <div class="score-card">
                <div class="score-label">{escape(metric.label)} · {metric_note(metric.label)}</div>
                <div class="score-value">{metric.score}<span style="font-size:0.9rem;">점</span></div>
                <div class="score-level">{escape(metric.level)}</div>
                <div class="meter"><span style="width:{metric.score}%; background:{metric_color(metric.label, metric.score)};"></span></div>
            </div>
            """
        )
    return f'<div class="score-grid">{"".join(cards)}</div>'


def render_dashboard(result: AnalysisResult) -> None:
    metric_rows = []
    for metric in result.metrics.values():
        group = "구조 점수" if metric.label in GOOD_METRICS else "위험 점수"
        metric_rows.append({"지표": metric.label, "점수": metric.score, "구분": group})

    df = pd.DataFrame(metric_rows)
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
        .encode(
            x=alt.X("지표:N", sort=None, axis=alt.Axis(labelAngle=0, title=None)),
            y=alt.Y("점수:Q", scale=alt.Scale(domain=[0, 100]), axis=alt.Axis(title=None)),
            color=alt.Color(
                "구분:N",
                scale=alt.Scale(domain=["구조 점수", "위험 점수"], range=["#10a37f", "#f59e0b"]),
                legend=alt.Legend(orient="top", title=None),
            ),
            tooltip=["지표", "점수", "구분"],
        )
        .properties(height=300)
    )
    labels = chart.mark_text(align="center", baseline="bottom", dy=-6, fontWeight="bold").encode(text="점수:Q")
    st.altair_chart(chart + labels, use_container_width=True)

    summary_df = pd.DataFrame(
        [
            {"영역": "구조", "점수": round((result.metrics["두괄식"].score + result.metrics["STAR"].score) / 2)},
            {
                "영역": "표현 안정성",
                "점수": round(
                    (100 - result.metrics["모호도"].score + 100 - result.metrics["자기중심"].score) / 2
                ),
            },
        ]
    )
    st.dataframe(summary_df, use_container_width=True, hide_index=True)


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


def metric_note(label: str) -> str:
    if label in GOOD_METRICS:
        return "높을수록 좋음"
    return "낮을수록 좋음"


def metric_color(label: str, score: int) -> str:
    if label in GOOD_METRICS:
        if score >= 75:
            return "#10a37f"
        if score >= 55:
            return "#2563eb"
        return "#dc2626"

    if score >= 65:
        return "#dc2626"
    if score >= 35:
        return "#f59e0b"
    return "#10a37f"


if __name__ == "__main__":
    main()
