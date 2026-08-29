"""
Streamlit entry point for the CCTV query agent.

This module is deliberately thin. It is responsible only for:

- Initializing the Streamlit interface.
- Maintaining chat history and the QueryAgent instance in session state.
- Receiving user messages and handing them to QueryAgent.run().
- Rendering the response, and optionally a "Query Details" debug panel.

No date resolution, SQL generation, or database logic lives here — all
of that is in agent.py and the modules it orchestrates.

Run with:
    streamlit run src/main.py
"""

import sys
from pathlib import Path

# Streamlit runs this file with its own directory (src/) inserted into
# sys.path, not the project root — so "import src.agent" fails with
# "No module named 'src'" even though the src/ package is right there.
# Adding the project root (this file's grandparent) here makes the
# import work regardless of the current working directory or how
# Streamlit was launched.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.agent import AgentResponse, QueryAgent

st.set_page_config(page_title="CCTV Query Agent", page_icon="🚗")


# ============================================================================
# Session state
# ============================================================================

if "agent" not in st.session_state:
    st.session_state.agent = QueryAgent()

if "messages" not in st.session_state:
    # Each entry: {"role": "user" | "assistant", "content": str, "details": AgentResponse | None}
    st.session_state.messages = []


def _reset_conversation() -> None:
    st.session_state.agent = QueryAgent()
    st.session_state.messages = []


# ============================================================================
# Sidebar
# ============================================================================

with st.sidebar:
    st.title("CCTV Query Agent")
    st.caption(
        "Ask about CCTV frame records from Singapore expressway cameras "
        "in plain English."
    )
    show_debug = st.toggle("Show query details", value=False)
    st.button("New conversation", on_click=_reset_conversation)

    with st.expander("Example queries"):
        st.markdown(
            "- Show me frames from CTE today.\n"
            "- Show me PIE frames yesterday.\n"
            "- Show me frames from TPE for the whole of August.\n"
            "- Show me frames from Kranji Highway from the 15th to the "
            "18th of last month.\n"
            "- Show me PIE frames between 8 AM and 10 AM yesterday.\n"
            "- Show me frames from MCE on every Tuesday.\n"
            "- Show me frames from Tampines Expresway."
        )


# ============================================================================
# Rendering helpers
# ============================================================================

def _render_details(details: AgentResponse) -> None:
    with st.expander("Query Details"):
        st.markdown(f"**Detected intent:** `{details.intent}`")

        if details.error:
            st.markdown(f"**Rejection reason:** {details.error}")

        if details.camera:
            st.markdown(f"**Resolved camera:** {details.camera}")

        if details.start_datetime and details.end_datetime:
            st.markdown(
                f"**Resolved date range:** {details.start_datetime} → "
                f"{details.end_datetime}"
            )

        if details.time_start and details.time_end:
            st.markdown(
                f"**Time-of-day filter:** {details.time_start}-{details.time_end}"
            )

        if details.weekday is not None:
            st.markdown(f"**Weekday filter (0=Sun...6=Sat):** {details.weekday}")

        if details.sql:
            st.markdown("**Generated SQL:**")
            st.code(details.sql, language="sql")
            st.markdown(f"**Params:** `{details.params}`")

        if details.row_count:
            st.markdown(f"**Rows matched:** {details.row_count:,}")


def _render_rows(rows: list[dict]) -> None:
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)


# ============================================================================
# Chat history
# ============================================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        details = message.get("details")
        if details is not None:
            if details.rows:
                _render_rows(details.rows)
            if show_debug:
                _render_details(details)


# ============================================================================
# Chat input
# ============================================================================

user_message = st.chat_input("Ask about CCTV frames...")

if user_message:
    st.session_state.messages.append(
        {"role": "user", "content": user_message, "details": None}
    )
    with st.chat_message("user"):
        st.markdown(user_message)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.agent.run(user_message)

        st.markdown(response.reply)
        if response.rows:
            _render_rows(response.rows)
        if show_debug:
            _render_details(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response.reply, "details": response}
    )