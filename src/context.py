"""
In-memory conversational context for the CCTV query agent.

Keeps just enough state to support follow-up requests like:

    User: "Show me frames from CTE."
    User: "How about only this week?"

Two things are tracked:

1. A short transcript of recent user turns, passed to the LLM parser so
   it can resolve references like "that camera" / "only this week".
2. The last successfully resolved QueryFrame, used as a deterministic
   fallback: if the LLM's own context handling misses a carried-over
   constraint, we fill it in ourselves rather than silently dropping
   it. This is the kind of "deterministic logic to improve robustness"
   the assignment calls for — not an agentic framework of its own.

This module is deliberately UI-framework-agnostic: it doesn't know
about Streamlit's session state. The UI layer (main.py, via agent.py)
is responsible for keeping one ConversationContext alive per session
(e.g. by storing the owning QueryAgent in st.session_state) so the
same backend can later be reused behind a CLI or API unchanged.
"""

from dataclasses import dataclass, field

from .query_schema import QueryFrame

MAX_HISTORY_TURNS = 6

# Fields that make sense to carry forward from one turn to the next.
_CARRYABLE_FIELDS = (
    "camera",
    "date_expression",
    "start_date",
    "end_date",
    "start_time",
    "end_time",
    "weekday",
    "recurring",
)


@dataclass
class ConversationContext:
    """Holds recent turns and the last resolved query for one session."""

    turns: list[str] = field(default_factory=list)
    last_frame: QueryFrame | None = None

    def add_turn(self, user_message: str) -> None:
        """Record a user turn, keeping only the most recent ones."""
        self.turns.append(user_message)
        self.turns = self.turns[-MAX_HISTORY_TURNS:]

    def as_context_string(self) -> str | None:
        """
        Recent turns (excluding the current one), formatted for
        llm_parser.parse_query's conversation_context argument.
        """
        if len(self.turns) <= 1:
            return None
        previous_turns = self.turns[:-1]
        return "\n".join(f"- {turn}" for turn in previous_turns)

    def merge(self, frame: QueryFrame) -> QueryFrame:
        """
        Fill in any constraint the LLM left blank on a follow-up using
        the last successfully resolved frame.

        For example, "how about this week?" should keep the camera
        from the previous turn even if the LLM's own context handling
        missed it. Only applied to retrieval requests — unsupported or
        clarification-needed frames are returned unchanged, since there
        is nothing meaningful to merge into them.

        Crucially, this only fires when the new message does NOT name
        its own camera. Naming a camera is treated as switching to a
        new subject, so a stale date/time/weekday from the previous
        turn is never silently carried over onto it — e.g. after "Show
        me CTE this week", a later "Show me frames from MCE on every
        Tuesday" must NOT inherit "this week" just because it left the
        date field blank. Without this guard, that's exactly what
        would happen.
        """
        if frame.intent != "retrieve_frames" or self.last_frame is None:
            return frame

        if frame.camera:
            # A new camera was named: treat this as a fresh query scope
            # rather than a continuation of the previous one.
            return frame

        merged_values = frame.model_dump()
        previous_values = self.last_frame.model_dump()

        for key in _CARRYABLE_FIELDS:
            new_value = merged_values.get(key)
            old_value = previous_values.get(key)
            if new_value in (None, False) and old_value not in (None, False):
                merged_values[key] = old_value

        return QueryFrame(**merged_values)

    def remember(self, frame: QueryFrame) -> None:
        """Record a successfully resolved frame for future follow-ups."""
        if frame.intent == "retrieve_frames":
            self.last_frame = frame

    def reset(self) -> None:
        """Clear all conversational state (e.g. on a 'new chat' action)."""
        self.turns.clear()
        self.last_frame = None


# ============================================================================
# Simple local test
# ============================================================================

if __name__ == "__main__":
    ctx = ConversationContext()

    ctx.add_turn("Show me frames from CTE.")
    first = QueryFrame(intent="retrieve_frames", camera="CTE")
    ctx.remember(first)

    ctx.add_turn("How about only this week?")
    followup = QueryFrame(intent="retrieve_frames", date_expression="this week")
    merged = ctx.merge(followup)

    print(merged.model_dump_json(indent=2))
    assert merged.camera == "CTE"
    assert merged.date_expression == "this week"
    print("OK")