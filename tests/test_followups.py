"""
Conversation-state tests for context.py and agent.py's follow-up
handling, per README.md's documented flows:

    "Show me CTE."           -> "How about this week?"
    "Show me PIE yesterday." -> "Only between 8 and 10."

The LLM is mocked throughout (via src.agent.parse_query) so these tests
are deterministic and don't require network access or a real API key —
they test context.py's merge logic and agent.py's orchestration, not
the LLM's language understanding.
"""

from unittest.mock import patch

import pytest

from src.agent import QueryAgent
from src.context import ConversationContext
from src.query_schema import QueryFrame


# ---------------------------------------------------------------------------
# context.py — unit-level merge behaviour
# ---------------------------------------------------------------------------

def test_context_merges_missing_fields_from_last_frame():
    ctx = ConversationContext()
    ctx.add_turn("Show me frames from CTE.")
    ctx.remember(QueryFrame(intent="retrieve_frames", camera="CTE"))

    ctx.add_turn("How about only this week?")
    followup = QueryFrame(intent="retrieve_frames", date_expression="this week")
    merged = ctx.merge(followup)

    assert merged.camera == "CTE"
    assert merged.date_expression == "this week"


def test_context_does_not_carry_date_onto_a_new_camera():
    """
    Regression test: naming a new camera should start a fresh query
    scope, not silently inherit a stale date from the previous turn.
    """
    ctx = ConversationContext()
    ctx.add_turn("Show me frames from CTE this week.")
    ctx.remember(
        QueryFrame(intent="retrieve_frames", camera="CTE", date_expression="this week")
    )

    ctx.add_turn("Show me frames from MCE on every Tuesday.")
    followup = QueryFrame(
        intent="retrieve_frames", camera="MCE", weekday="Tuesday", recurring=True
    )
    merged = ctx.merge(followup)

    assert merged.camera == "MCE"
    assert merged.date_expression is None  # must NOT inherit "this week"


def test_context_does_not_merge_unsupported_or_clarification_frames():
    ctx = ConversationContext()
    ctx.remember(QueryFrame(intent="retrieve_frames", camera="CTE"))

    unsupported = QueryFrame(intent="unsupported")
    assert ctx.merge(unsupported).camera is None

    clarification = QueryFrame(intent="clarification_needed", needs_clarification=True)
    assert ctx.merge(clarification).camera is None


def test_context_reset_clears_state():
    ctx = ConversationContext()
    ctx.add_turn("Show me frames from CTE.")
    ctx.remember(QueryFrame(intent="retrieve_frames", camera="CTE"))

    ctx.reset()

    assert ctx.turns == []
    assert ctx.last_frame is None


# ---------------------------------------------------------------------------
# agent.py — end-to-end follow-up conversations (LLM mocked)
# ---------------------------------------------------------------------------

def test_followup_retains_camera_and_adds_date():
    llm_outputs = {
        "Show me frames from CTE.": QueryFrame(intent="retrieve_frames", camera="CTE"),
        "How about only those from this week?": QueryFrame(
            intent="retrieve_frames", date_expression="this week"
        ),
    }

    with patch(
        "src.agent.parse_query",
        side_effect=lambda q, conversation_context=None: llm_outputs[q],
    ):
        agent = QueryAgent()

        first = agent.run("Show me frames from CTE.")
        assert first.camera == "Central Expressway (CTE)"
        assert first.error is None

        second = agent.run("How about only those from this week?")
        assert second.camera == "Central Expressway (CTE)"
        assert second.start_datetime is not None
        assert second.error is None


def test_followup_retains_camera_and_date_when_only_time_added():
    llm_outputs = {
        "Show me PIE frames yesterday.": QueryFrame(
            intent="retrieve_frames", camera="PIE", date_expression="yesterday"
        ),
        "Only between 8 and 10.": QueryFrame(
            intent="retrieve_frames", start_time="08:00", end_time="10:00"
        ),
    }

    with patch(
        "src.agent.parse_query",
        side_effect=lambda q, conversation_context=None: llm_outputs[q],
    ):
        agent = QueryAgent()

        first = agent.run("Show me PIE frames yesterday.")
        assert first.camera == "Pan Island Expressway (PIE)"

        second = agent.run("Only between 8 and 10.")
        assert second.camera == "Pan Island Expressway (PIE)"  # carried forward
        assert second.time_start == "08:00"
        assert second.time_end == "10:00"
        assert second.start_datetime == first.start_datetime  # date carried forward too


def test_conversation_context_is_passed_to_llm_on_followup():
    """The agent should hand the LLM parser a transcript of prior turns."""

    seen_contexts = []

    def fake_parse_query(user_query, conversation_context=None):
        seen_contexts.append(conversation_context)
        if user_query == "Show me frames from CTE.":
            return QueryFrame(intent="retrieve_frames", camera="CTE")
        return QueryFrame(intent="retrieve_frames", date_expression="this week")

    with patch("src.agent.parse_query", side_effect=fake_parse_query):
        agent = QueryAgent()
        agent.run("Show me frames from CTE.")
        agent.run("How about only this week?")

    assert seen_contexts[0] is None  # no prior turns on the first message
    assert seen_contexts[1] is not None
    assert "Show me frames from CTE." in seen_contexts[1]


def test_rejected_turn_is_not_remembered_for_future_followups():
    """
    A follow-up after a rejected/ambiguous request shouldn't inherit
    anything from it — there was nothing valid to carry forward.
    """
    llm_outputs = {
        "Show me frames from Mars Expressway.": QueryFrame(
            intent="retrieve_frames", camera="Mars Expressway"
        ),
        "How about this week?": QueryFrame(intent="retrieve_frames", date_expression="this week"),
    }

    with patch(
        "src.agent.parse_query",
        side_effect=lambda q, conversation_context=None: llm_outputs[q],
    ):
        agent = QueryAgent()

        first = agent.run("Show me frames from Mars Expressway.")
        assert first.error is not None  # unresolvable camera, rejected

        second = agent.run("How about this week?")
        assert second.camera is None  # nothing valid was remembered to inherit


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))