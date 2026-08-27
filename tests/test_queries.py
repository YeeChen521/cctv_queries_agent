"""
Normal-case tests for the deterministic CCTV query pipeline:
resolver.py -> query_schema.py -> query_builder.py -> database.py.

These tests build the QueryFrame the LLM is expected to produce for
each natural-language example from README.md's "8. Example Queries"
section, and verify the deterministic backbone resolves and executes
it correctly. They intentionally do NOT call the LLM (see
test_followups.py / test_edge_cases.py for the LLM-mocked and
guardrail tests) — this file is about whether resolver.py and
query_builder.py get the *mechanics* right.

Row-count assertions are only made against dates guaranteed to exist in
the generated dataset (data/cctv.db spans 2026-01-01 through whenever
scripts/generate_data.py was last run). "today"/"yesterday" tests
instead assert structural correctness only, since a real deployment's
clock may have moved past the last generated frame.
"""

from datetime import date, timedelta

import pytest

from src.database import execute_query
from src.query_builder import build_sql
from src.query_schema import QueryFrame, build_resolved_query

FRAMES_PER_DAY = 24 * 60 // 5  # one frame every 5 minutes


def _run(frame: QueryFrame):
    resolved = build_resolved_query(frame)
    assert resolved.is_valid, resolved.rejection_reason
    sql, params = build_sql(resolved)
    rows = execute_query(sql, params)
    return resolved, sql, params, rows


# ---------------------------------------------------------------------------
# Basic camera + relative date
# ---------------------------------------------------------------------------

def test_camera_today_resolves_structurally():
    # "Show me frames from CTE today."
    frame = QueryFrame(intent="retrieve_frames", camera="CTE", date_expression="today")
    resolved, sql, params, rows = _run(frame)

    assert resolved.camera == "Central Expressway (CTE)"
    assert resolved.start_datetime is not None
    assert resolved.end_datetime is not None
    assert isinstance(rows, list)  # may be empty if "today" is past the dataset


def test_full_camera_name_resolves_same_as_code():
    # "Show me frames from Central Expressway today."
    frame = QueryFrame(
        intent="retrieve_frames", camera="Central Expressway", date_expression="today"
    )
    resolved, *_ = _run(frame)
    assert resolved.camera == "Central Expressway (CTE)"


def test_camera_yesterday_resolves_structurally():
    # "Show me PIE frames yesterday."
    frame = QueryFrame(intent="retrieve_frames", camera="PIE", date_expression="yesterday")
    resolved, sql, params, rows = _run(frame)
    assert resolved.camera == "Pan Island Expressway (PIE)"
    assert isinstance(rows, list)


# ---------------------------------------------------------------------------
# Month
# ---------------------------------------------------------------------------

def test_whole_month_with_explicit_year():
    # "Show me frames from TPE for the whole of August."
    # Using an explicit year in the expression (rather than "the whole
    # of August", which assumes the current year) keeps this test valid
    # regardless of when it's run.
    frame = QueryFrame(intent="retrieve_frames", camera="TPE", date_expression="August 2026")
    resolved, sql, params, rows = _run(frame)

    assert resolved.camera == "Tampines Expressway (TPE)"
    assert resolved.start_datetime.startswith("2026-08-01")
    assert resolved.end_datetime.startswith("2026-09-01")
    assert len(rows) > 0  # August 1st onward is well within the generated range


# ---------------------------------------------------------------------------
# Explicit single date — exact, independently-verifiable count
# ---------------------------------------------------------------------------

def test_explicit_single_date_returns_exact_frame_count():
    frame = QueryFrame(intent="retrieve_frames", camera="CTE", start_date="2026-06-15")
    resolved, sql, params, rows = _run(frame)

    assert resolved.start_datetime == "2026-06-15T00:00:00+08:00"
    assert resolved.end_datetime == "2026-06-16T00:00:00+08:00"
    assert len(rows) == FRAMES_PER_DAY  # exactly one full day, one camera


# ---------------------------------------------------------------------------
# Time range — exact, independently-verifiable count
# ---------------------------------------------------------------------------

def test_time_range_returns_exact_frame_count():
    # Mechanically equivalent to "PIE frames between 8 AM and 10 AM
    # yesterday", but pinned to a historical date so the count is exact
    # regardless of when the dataset was last generated.
    frame = QueryFrame(
        intent="retrieve_frames",
        camera="PIE",
        start_date="2026-06-15",
        start_time="08:00",
        end_time="10:00",
    )
    resolved, sql, params, rows = _run(frame)

    expected = (10 - 8) * 60 // 5  # 2 hours of 5-minute frames
    assert len(rows) == expected
    assert all(row["datetime"][11:16] >= "08:00" for row in rows)
    assert all(row["datetime"][11:16] < "10:00" for row in rows)


def test_time_range_uses_substr_not_strftime_semantics():
    """
    Regression test: SQLite's strftime() converts +08:00 offset
    timestamps to UTC before extracting fields, which would silently
    shift every time-of-day comparison by 8 hours. query_builder.py
    must use substr() instead. This test fails if that regresses.
    """
    frame = QueryFrame(
        intent="retrieve_frames",
        camera="PIE",
        start_date="2026-06-15",
        start_time="08:00",
        end_time="08:05",
    )
    resolved, sql, params, rows = _run(frame)
    assert len(rows) == 1
    assert rows[0]["datetime"] == "2026-06-15T08:00:00+08:00"


# ---------------------------------------------------------------------------
# Date range ("15th to 18th of last month")
# ---------------------------------------------------------------------------

def test_date_range_last_month():
    frame = QueryFrame(
        intent="retrieve_frames",
        camera="Kranji Highway",
        date_expression="15th to 18th of last month",
    )
    resolved, sql, params, rows = _run(frame)

    assert resolved.camera == "Kranji Expressway (KJE)"
    # 4 inclusive days (15, 16, 17, 18) — every month has at least 18
    # days, so this is valid regardless of which month "last month" is.
    assert resolved.limit is not None
    assert len(rows) == min(resolved.limit, FRAMES_PER_DAY * 4)


# ---------------------------------------------------------------------------
# Recurring weekday — exact, independently-verifiable count
# ---------------------------------------------------------------------------

def test_recurring_weekday_exact_count_within_bounded_month():
    # "Show me frames from MCE on every Tuesday", bounded to June 2026
    # so the expected count can be computed independently below rather
    # than trusting the system under test to say how many Tuesdays
    # there were.
    frame = QueryFrame(
        intent="retrieve_frames",
        camera="MCE",
        weekday="Tuesday",
        recurring=True,
        start_date="2026-06-01",
        end_date="2026-06-30",
    )
    resolved, sql, params, rows = _run(frame)

    tuesdays_in_june = sum(
        1
        for offset in range(30)
        if (date(2026, 6, 1) + timedelta(days=offset)).weekday() == 1  # Mon=0, Tue=1
    )
    expected_count = tuesdays_in_june * FRAMES_PER_DAY
    assert resolved.limit is not None
    assert len(rows) == min(resolved.limit, expected_count)

    for row in rows:
        d = date.fromisoformat(row["datetime"][:10])
        assert d.weekday() == 1  # every returned row really is a Tuesday


# ---------------------------------------------------------------------------
# Typo
# ---------------------------------------------------------------------------

def test_typo_resolves_to_canonical_camera():
    # "Show me frames from Tampines Expresway."
    frame = QueryFrame(intent="retrieve_frames", camera="Tampines Expresway")
    resolved, *_ = _run(frame)
    assert resolved.camera == "Tampines Expressway (TPE)"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))