"""
Normalized, fully-resolved representation of a user query.

This is the single object query_builder.py consumes. By the time a
ResolvedQuery exists, all ambiguity has been settled:

- camera is either one canonical camera name or None (no camera filter).
- datetime bounds, if present, are absolute, concrete instants.
- time-of-day bounds (if present) are "HH:MM" strings applied within
  every day of the range.
- weekday is a SQLite strftime('%w') integer (0=Sunday ... 6=Saturday).

query_builder.py never has to interpret natural language, resolve
aliases, or reason about dates — that all happens upstream, here.
"""

from pydantic import BaseModel, Field

from .model import QueryFrame
from .resolver import resolve_camera
from .datetime_resolver import resolve_datetime

DEFAULT_ROW_LIMIT = 1000


class ResolvedQuery(BaseModel):
    """Deterministic, ready-to-build query representation."""

    is_valid: bool = True
    rejection_reason: str | None = None

    camera: str | None = None

    start_datetime: str | None = None  # ISO 8601, inclusive
    end_datetime: str | None = None  # ISO 8601, exclusive

    time_start: str | None = None  # "HH:MM"
    time_end: str | None = None  # "HH:MM"

    weekday: int | None = Field(default=None, ge=0, le=6)
    recurring: bool = False

    limit: int | None = DEFAULT_ROW_LIMIT

    @classmethod
    def rejected(cls, reason: str) -> "ResolvedQuery":
        return cls(is_valid=False, rejection_reason=reason)


def build_resolved_query(frame: QueryFrame) -> ResolvedQuery:
    """
    Fuse the LLM's QueryFrame with the deterministic camera and
    date/time resolvers into a single ResolvedQuery.

    This function is the only place camera resolution and date/time
    resolution results are combined, and the only place guardrail-style
    rejections for "we couldn't safely interpret this" are produced.
    Explicit malicious-intent guardrails (SQL injection, prompt
    injection, etc.) belong in guardrails.py and should run before this
    is ever called.
    """

    if frame.intent == "unsupported":
        return ResolvedQuery.rejected(
            "This request is outside the scope of the CCTV retrieval system."
        )

    if frame.intent == "clarification_needed" or frame.needs_clarification:
        return ResolvedQuery.rejected(
            frame.clarification_reason
            or "The request is ambiguous and needs clarification."
        )

    canonical_camera: str | None = None
    if frame.camera:
        canonical_camera = resolve_camera(frame.camera)
        if canonical_camera is None:
            return ResolvedQuery.rejected(
                f"Could not recognize camera '{frame.camera}'. "
                "Please provide a supported expressway camera name."
            )

    dt = resolve_datetime(
        date_expression=frame.date_expression,
        start_date=frame.start_date,
        end_date=frame.end_date,
        start_time=frame.start_time,
        end_time=frame.end_time,
        weekday=frame.weekday,
        recurring=frame.recurring,
    )

    if dt.ambiguous:
        return ResolvedQuery.rejected(
            dt.reason or "Could not resolve the requested date/time."
        )

    return ResolvedQuery(
        camera=canonical_camera,
        start_datetime=dt.start_datetime,
        end_datetime=dt.end_datetime,
        time_start=dt.time_start,
        time_end=dt.time_end,
        weekday=dt.weekday,
        recurring=dt.recurring,
    )