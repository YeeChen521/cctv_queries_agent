"""
Raw output schema for the LLM semantic parser.

QueryFrame is intentionally "dumb": it holds whatever the LLM extracted
from the user's natural-language request, in largely un-resolved form
(camera name as typed by the user, date expressions as phrases, etc).

No validation, normalization, or business logic lives here. Resolving
this into something a SQL query can be built from is the job of
resolver.py (camera) + datetime_resolver.py (date/time), which together
produce a query_schema.ResolvedQuery.
"""

from typing import Literal

from pydantic import BaseModel


class QueryFrame(BaseModel):
    """Structured, but not yet resolved, interpretation of a user request."""

    intent: Literal["retrieve_frames", "unsupported", "clarification_needed"]

    # Camera, exactly as worded by the user (e.g. "Kranji Highway", "CTE").
    camera: str | None = None

    # Relative / natural-language date expressions the LLM chose not to
    # resolve itself (e.g. "yesterday", "15th to 18th of last month").
    date_expression: str | None = None

    # Explicit dates only, already normalized to ISO (YYYY-MM-DD) by the LLM.
    start_date: str | None = None
    end_date: str | None = None

    # Clearly-stated times of day, normalized to HH:MM by the LLM.
    start_time: str | None = None
    end_time: str | None = None

    # Recurring weekday conditions, e.g. "every Tuesday".
    weekday: str | None = None
    recurring: bool = False

    # Set when intent == "clarification_needed".
    needs_clarification: bool = False
    clarification_reason: str | None = None