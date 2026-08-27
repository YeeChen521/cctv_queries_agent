"""
Query agent: the orchestration layer for the CCTV query agent.

This is the only component the UI (main.py) talks to. It wires together
every other module in the pipeline:

    raw guardrail check
        -> llm_parser             (NL -> QueryFrame)
        -> context merge          (fill in follow-up constraints)
        -> query_schema.resolver   (camera + date/time -> ResolvedQuery,
                                     semantic guardrail rejection)
        -> query_builder           (ResolvedQuery -> parameterized SQL)
        -> SQL guardrail check
        -> database                (execute, read-only)

One QueryAgent instance holds the conversation state (via a
ConversationContext) for a single chat session. The UI layer creates
one instance per session and calls .run() for every user message; it
never talks to the LLM, resolver, or database directly, which is what
keeps main.py thin and lets the same backend be reused behind a CLI or
API later without changes.
"""

from dataclasses import dataclass, field

from . import guardrails
from .context import ConversationContext
from .database import DatabaseError, execute_query
from .llm_parser import parse_query
from .query_builder import QueryBuildError, build_sql
from .query_schema import ResolvedQuery, build_resolved_query

MAX_DISPLAY_ROWS = 50


@dataclass
class AgentResponse:
    """Everything the UI needs to render one turn of the conversation."""

    reply: str
    rows: list[dict] = field(default_factory=list)
    row_count: int = 0

    # "Query Details" debug panel info (all optional; None when not applicable).
    intent: str | None = None
    camera: str | None = None
    start_datetime: str | None = None
    end_datetime: str | None = None
    time_start: str | None = None
    time_end: str | None = None
    weekday: int | None = None
    sql: str | None = None
    params: list | None = None
    error: str | None = None


class QueryAgent:
    """Coordinates one conversation's worth of CCTV queries."""

    def __init__(self) -> None:
        self.context = ConversationContext()

    def run(self, user_message: str) -> AgentResponse:
        """Process one user message end-to-end and return the result."""

        # 1. Cheap, deterministic screen on the raw text, before the LLM
        #    ever sees it. Catches SQL/prompt injection and requests for
        #    sensitive info without relying on the model's judgement.
        input_check = guardrails.check_input(user_message)
        if not input_check.allowed:
            return AgentResponse(
                reply=f"I can't help with that: {input_check.reason}",
                error=input_check.reason,
            )

        self.context.add_turn(user_message)

        # 2. Natural language -> structured QueryFrame.
        frame = parse_query(
            user_message,
            conversation_context=self.context.as_context_string(),
        )

        # 3. Fill in follow-up constraints the LLM may have dropped.
        frame = self.context.merge(frame)

        if frame.intent == "unsupported":
            return AgentResponse(
                reply="That request is outside what this system can help with.",
                intent=frame.intent,
                error="unsupported_intent",
            )

        if frame.intent == "clarification_needed" or frame.needs_clarification:
            reason = frame.clarification_reason or "Could you clarify your request?"
            return AgentResponse(reply=reason, intent=frame.intent, error=reason)

        # 4. Deterministic camera + date/time resolution. This is also
        #    where "we couldn't safely interpret this" rejections surface
        #    (unrecognized camera, ambiguous date, invalid time range).
        resolved = build_resolved_query(frame)
        if not resolved.is_valid:
            return AgentResponse(
                reply=f"I couldn't complete that request: {resolved.rejection_reason}",
                intent=frame.intent,
                camera=frame.camera,
                error=resolved.rejection_reason,
            )

        # 5. ResolvedQuery -> parameterized SQL.
        try:
            sql, params = build_sql(resolved)
        except QueryBuildError as exc:
            return AgentResponse(
                reply=f"I couldn't complete that request: {exc}",
                intent=frame.intent,
                error=str(exc),
            )

        # 6. Defense-in-depth check on the generated SQL text itself.
        #    query_builder.py should never fail this; it's a backstop.
        sql_check = guardrails.check_sql(sql)
        if not sql_check.allowed:
            return AgentResponse(
                reply=f"I couldn't complete that request: {sql_check.reason}",
                intent=frame.intent,
                sql=sql,
                error=sql_check.reason,
            )

        # 7. Execute against the read-only database.
        try:
            rows = execute_query(sql, params)
        except DatabaseError as exc:
            return AgentResponse(
                reply=f"Something went wrong reading the database: {exc}",
                intent=frame.intent,
                sql=sql,
                params=params,
                error=str(exc),
            )

        # Only remember constraints from a request that actually succeeded,
        # so a rejected/ambiguous turn doesn't poison future follow-ups.
        self.context.remember(frame)

        return AgentResponse(
            reply=_summarize(resolved, rows),
            rows=rows[:MAX_DISPLAY_ROWS],
            row_count=len(rows),
            intent=frame.intent,
            camera=resolved.camera,
            start_datetime=resolved.start_datetime,
            end_datetime=resolved.end_datetime,
            time_start=resolved.time_start,
            time_end=resolved.time_end,
            weekday=resolved.weekday,
            sql=sql,
            params=params,
        )


def _summarize(resolved: ResolvedQuery, rows: list[dict]) -> str:
    """Build the human-facing reply text for a successful query."""

    if not rows:
        return "No matching frames were found."

    # rows was capped at resolved.limit by query_builder's LIMIT clause,
    # so hitting that count exactly means there may be more matches than
    # what's shown — say so rather than implying it's an exact total.
    hit_limit = resolved.limit is not None and len(rows) >= resolved.limit
    count_phrase = f"at least {len(rows):,}" if hit_limit else f"{len(rows):,}"
    parts = [f"Found {count_phrase} matching frame(s)"]

    if resolved.camera:
        parts.append(f"from {resolved.camera}")

    if resolved.start_datetime and resolved.end_datetime:
        parts.append(
            f"between {resolved.start_datetime[:10]} and {resolved.end_datetime[:10]}"
        )

    if resolved.time_start and resolved.time_end:
        parts.append(f"({resolved.time_start}-{resolved.time_end} daily)")

    return " ".join(parts) + "."