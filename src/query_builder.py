"""
Deterministic SQL builder for the CCTV query agent.

Takes a fully-resolved ResolvedQuery and turns it into a parameterized,
read-only SQL SELECT statement against cctv_frames.

Every value derived from the user ends up as a bound parameter (`?`),
never interpolated into the SQL text. The only thing that varies
structurally between queries is which WHERE clauses are included, all
of which come from a fixed, hard-coded set below. This module cannot
be coerced into emitting anything other than a single SELECT against
the one allowed table.
"""

from config.metadata import DEFAULT_RESULT_ORDER, QUERY_RULES, RESULT_COLUMNS

from .query_schema import ResolvedQuery

TABLE = QUERY_RULES["allowed_table"]


class QueryBuildError(Exception):
    """Raised when a ResolvedQuery cannot safely be turned into SQL."""


def build_sql(query: ResolvedQuery) -> tuple[str, list]:
    """
    Build a parameterized SQL SELECT statement for a ResolvedQuery.

    Returns:
        (sql, params) - sql contains only "?" placeholders; params is
        the ordered list of values to bind to them.

    Raises:
        QueryBuildError: if the query was already rejected upstream
        (invalid camera, ambiguous date, unsupported/rejected intent).
    """

    if not query.is_valid:
        raise QueryBuildError(
            query.rejection_reason or "Query was rejected upstream."
        )

    columns = ", ".join(RESULT_COLUMNS)
    clauses: list[str] = []
    params: list = []

    if query.camera:
        clauses.append("camera_name = ?")
        params.append(query.camera)

    if query.start_datetime:
        clauses.append("datetime >= ?")
        params.append(query.start_datetime)

    if query.end_datetime:
        clauses.append("datetime < ?")
        params.append(query.end_datetime)

    # Time-of-day filter applies within every day of the range (or the
    # whole dataset, if no date range was given).
    #
    # NOTE: `datetime` is stored as a fixed-width ISO 8601 string with a
    # +08:00 offset, e.g. "2026-08-15T08:00:00+08:00". We deliberately use
    # substr() rather than strftime('%H:%M', datetime): SQLite's strftime
    # silently converts offset timestamps to UTC before extracting fields,
    # which would shift every time-of-day/weekday comparison by 8 hours.
    # substr() reads the wall-clock value straight out of the fixed-format
    # string, which is what "8 AM local time" actually means here.
    if query.time_start:
        clauses.append("substr(datetime, 12, 5) >= ?")
        params.append(query.time_start)

    if query.time_end:
        clauses.append("substr(datetime, 12, 5) < ?")
        params.append(query.time_end)

    # Recurring weekday filter, e.g. "every Tuesday". strftime('%w', ...) is
    # safe here because we pass it only the date portion (no offset), so no
    # UTC conversion happens.
    if query.weekday is not None:
        clauses.append("CAST(strftime('%w', substr(datetime, 1, 10)) AS INTEGER) = ?")
        params.append(query.weekday)

    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    sql_parts = [f"SELECT {columns}", f"FROM {TABLE}"]
    if where_sql:
        sql_parts.append(where_sql)
    sql_parts.append(f"ORDER BY {DEFAULT_RESULT_ORDER}")

    sql = "\n".join(sql_parts)

    if query.limit:
        sql += "\nLIMIT ?"
        params.append(query.limit)

    return sql, params


# ============================================================================
# Simple local test
# ============================================================================

if __name__ == "__main__":
    examples = [
        ResolvedQuery(
            camera="Central Expressway (CTE)",
            start_datetime="2026-08-26T00:00:00+08:00",
            end_datetime="2026-08-27T00:00:00+08:00",
        ),
        ResolvedQuery(
            camera="Pan Island Expressway (PIE)",
            start_datetime="2026-08-25T00:00:00+08:00",
            end_datetime="2026-08-26T00:00:00+08:00",
            time_start="08:00",
            time_end="10:00",
        ),
        ResolvedQuery(
            camera="Marina Coastal Expressway (MCE)",
            weekday=2,
            recurring=True,
        ),
        ResolvedQuery.rejected("Could not recognize camera 'Mars Expressway'."),
    ]

    for query in examples:
        print("\n" + "=" * 70)
        try:
            sql, params = build_sql(query)
            print(sql)
            print("PARAMS:", params)
        except QueryBuildError as exc:
            print("REJECTED:", exc)