"""
Deterministic date/time resolver for the CCTV query agent.

Takes the semantic (but unresolved) date/time fields produced by the LLM
parser (QueryFrame) and turns them into concrete boundaries that
query_builder.py can turn into SQL:

- start_datetime / end_datetime : absolute, timezone-aware instants
  (end_datetime is EXCLUSIVE, so ranges are always [start, end)).
- time_start / time_end         : "HH:MM" strings applied WITHIN each day
  of the range (e.g. "8 AM to 10 AM yesterday").
- weekday                       : SQLite strftime('%w') convention
  (0 = Sunday ... 6 = Saturday), for recurring queries.

This module does NOT touch the database, does NOT generate SQL, and does
NOT call the LLM. It is pure, deterministic date arithmetic, which is
what makes it testable and auditable.
"""

import calendar
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

SGT = ZoneInfo("Asia/Singapore")

WEEKDAY_NAME_TO_SQLITE = {
    "sunday": 0,
    "monday": 1,
    "tuesday": 2,
    "wednesday": 3,
    "thursday": 4,
    "friday": 5,
    "saturday": 6,
}

_MONTH_NAMES = {
    name.lower(): i for i, name in enumerate(calendar.month_name) if name
}
_MONTH_ABBR = {
    name.lower(): i for i, name in enumerate(calendar.month_abbr) if name
}
MONTH_NAME_TO_NUMBER = {**_MONTH_NAMES, **_MONTH_ABBR}

_TIME_RE = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")


@dataclass
class DateTimeResolution:
    """Result of resolving a QueryFrame's date/time fields."""

    start_datetime: str | None = None  # ISO 8601, inclusive
    end_datetime: str | None = None  # ISO 8601, exclusive
    time_start: str | None = None  # "HH:MM"
    time_end: str | None = None  # "HH:MM"
    weekday: int | None = None  # 0=Sunday ... 6=Saturday
    recurring: bool = False
    ambiguous: bool = False
    reason: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now() -> datetime:
    return datetime.now(tz=SGT)


def _day_start(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=SGT)


def _to_iso(d: date) -> str:
    return _day_start(d).isoformat()


def _add_months(year: int, month: int, delta: int) -> tuple[int, int]:
    """Return (year, month) for `month` shifted by `delta` months."""
    index = year * 12 + (month - 1) + delta
    return index // 12, index % 12 + 1


def _month_bounds(year: int, month: int) -> tuple[date, date]:
    """[first_of_month, first_of_next_month)."""
    start = date(year, month, 1)
    ny, nm = _add_months(year, month, 1)
    return start, date(ny, nm, 1)


def _valid_time(value: str) -> bool:
    return bool(_TIME_RE.match(value))


def _parse_iso_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value.strip())
    except (ValueError, AttributeError):
        return None


def _parse_named_date(text: str) -> date | None:
    """Parse things like '25 august 2026' or 'august 25 2026' / 'august 25, 2026'."""
    text = text.strip().lower().replace(",", "")

    m = re.fullmatch(r"(\d{1,2})\s+([a-z]+)\s+(\d{4})", text)
    if m:
        day, month_name, year = m.groups()
        month = MONTH_NAME_TO_NUMBER.get(month_name)
        if month:
            try:
                return date(int(year), month, int(day))
            except ValueError:
                return None

    m = re.fullmatch(r"([a-z]+)\s+(\d{1,2})\s+(\d{4})", text)
    if m:
        month_name, day, year = m.groups()
        month = MONTH_NAME_TO_NUMBER.get(month_name)
        if month:
            try:
                return date(int(year), month, int(day))
            except ValueError:
                return None

    return None


def _strip_ordinal(text: str) -> str:
    return re.sub(r"(\d+)(st|nd|rd|th)", r"\1", text)


# ---------------------------------------------------------------------------
# Simple relative-date expressions (whole-day / whole-period ranges)
# ---------------------------------------------------------------------------


def _range_today(now: datetime) -> tuple[date, date]:
    d = now.date()
    return d, d + timedelta(days=1)


def _range_yesterday(now: datetime) -> tuple[date, date]:
    d = now.date() - timedelta(days=1)
    return d, d + timedelta(days=1)


def _range_tomorrow(now: datetime) -> tuple[date, date]:
    d = now.date() + timedelta(days=1)
    return d, d + timedelta(days=1)


def _range_this_week(now: datetime) -> tuple[date, date]:
    monday = now.date() - timedelta(days=now.weekday())
    return monday, monday + timedelta(days=7)


def _range_last_week(now: datetime) -> tuple[date, date]:
    this_monday = now.date() - timedelta(days=now.weekday())
    monday = this_monday - timedelta(days=7)
    return monday, monday + timedelta(days=7)


def _range_next_week(now: datetime) -> tuple[date, date]:
    this_monday = now.date() - timedelta(days=now.weekday())
    monday = this_monday + timedelta(days=7)
    return monday, monday + timedelta(days=7)


def _range_this_month(now: datetime) -> tuple[date, date]:
    return _month_bounds(now.year, now.month)


def _range_last_month(now: datetime) -> tuple[date, date]:
    y, m = _add_months(now.year, now.month, -1)
    return _month_bounds(y, m)


def _range_next_month(now: datetime) -> tuple[date, date]:
    y, m = _add_months(now.year, now.month, 1)
    return _month_bounds(y, m)


_SIMPLE_RELATIVE = {
    "today": _range_today,
    "yesterday": _range_yesterday,
    "tomorrow": _range_tomorrow,
    "this week": _range_this_week,
    "last week": _range_last_week,
    "next week": _range_next_week,
    "this month": _range_this_month,
    "last month": _range_last_month,
    "next month": _range_next_month,
}

_MONTH_REFERENCE_TO_RANGE = {
    "last month": _range_last_month,
    "this month": _range_this_month,
    "next month": _range_next_month,
}


# ---------------------------------------------------------------------------
# Expression parsing (returns an inclusive/exclusive (date, date) range)
# ---------------------------------------------------------------------------


def _resolve_expression(expr: str, now: datetime) -> tuple[date, date] | None:
    expr = _strip_ordinal(expr.strip().lower())

    if expr in _SIMPLE_RELATIVE:
        return _SIMPLE_RELATIVE[expr](now)

    # "last 7 days", "past 7 days", "past week"
    if expr == "past week":
        expr = "last 7 days"

    m = re.fullmatch(r"(?:last|past)\s+(\d+)\s+days?", expr)
    if m:
        n = int(m.group(1))
        end_d = now.date() + timedelta(days=1)
        start_d = now.date() - timedelta(days=n - 1)
        return start_d, end_d

    # "from yesterday to today"
    if expr == "from yesterday to today":
        start_d, _ = _range_yesterday(now)
        _, end_d = _range_today(now)
        return start_d, end_d

    # "the whole of august" / "all of august" (current year assumed)
    m = re.fullmatch(r"(?:the whole of|all of)\s+([a-z]+)", expr)
    if m and m.group(1) in MONTH_NAME_TO_NUMBER:
        month = MONTH_NAME_TO_NUMBER[m.group(1)]
        return _month_bounds(now.year, month)

    # "august 2026" / "august" (month name + optional year, current year assumed)
    m = re.fullmatch(r"([a-z]+)\s*(\d{4})?", expr)
    if m and m.group(1) in MONTH_NAME_TO_NUMBER:
        month = MONTH_NAME_TO_NUMBER[m.group(1)]
        year = int(m.group(2)) if m.group(2) else now.year
        return _month_bounds(year, month)

    # "15 to 18 of last month" / "15 to 18 of this month" / "15-18 of next month"
    m = re.fullmatch(
        r"(\d{1,2})\s*(?:to|-)\s*(\d{1,2})\s+of\s+(last|this|next)\s+month", expr
    )
    if m:
        day_start, day_end, which = m.groups()
        base_start, _ = _MONTH_REFERENCE_TO_RANGE[f"{which} month"](now)
        try:
            start_d = base_start.replace(day=int(day_start))
            end_d = base_start.replace(day=int(day_end)) + timedelta(days=1)
            return start_d, end_d
        except ValueError:
            return None

    # "15 august to 18 august" / "from 15 august to 18 august"
    m = re.fullmatch(
        r"(?:from\s+)?(\d{1,2})\s+([a-z]+)\s+to\s+(\d{1,2})\s+([a-z]+)", expr
    )
    if m:
        d1, mon1, d2, mon2 = m.groups()
        month1 = MONTH_NAME_TO_NUMBER.get(mon1)
        month2 = MONTH_NAME_TO_NUMBER.get(mon2)
        if month1 and month2:
            try:
                start_d = date(now.year, month1, int(d1))
                end_d = date(now.year, month2, int(d2)) + timedelta(days=1)
                return start_d, end_d
            except ValueError:
                return None

    # A single explicit date written out, e.g. "25 august 2026"
    named = _parse_named_date(expr)
    if named:
        return named, named + timedelta(days=1)

    # A single ISO date passed through in date_expression instead of start_date
    iso = _parse_iso_date(expr)
    if iso:
        return iso, iso + timedelta(days=1)

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_datetime(
    date_expression: str | None,
    start_date: str | None,
    end_date: str | None,
    start_time: str | None,
    end_time: str | None,
    weekday: str | None,
    recurring: bool,
) -> DateTimeResolution:
    """
    Resolve a QueryFrame's raw date/time fields into absolute boundaries.

    Precedence: explicit start_date/end_date > date_expression.
    Time-of-day and weekday are resolved independently and can be
    combined with either date source (e.g. "8 AM to 10 AM yesterday",
    "every Tuesday in August").
    """

    result = DateTimeResolution()

    # --- time-of-day -------------------------------------------------
    if start_time:
        if not _valid_time(start_time):
            result.ambiguous = True
            result.reason = f"Invalid start_time: '{start_time}'"
            return result
        result.time_start = start_time

    if end_time:
        if not _valid_time(end_time):
            result.ambiguous = True
            result.reason = f"Invalid end_time: '{end_time}'"
            return result
        result.time_end = end_time

    if (
        result.time_start
        and result.time_end
        and result.time_start >= result.time_end
    ):
        result.ambiguous = True
        result.reason = (
            f"start_time '{result.time_start}' is not before "
            f"end_time '{result.time_end}'"
        )
        return result

    # --- weekday / recurring ------------------------------------------
    if weekday:
        key = weekday.strip().lower()
        if key not in WEEKDAY_NAME_TO_SQLITE:
            result.ambiguous = True
            result.reason = f"Unrecognized weekday: '{weekday}'"
            return result
        result.weekday = WEEKDAY_NAME_TO_SQLITE[key]
        result.recurring = True
    else:
        result.recurring = bool(recurring)

    # --- explicit dates take priority over free-text expressions ------
    if start_date:
        sd = _parse_iso_date(start_date)
        if sd is None:
            result.ambiguous = True
            result.reason = f"Invalid start_date: '{start_date}'"
            return result

        ed = _parse_iso_date(end_date) if end_date else sd
        if ed is None:
            result.ambiguous = True
            result.reason = f"Invalid end_date: '{end_date}'"
            return result
        if ed < sd:
            result.ambiguous = True
            result.reason = f"end_date '{end_date}' is before start_date '{start_date}'"
            return result

        result.start_datetime = _to_iso(sd)
        result.end_datetime = _to_iso(ed + timedelta(days=1))
        return result

    # --- otherwise fall back to the free-text date expression ----------
    if date_expression:
        now = _now()
        span = _resolve_expression(date_expression, now)

        if span is None:
            result.ambiguous = True
            result.reason = f"Could not resolve date expression: '{date_expression}'"
            return result

        start_d, end_d = span
        result.start_datetime = _to_iso(start_d)
        result.end_datetime = _to_iso(end_d)
        return result

    # No date constraint at all (e.g. bare "every Tuesday" with no range,
    # or a camera-only query) — that's valid, just unbounded in time.
    return result


if __name__ == "__main__":
    examples = [
        dict(date_expression="today"),
        dict(date_expression="yesterday", start_time="08:00", end_time="10:00"),
        dict(date_expression="the whole of August"),
        dict(date_expression="15 to 18 of last month"),
        dict(weekday="Tuesday", recurring=True),
        dict(start_date="2026-08-25"),
        dict(date_expression="this week"),
        dict(date_expression="not a real date"),
    ]

    for kwargs in examples:
        kwargs.setdefault("date_expression", None)
        kwargs.setdefault("start_date", None)
        kwargs.setdefault("end_date", None)
        kwargs.setdefault("start_time", None)
        kwargs.setdefault("end_time", None)
        kwargs.setdefault("weekday", None)
        kwargs.setdefault("recurring", False)

        res = resolve_datetime(**kwargs)
        print(kwargs, "->", res)