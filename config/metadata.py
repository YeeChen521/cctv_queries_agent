"""
Metadata used by the natural-language query agent.

This file provides the LLM and deterministic resolver with:
- Database schema
- Canonical camera names
- Camera aliases and common variations
- Supported date/time expressions
- Supported query capabilities
- Query restrictions and guardrails
- Few-shot examples

The metadata is intentionally lightweight. It acts as a small,
structured knowledge base rather than a vector-based RAG system.
"""


# ============================================================================
# DATABASE SCHEMA
# ============================================================================

DATABASE_SCHEMA = {
    "table": "cctv_frames",
    "description": (
        "CCTV frame records captured every 5 minutes from cameras "
        "located on Singapore expressways."
    ),
    "columns": {
        "frame_id": {
            "type": "INTEGER",
            "description": "Unique identifier for each CCTV frame.",
        },
        "datetime": {
            "type": "TEXT",
            "description": (
                "Frame capture timestamp in ISO 8601 format with "
                "Singapore UTC+08:00 timezone."
            ),
        },
        "camera_name": {
            "type": "TEXT",
            "description": (
                "Canonical expressway camera name. "
                "Must match one of the supported camera names."
            ),
        },
    },
}


# ============================================================================
# CANONICAL CAMERA NAMES
# ============================================================================

CAMERAS = {
    "PIE": "Pan Island Expressway (PIE)",
    "AYE": "Ayer Rajah Expressway (AYE)",
    "ECP": "East Coast Parkway (ECP)",
    "CTE": "Central Expressway (CTE)",
    "TPE": "Tampines Expressway (TPE)",
    "KPE": "Kallang-Paya Lebar Expressway (KPE)",
    "SLE": "Seletar Expressway (SLE)",
    "BKE": "Bukit Timah Expressway (BKE)",
    "KJE": "Kranji Expressway (KJE)",
    "MCE": "Marina Coastal Expressway (MCE)",
}


# ============================================================================
# CAMERA ALIASES
# ============================================================================
#
# All aliases map to a canonical camera code.
#
# The deterministic resolver should:
#
#     user input
#          ↓
#     normalize text
#          ↓
#     exact alias lookup
#          ↓
#     fuzzy matching if necessary
#          ↓
#     canonical camera code
#
# Example:
#
#     "Kranji Highway"
#             ↓
#            KJE
#             ↓
#     "Kranji Expressway (KJE)"
#

CAMERA_ALIASES = {

    # ------------------------------------------------------------------------
    # PIE
    # ------------------------------------------------------------------------

    "pie": "PIE",
    "p i e": "PIE",
    "pan island expressway": "PIE",
    "pan island expressway pie": "PIE",
    "pan island expressway (pie)": "PIE",

    # ------------------------------------------------------------------------
    # AYE
    # ------------------------------------------------------------------------

    "aye": "AYE",
    "a y e": "AYE",
    "ayer rajah expressway": "AYE",
    "ayer rajah expressway aye": "AYE",
    "ayer rajah expressway (aye)": "AYE",

    # ------------------------------------------------------------------------
    # ECP
    # ------------------------------------------------------------------------

    "ecp": "ECP",
    "e c p": "ECP",
    "east coast parkway": "ECP",
    "east coast parkway ecp": "ECP",
    "east coast parkway (ecp)": "ECP",

    # ------------------------------------------------------------------------
    # CTE
    # ------------------------------------------------------------------------

    "cte": "CTE",
    "c t e": "CTE",
    "central expressway": "CTE",
    "central expressway cte": "CTE",
    "central expressway (cte)": "CTE",

    # ------------------------------------------------------------------------
    # TPE
    # ------------------------------------------------------------------------

    "tpe": "TPE",
    "t p e": "TPE",
    "tampines expressway": "TPE",
    "tampines expressway tpe": "TPE",
    "tampines expressway (tpe)": "TPE",

    # ------------------------------------------------------------------------
    # KPE
    # ------------------------------------------------------------------------

    "kpe": "KPE",
    "k p e": "KPE",
    "kallang paya lebar expressway": "KPE",
    "kallang-paya lebar expressway": "KPE",
    "kallang paya lebar expressway kpe": "KPE",
    "kallang-paya lebar expressway (kpe)": "KPE",

    # ------------------------------------------------------------------------
    # SLE
    # ------------------------------------------------------------------------

    "sle": "SLE",
    "s l e": "SLE",
    "seletar expressway": "SLE",
    "seletar expressway sle": "SLE",
    "seletar expressway (sle)": "SLE",

    # ------------------------------------------------------------------------
    # BKE
    # ------------------------------------------------------------------------

    "bke": "BKE",
    "b k e": "BKE",
    "bukit timah expressway": "BKE",
    "bukit timah expressway bke": "BKE",
    "bukit timah expressway (bke)": "BKE",

    # ------------------------------------------------------------------------
    # KJE
    # ------------------------------------------------------------------------

    "kje": "KJE",
    "k j e": "KJE",
    "kranji expressway": "KJE",
    "kranji expressway kje": "KJE",
    "kranji expressway (kje)": "KJE",

    # Requirement example:
    # "Kranji Highway" should resolve to KJE.
    "kranji highway": "KJE",

    # ------------------------------------------------------------------------
    # MCE
    # ------------------------------------------------------------------------

    "mce": "MCE",
    "m c e": "MCE",
    "marina coastal expressway": "MCE",
    "marina coastal expressway mce": "MCE",
    "marina coastal expressway (mce)": "MCE",
}


# ============================================================================
# COMMON TYPO / FUZZY-MATCH EXAMPLES
# ============================================================================
#
# These are examples rather than an exhaustive typo dictionary.
#
# The actual resolver should use fuzzy matching (e.g. RapidFuzz) after
# exact alias matching fails.
#
# Example:
#
#     "Tampines Expresway"
#              ↓
#     fuzzy match
#              ↓
#     "Tampines Expressway"
#              ↓
#             TPE
#

COMMON_CAMERA_TYPOS = {
    "tampines expresway": "TPE",
    "tampines expresswy": "TPE",
    "central expresway": "CTE",
    "central expresswy": "CTE",
    "kranji expresway": "KJE",
    "pan island expresway": "PIE",
    "ayer rajah expresway": "AYE",
    "seletar expresway": "SLE",
    "bukit timah expresway": "BKE",
    "marina coastal expresway": "MCE",
}


# ============================================================================
# SUPPORTED QUERY CAPABILITIES
# ============================================================================

SUPPORTED_FILTERS = [
    "camera",
    "exact_date",
    "date_range",
    "relative_date",
    "month",
    "time_range",
    "weekday",
    "recurring_weekday",
]


# ============================================================================
# DATE / TIME EXPRESSIONS
# ============================================================================

SUPPORTED_RELATIVE_DATES = [
    "today",
    "yesterday",
    "tomorrow",
    "this week",
    "last week",
    "next week",
    "this month",
    "last month",
    "next month",
]


SUPPORTED_DATE_EXPRESSIONS = [
    # Exact dates
    "2026-08-25",
    "25 August 2026",
    "August 25, 2026",

    # Month
    "August 2026",
    "the whole of August",
    "all of August",

    # Date ranges
    "15th to 18th of August",
    "15 August to 18 August",
    "from 15 August to 18 August",

    # Relative ranges
    "from yesterday to today",
    "last 7 days",
    "past week",
]


SUPPORTED_TIME_EXPRESSIONS = [
    "8 AM",
    "10 PM",
    "08:00",
    "22:00",
    "8:00 AM",
    "8 AM to 10 AM",
    "between 8 AM and 10 AM",
    "from 8 AM to 10 AM",
]


# ============================================================================
# WEEKDAYS / RECURRING CONDITIONS
# ============================================================================

SUPPORTED_WEEKDAYS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


SUPPORTED_RECURRING_PATTERNS = [
    "every Monday",
    "every Tuesday",
    "every Wednesday",
    "every Thursday",
    "every Friday",
    "every Saturday",
    "every Sunday",
]


# ============================================================================
# SUPPORTED INTENTS
# ============================================================================

SUPPORTED_INTENTS = [
    "retrieve_frames",
]


# ============================================================================
# CONVERSATIONAL FOLLOW-UPS
# ============================================================================
#
# The system should retain previously resolved constraints when a
# follow-up only adds or changes part of the query.
#
# Example:
#
# User:
#     "Show me frames from CTE."
#
# Context:
#     camera = CTE
#
# User:
#     "How about this week?"
#
# Result:
#     camera = CTE
#     date = this week
#

CONTEXT_FIELDS = [
    "camera",
    "date",
    "date_range",
    "time_range",
    "weekday",
]


FOLLOW_UP_EXAMPLES = [
    {
        "conversation": [
            "Show me frames from CTE.",
            "How about only those from this week?",
        ],
        "expected": {
            "camera": "CTE",
            "date_expression": "this week",
        },
    },
    {
        "conversation": [
            "Show me frames from PIE yesterday.",
            "Only between 8 AM and 10 AM.",
        ],
        "expected": {
            "camera": "PIE",
            "date_expression": "yesterday",
            "time_range": ["08:00", "10:00"],
        },
    },
]


# ============================================================================
# QUERY RESTRICTIONS / GUARDRAILS
# ============================================================================

READ_ONLY = True

ALLOWED_SQL_OPERATION = "SELECT"

FORBIDDEN_SQL_OPERATIONS = [
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "CREATE",
    "TRUNCATE",
    "REPLACE",
    "UPSERT",
    "ATTACH",
    "DETACH",
]


# ============================================================================
# OUT-OF-SCOPE REQUEST TYPES
# ============================================================================
#
# These are semantic categories. The LLM should identify them, while the
# application performs the final deterministic guardrail check.
#

UNSUPPORTED_REQUEST_TYPES = [
    "unrelated_question",
    "database_modification",
    "arbitrary_sql",
    "unsupported_database_operation",
    "sensitive_information_request",
    "prompt_injection",
]


# ============================================================================
# FEW-SHOT QUERY EXAMPLES
# ============================================================================

QUERY_EXAMPLES = [

    # ------------------------------------------------------------------------
    # Basic camera query
    # ------------------------------------------------------------------------

    {
        "user": "Show me frames from CTE.",
        "intent": "retrieve_frames",
        "camera": "CTE",
    },

    # ------------------------------------------------------------------------
    # Camera + relative date
    # ------------------------------------------------------------------------

    {
        "user": "Show me frames from CTE today.",
        "intent": "retrieve_frames",
        "camera": "CTE",
        "date_expression": "today",
    },

    # ------------------------------------------------------------------------
    # Full camera name
    # ------------------------------------------------------------------------

    {
        "user": "Show me frames from Central Expressway today.",
        "intent": "retrieve_frames",
        "camera": "Central Expressway",
        "date_expression": "today",
    },

    # ------------------------------------------------------------------------
    # Month
    # ------------------------------------------------------------------------

    {
        "user": (
            "Show me frames from Tampines Expressway "
            "for the whole of August."
        ),
        "intent": "retrieve_frames",
        "camera": "TPE",
        "date_expression": "August 2026",
    },

    # ------------------------------------------------------------------------
    # Recurring weekday
    # ------------------------------------------------------------------------

    {
        "user": "Show me frames from MCE on every Tuesday.",
        "intent": "retrieve_frames",
        "camera": "MCE",
        "weekday": "Tuesday",
        "recurring": True,
    },

    # ------------------------------------------------------------------------
    # Relative date range
    # ------------------------------------------------------------------------

    {
        "user": (
            "Show me frames from Kranji Highway "
            "from the 15th to 18th of last month."
        ),
        "intent": "retrieve_frames",
        "camera": "KJE",
        "date_expression": "15th to 18th of last month",
    },

    # ------------------------------------------------------------------------
    # Camera + date + time
    # ------------------------------------------------------------------------

    {
        "user": (
            "Show me PIE frames between "
            "8 AM and 10 AM yesterday."
        ),
        "intent": "retrieve_frames",
        "camera": "PIE",
        "date_expression": "yesterday",
        "time_range": ["08:00", "10:00"],
    },

    # ------------------------------------------------------------------------
    # Typo
    # ------------------------------------------------------------------------

    {
        "user": "Show me frames from Tampines Expresway today.",
        "intent": "retrieve_frames",
        "camera": "Tampines Expresway",
        "date_expression": "today",
    },
]


# ============================================================================
# RESULT CONFIGURATION
# ============================================================================

DEFAULT_RESULT_ORDER = "datetime ASC"

RESULT_COLUMNS = [
    "frame_id",
    "datetime",
    "camera_name",
]


# ============================================================================
# DATABASE QUERY RULES
# ============================================================================

QUERY_RULES = {
    "use_parameterized_queries": True,
    "allow_raw_sql_from_user": False,
    "allow_write_operations": False,
    "allow_multiple_statements": False,
    "allowed_table": "cctv_frames",
}