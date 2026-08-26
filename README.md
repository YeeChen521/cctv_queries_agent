# cctv_queries_agent
Take home assignment for cynapse.ai

Recommended project structure
cctv-query-agent/
│
├── README.md
├── requirements.txt
├── .env.example
│
├── data/
│   └── cctv.db
│
├── src/
│   ├── main.py
│   │
│   ├── llm_parser.py
│   ├── query_schema.py
│   ├── context.py
│   ├── resolver.py
│   ├── query_builder.py
│   ├── database.py
│   └── guardrails.py
│
├── config/
│   └── metadata.py
│
├── scripts/
│   └── generate_data.py
│
└── tests/
    ├── test_queries.py
    ├── test_followups.py
    └── test_edge_cases.py

That's it. No LangChain, no LangGraph, no vector DB, no Redis initially.

What each file does
main.py

The entry point.

If you're making a simple CLI:

User
 ↓
main.py
 ↓
parser
 ↓
resolver
 ↓
query builder
 ↓
database
 ↓
response

If you want an API, this can contain your FastAPI endpoints.

llm_parser.py

Only responsible for:

Natural language → structured QueryFrame

For example:

"Show me PIE frames yesterday"

becomes:

{
  "intent": "retrieve_frames",
  "camera": "PIE",
  "date_expression": "yesterday",
  "time_start": null,
  "time_end": null,
  "weekdays": null
}

No SQL here.

query_schema.py

Defines the structure of the LLM output.

For example, using Pydantic:

class QueryFrame(BaseModel):
    intent: Literal["retrieve_frames", "unsupported"]
    camera: str | None = None
    date_expression: str | None = None
    time_start: str | None = None
    time_end: str | None = None
    weekdays: list[int] | None = None

This is important because the LLM is forced to produce a predictable structure.

context.py

Handles conversational follow-ups.

Example:

User:
Show me frames from CTE.

      ↓

context:
camera = CTE

User:
How about this week?

      ↓

context:
camera = CTE
date = this week

You can simply use an in-memory Python object initially.

No Redis needed.

resolver.py

This is your main deterministic logic.

It handles:

Camera

CTE
Central Expressway
central expressway
Kranji Highway
Tampines Expresway

→ canonical camera.

And:

Date/time

today
yesterday
this week
last month
15th–18th of last month
8 AM–10 AM yesterday
every Tuesday

→ actual datetime constraints.

This file will probably be one of the most important parts of your implementation.

query_builder.py

This converts your resolved QueryFrame into SQL.

For example:

camera = CTE
start = 2026-08-25 08:00
end = 2026-08-25 10:00

becomes:

SELECT frame_id, datetime, camera_name
FROM cctv_frames
WHERE camera_name = ?
  AND datetime >= ?
  AND datetime < ?
ORDER BY datetime

with parameters:

[
    "Central Expressway",
    start_datetime,
    end_datetime
]

This is where you ensure parameterized SQL.

database.py

Only database operations.

Something like:

def execute_query(sql, params):
    ...

and perhaps:

def get_frames(query):
    ...

Don't put natural-language logic here.

guardrails.py

Handles things like:

"Delete all frames"
"DROP TABLE"
"Show me passwords"
"What is today's weather?"
"Ignore previous instructions..."

It determines whether the request is within scope.

Also enforce that the generated operation is read-only.

config/metadata.py

This is the part we added based on your RAG idea.

But instead of RAG, keep a tiny metadata registry.

For example:

CAMERAS = {
    "PIE": "Pan Island Expressway",
    "AYE": "Ayer Rajah Expressway",
    "ECP": "East Coast Parkway",
    "CTE": "Central Expressway",
    "TPE": "Tampines Expressway",
    "KPE": "Kallang-Paya Lebar Expressway",
    "SLE": "Seletar Expressway",
    "BKE": "Bukit Timah Expressway",
    "KJE": "Kranji Expressway",
    "MCE": "Marina Coastal Expressway",
}

You can also put a small schema description here:

DATABASE_SCHEMA = {
    "table": "cctv_frames",
    "columns": {
        "frame_id": "Unique frame identifier",
        "datetime": "Frame capture timestamp",
        "camera_name": "Expressway camera name"
    }
}

And perhaps supported capabilities:

SUPPORTED_FILTERS = [
    "camera",
    "date",
    "time",
    "date_range",
    "weekday"
]

Then llm_parser.py can use this metadata when constructing its prompt.

This gives you the grounding benefit without building a RAG pipeline.

scripts/generate_data.py

Generate the synthetic dataset.

You have:

10 cameras
×
288 frames/day
×
365 days
≈ 1.05 million rows

I'd generate the full dataset because it makes your scalability discussion more credible.

Tests

You don't need a huge test suite.

test_queries.py

Normal cases:

CTE today
PIE yesterday
MCE August
PIE 8–10 AM yesterday
test_followups.py

Conversation:

"Show me CTE."
"How about this week?"

and:

"Show me PIE yesterday."
"Only between 8 and 10."
test_edge_cases.py

Things like:

Kranji Highway
Tampines Expresway
every Tuesday
unsupported request
SQL injection
prompt injection
ambiguous camera

This is enough to demonstrate robustness.

Final architecture

So your actual implementation is basically:

                    USER
                      │
                      ▼
               ┌─────────────┐
               │  main.py    │
               └──────┬──────┘
                      │
                      ▼
               ┌─────────────┐
               │  context.py │
               └──────┬──────┘
                      │
                      ▼
             ┌─────────────────┐
             │  llm_parser.py  │◄──── config/metadata.py
             │                 │
             │ NL → QueryFrame │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │  guardrails.py  │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │   resolver.py   │
             │                 │
             │ Camera + Date   │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │query_builder.py │
             │                 │
             │ QueryFrame → SQL │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │   database.py   │
             └────────┬────────┘
                      │
                      ▼
                  cctv.db