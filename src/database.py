"""
Read-only database access for the CCTV query agent.

This module owns SQLite connection handling and query execution only.
It has no knowledge of natural language, camera aliases, or date
resolution — it only ever runs the parameterized SQL that
query_builder.py hands it, after guardrails.py has already screened it.

The connection is opened in SQLite's read-only URI mode as a second,
connection-level backstop: even if every check upstream were somehow
bypassed, the database driver itself would still refuse to execute a
write.
"""

import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "cctv.db"


class DatabaseError(Exception):
    """Raised when a query cannot be executed against the database."""


def get_connection(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open a read-only connection to the CCTV database."""

    db_path = Path(db_path)

    if not db_path.exists():
        raise DatabaseError(
            f"Database not found at {db_path}. "
            "Run `python scripts/generate_data.py` first."
        )

    # mode=ro: SQLite itself refuses writes at the OS/driver level, even
    # if every check upstream (guardrails, query_builder) were somehow
    # bypassed. This is a backstop, not the primary line of defense.
    uri = f"file:{db_path.as_posix()}?mode=ro"

    try:
        connection = sqlite3.connect(uri, uri=True)
    except sqlite3.OperationalError as exc:
        raise DatabaseError(f"Could not open database: {exc}") from exc

    connection.execute("PRAGMA query_only = ON;")
    connection.row_factory = sqlite3.Row
    return connection


def execute_query(
    sql: str,
    params: list | tuple = (),
    db_path: Path | str = DEFAULT_DB_PATH,
) -> list[dict]:
    """
    Execute a parameterized, read-only SELECT and return rows as dicts.

    Args:
        sql:     SQL text containing only "?" placeholders. Expected to
                 already have passed guardrails.check_sql().
        params:  Ordered values to bind to the placeholders.
        db_path: Path to the SQLite database file.

    Returns:
        A list of row dicts, keyed by column name (e.g. "frame_id",
        "datetime", "camera_name").

    Raises:
        DatabaseError: on any failure to open the database or execute
        the query (including an attempted write, which `PRAGMA
        query_only` will reject).
    """

    connection = get_connection(db_path)
    try:
        cursor = connection.execute(sql, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    except sqlite3.Error as exc:
        raise DatabaseError(f"Query execution failed: {exc}") from exc
    finally:
        connection.close()


# ============================================================================
# Simple local test
# ============================================================================

if __name__ == "__main__":
    sql = (
        "SELECT frame_id, datetime, camera_name\n"
        "FROM cctv_frames\n"
        "WHERE camera_name = ?\n"
        "ORDER BY datetime ASC\n"
        "LIMIT ?"
    )
    rows = execute_query(sql, ["Central Expressway (CTE)", 5])
    for row in rows:
        print(row)