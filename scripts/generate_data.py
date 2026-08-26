"""
Generate the synthetic CCTV frame database.

Dataset requirements:
- 10 specified expressway cameras
- One frame every 5 minutes
- Throughout 2026, up to the current date/time
- Each record contains:
    - Frame ID [int]
    - Datetime [ISO 8601]
    - Camera name [string enum]

Usage:
    python scripts/generate_data.py

The script creates:
    data/cctv.db
"""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CAMERAS = [
    "Pan Island Expressway (PIE)",
    "Ayer Rajah Expressway (AYE)",
    "East Coast Parkway (ECP)",
    "Central Expressway (CTE)",
    "Tampines Expressway (TPE)",
    "Kallang-Paya Lebar Expressway (KPE)",
    "Seletar Expressway (SLE)",
    "Bukit Timah Expressway (BKE)",
    "Kranji Expressway (KJE)",
    "Marina Coastal Expressway (MCE)",
]

# The assignment says data exists throughout 2026
# up to the current date/time.
CURRENT_DATETIME = datetime.now()

START_DATETIME = datetime(2026, 1, 1, 0, 0, 0)

# CCTV captures occur every 5 minutes.
# Round the current time down to the latest available 5-minute timestamp.
minutes = (CURRENT_DATETIME.minute // 5) * 5

END_DATETIME = CURRENT_DATETIME.replace(
    minute=minutes,
    second=0,
    microsecond=0,
)

FRAME_INTERVAL = timedelta(minutes=5)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATABASE_PATH = DATA_DIR / "cctv.db"

BATCH_SIZE = 10_000


# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

def create_database(connection: sqlite3.Connection) -> None:
    """Create the CCTV table and indexes."""

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE cctv_frames (
            frame_id INTEGER PRIMARY KEY,
            datetime TEXT NOT NULL,
            camera_name TEXT NOT NULL
        )
    """)

    # Most queries will filter by camera and datetime.
    cursor.execute("""
        CREATE INDEX idx_cctv_camera_datetime
        ON cctv_frames(camera_name, datetime)
    """)

    connection.commit()


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_records():
    """
    Yield CCTV frame records.

    For every 5-minute timestamp, one frame is generated
    for each expressway camera.
    """

    frame_id = 1
    current_datetime = START_DATETIME

    while current_datetime <= END_DATETIME:

        # ISO 8601 timestamp with Singapore UTC+08:00 offset.
        timestamp = current_datetime.strftime(
            "%Y-%m-%dT%H:%M:%S+08:00"
        )

        for camera_name in CAMERAS:
            yield (
                frame_id,
                timestamp,
                camera_name,
            )

            frame_id += 1

        current_datetime += FRAME_INTERVAL


def populate_database(connection: sqlite3.Connection) -> int:
    """Insert generated records in batches."""

    cursor = connection.cursor()

    batch = []
    total_rows = 0

    for record in generate_records():

        batch.append(record)

        if len(batch) >= BATCH_SIZE:

            cursor.executemany(
                """
                INSERT INTO cctv_frames (
                    frame_id,
                    datetime,
                    camera_name
                )
                VALUES (?, ?, ?)
                """,
                batch,
            )

            total_rows += len(batch)
            batch.clear()

    # Insert remaining records.
    if batch:

        cursor.executemany(
            """
            INSERT INTO cctv_frames (
                frame_id,
                datetime,
                camera_name
            )
            VALUES (?, ?, ?)
            """,
            batch,
        )

        total_rows += len(batch)

    connection.commit()

    return total_rows


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_database(connection: sqlite3.Connection) -> None:
    """Run basic checks to ensure the generated dataset is correct."""

    cursor = connection.cursor()

    # Total rows
    row_count = cursor.execute(
        "SELECT COUNT(*) FROM cctv_frames"
    ).fetchone()[0]

    # Number of cameras
    camera_count = cursor.execute(
        "SELECT COUNT(DISTINCT camera_name) FROM cctv_frames"
    ).fetchone()[0]

    # Date range
    min_datetime, max_datetime = cursor.execute(
        """
        SELECT MIN(datetime), MAX(datetime)
        FROM cctv_frames
        """
    ).fetchone()

    print("\nValidation")
    print("-" * 60)

    print(f"Total rows        : {row_count:,}")
    print(f"Number of cameras : {camera_count}")
    print(f"First timestamp   : {min_datetime}")
    print(f"Last timestamp    : {max_datetime}")

    # Rows per camera
    print("\nRows per camera")
    print("-" * 60)

    rows_per_camera = cursor.execute(
        """
        SELECT camera_name, COUNT(*)
        FROM cctv_frames
        GROUP BY camera_name
        ORDER BY camera_name
        """
    ).fetchall()

    for camera_name, count in rows_per_camera:
        print(f"{camera_name:<40} {count:,}")

    # -----------------------------------------------------------------------
    # Validation checks
    # -----------------------------------------------------------------------

    expected_camera_count = len(CAMERAS)

    if camera_count != expected_camera_count:
        raise RuntimeError(
            f"Expected {expected_camera_count} cameras, "
            f"but found {camera_count}."
        )

    # Calculate expected number of timestamps.
    expected_timestamps = (
        int((END_DATETIME - START_DATETIME) / FRAME_INTERVAL) + 1
    )

    expected_rows = (
        expected_timestamps * expected_camera_count
    )

    if row_count != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows:,} rows, "
            f"but generated {row_count:,}."
        )

    print("\nValidation passed.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Create and populate the CCTV database."""

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Remove an existing database so that every run
    # starts with a clean dataset.
    if DATABASE_PATH.exists():
        DATABASE_PATH.unlink()

    print("Generating CCTV database...")
    print("-" * 60)

    print(f"Database      : {DATABASE_PATH}")
    print(f"Start         : {START_DATETIME.isoformat()}")
    print(f"End           : {END_DATETIME.isoformat()}")
    print(f"Cameras       : {len(CAMERAS)}")
    print(f"Frame interval: {FRAME_INTERVAL}")

    connection = sqlite3.connect(DATABASE_PATH)

    try:

        # Create table and indexes.
        create_database(connection)

        # Generate and insert records.
        total_rows = populate_database(connection)

        print(f"\nInserted {total_rows:,} records.")

        # Validate generated dataset.
        validate_database(connection)

    finally:

        connection.close()

    print("\nDatabase created successfully!")
    print(f"Location: {DATABASE_PATH}")


if __name__ == "__main__":
    main()