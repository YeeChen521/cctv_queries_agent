from rapidfuzz import process, fuzz

from config.metadata import CAMERAS, CAMERA_ALIASES


def resolve_camera(user_input: str) -> str | None:
    """Resolve a user-provided camera name to its canonical name."""

    normalized = user_input.lower().strip()

    # 1. Exact alias match
    if normalized in CAMERA_ALIASES:
        camera_code = CAMERA_ALIASES[normalized]
        return CAMERAS[camera_code]

    # 2. Fuzzy match
    match = process.extractOne(
        normalized,
        CAMERA_ALIASES.keys(),
        scorer=fuzz.ratio,
    )

    if match is None:
        return None

    matched_alias, score, _ = match

    # Require sufficiently high similarity.
    if score >= 85:
        camera_code = CAMERA_ALIASES[matched_alias]
        return CAMERAS[camera_code]

    return None