"""
Shared pytest configuration for the CCTV query agent test suite.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# src/llm_parser.py requires OPENAI_API_KEY to be set at import time.
# The tests never make a real API call — llm_parser.parse_query is
# always mocked — but importing src.agent (which imports src.llm_parser)
# must not blow up just because a real key isn't configured in the test
# environment (e.g. CI).
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used-for-real-calls")