"""
Benchmark harness: compares LLM providers on the CCTV query agent.

This is a SEPARATE evaluation tool. It does not modify the production
pipeline (src/agent.py, src/llm_parser.py, etc.) — it drives the real,
unmodified pipeline through its public functions and grades the result
at two levels:

    Level 1 (LLM parsing):
        natural language -> QueryFrame, graded against a hand-authored
        "ideal" QueryFrame for that turn. Camera/date/time fields are
        graded through the SAME deterministic resolver used in
        production (resolve_camera / resolve_datetime), so a model is
        never penalized for leaving final date math to the resolver —
        only for extracting the wrong camera/expression/field.

    Level 2 (end-to-end):
        the real QueryAgent.run() pipeline (guardrails -> LLM ->
        context merge -> resolver -> query builder -> SQL guardrail ->
        database), graded against a "gold" ResolvedQuery obtained by
        feeding the same ideal QueryFrame(s) through the exact same
        context-merge and resolver functions the production code uses.

Guardrail-only cases (SQL injection, prompt injection, destructive
requests that a regex catches) never reach the LLM at all — this is
verified by asserting the parse_query spy was not called — so they are
reported separately and are not a point of comparison between models
(both providers necessarily score identically on them, since the LLM
is never invoked).

Usage:
    python benchmark/benchmark_models.py                # full run, both providers
    python benchmark/benchmark_models.py --provider anthropic
    python benchmark/benchmark_models.py --sample 10     # first N cases only (dev)
    python benchmark/benchmark_models.py --categories camera_standard,recurring
    python benchmark/benchmark_models.py --case-ids cam_std_01,exact_date_02   # explicit subset
                                                                                 # (writes *_subset.* results, separate
                                                                                 # from the full run's checkpoint files)

Credentials are read from .env via python-dotenv and are never printed,
logged, or written to any result file.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
import traceback
from pathlib import Path
from unittest.mock import patch

# --- Workaround for an unrelated environment landmine -----------------------
# A personal script literally named `outcome.py` (unrelated to this project)
# sits directly in the Python install root, which is on sys.path by default
# on Windows. That shadows the real `outcome` package (a transitive dependency
# of `trio`/`httpcore2`, which the `anthropic` SDK pulls in) and makes it
# block on input(). Strip that one path entry — for this process only, no
# files are touched — before importing anything that depends on it.
import sysconfig

_INSTALL_ROOT = str(Path(sysconfig.get_paths()["stdlib"]).parent)
sys.path = [p for p in sys.path if p != _INSTALL_ROOT]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import dotenv_values  # noqa: E402

from src import agent as agent_module  # noqa: E402
from src import llm_parser  # noqa: E402
from src.context import ConversationContext  # noqa: E402
from src.query_schema import QueryFrame, build_resolved_query  # noqa: E402
from src.resolver import resolve_camera  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Real API pricing, documented as of the date this benchmark was written
# (see README.md "Model Selection / Evaluation" for sources). Used only to
# turn measured token counts into a cost estimate — never fabricated.
PRICING_USD_PER_1M = {
    "anthropic": {"input": 2.00, "output": 10.00},   # Claude Sonnet 5
    "gemini": {"input": 0.50, "output": 3.00},        # Gemini 3 Flash Preview (text)
}


# ============================================================================
# Provider construction (bypasses the env-var-based factory in llm_parser.py
# so both providers can be benchmarked in one process without relying on
# LLM_PROVIDER / module-level caching).
# ============================================================================


def _load_env() -> dict:
    values = dict(dotenv_values(PROJECT_ROOT / ".env"))
    # A single .env in this repo happens to define LLM_PROVIDER twice
    # (once per provider example) — irrelevant here since we construct
    # providers directly, but ANTHROPIC_API_KEY / GEMINI_API_KEY are each
    # still present exactly once and are read straight from the dict.
    return values


def build_providers(env: dict, only: str | None = None) -> dict:
    providers = {}
    if only in (None, "anthropic"):
        providers["anthropic"] = llm_parser.AnthropicProvider(
            api_key=env["ANTHROPIC_API_KEY"],
            model=env.get("ANTHROPIC_MODEL", "claude-sonnet-5").strip(),
        )
    if only in (None, "gemini"):
        providers["gemini"] = llm_parser.GeminiProvider(
            api_key=env["GEMINI_API_KEY"],
            # NOTE: the code default ("gemini-3-flash") 404s against the
            # live API today; the only deployed model matching that name
            # is the preview build. See README "Design Decisions" for the
            # llm_parser.py default fix that accompanies this benchmark.
            model=env.get("GEMINI_MODEL", "gemini-3-flash-preview").strip() or "gemini-3-flash-preview",
        )
    return providers


# ============================================================================
# Grading helpers
# ============================================================================


def _frame(fields: dict) -> QueryFrame:
    payload = {"intent": "retrieve_frames", **fields}
    return QueryFrame(**payload)


def _camera_signature(camera: str | None) -> str | None:
    if not camera:
        return None
    return resolve_camera(camera)


def _date_signature(frame: QueryFrame):
    from src.resolver import resolve_datetime

    res = resolve_datetime(
        date_expression=frame.date_expression,
        start_date=frame.start_date,
        end_date=frame.end_date,
        start_time=frame.start_time,
        end_time=frame.end_time,
        weekday=frame.weekday,
        recurring=frame.recurring,
    )
    return (
        res.start_datetime,
        res.end_datetime,
        res.time_start,
        res.time_end,
        res.weekday,
        res.recurring,
        res.ambiguous,
    )


def grade_level1_turn(actual_frame: QueryFrame, ideal_fields: dict, level1_fields: list[str]) -> dict:
    """Grade one turn's raw QueryFrame against its ideal, field by field."""

    ideal = _frame(ideal_fields)
    checks = {}

    expected_intent = ideal_fields.get("intent", "retrieve_frames")
    checks["intent"] = actual_frame.intent == expected_intent

    fields_to_check = [f for f in level1_fields if f != "intent"]

    if "camera" in fields_to_check:
        checks["camera"] = _camera_signature(actual_frame.camera) == _camera_signature(ideal.camera)

    date_fields = {"date_expression", "start_date", "end_date"}
    if date_fields & set(fields_to_check):
        checks["date"] = _date_signature(actual_frame)[:2] == _date_signature(ideal)[:2]

    time_fields = {"start_time", "end_time"}
    if time_fields & set(fields_to_check):
        checks["time"] = _date_signature(actual_frame)[2:4] == _date_signature(ideal)[2:4]

    weekday_fields = {"weekday", "recurring"}
    if weekday_fields & set(fields_to_check):
        checks["weekday"] = _date_signature(actual_frame)[4:6] == _date_signature(ideal)[4:6]

    checks["schema_valid"] = True  # only reachable if the provider call didn't raise
    checks["_pass"] = all(checks.values())
    return checks


def compute_gold_resolved(ideal_frames: list[dict]):
    """Replay the ideal per-turn frames through the real context-merge +
    resolver pipeline to get the ground-truth end state of the conversation."""

    ctx = ConversationContext()
    merged = None
    for fields in ideal_frames:
        raw = _frame(fields)
        ctx.add_turn("(gold)")
        merged = ctx.merge(raw)
        resolved = build_resolved_query(merged)
        if resolved.is_valid:
            ctx.remember(merged)
    return build_resolved_query(merged)


def grade_level2_resolved_match(gold, actual_response) -> dict:
    checks = {
        "is_valid": actual_response.error is None,
        "camera": actual_response.camera == gold.camera,
        "start_date": (actual_response.start_datetime or "")[:10] == (gold.start_datetime or "")[:10],
        "end_date": (actual_response.end_datetime or "")[:10] == (gold.end_datetime or "")[:10],
        "time_start": actual_response.time_start == gold.time_start,
        "time_end": actual_response.time_end == gold.time_end,
        "weekday": actual_response.weekday == gold.weekday,
    }
    checks["_pass"] = all(checks.values())
    return checks


# ============================================================================
# Rate-limit handling
#
# The Gemini free tier caps gemini-3-flash-preview at 5 requests/minute.
# Rather than let that show up as spurious "schema_or_api_failures" (which
# would misrepresent the model's actual parsing ability), the harness
# paces requests under that ceiling and retries on 429s using the server's
# own suggested delay when present. Non-rate-limit errors are never
# retried — those are genuine results.
# ============================================================================

_MIN_INTERVAL_SECONDS = {"gemini": 13.0}  # 5 req/min = 12s min spacing; +1s margin
_LAST_CALL_TIME: dict[str, float] = {}
_RETRY_DELAY_RE = re.compile(r"retry in (\d+(?:\.\d+)?)s", re.IGNORECASE)


def _pace(provider_name: str) -> None:
    min_interval = _MIN_INTERVAL_SECONDS.get(provider_name)
    if not min_interval:
        return
    last = _LAST_CALL_TIME.get(provider_name)
    if last is not None:
        elapsed = time.perf_counter() - last
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
    _LAST_CALL_TIME[provider_name] = time.perf_counter()


class DailyQuotaExhaustedError(RuntimeError):
    """
    Raised in place of retrying when a 429 is the provider's hard daily
    request cap (e.g. Gemini free tier's GenerateRequestsPerDay... quota),
    as opposed to a transient per-minute throttle. Retrying this within
    the same day cannot succeed, so call_with_retry raises it immediately
    instead of burning the normal retry/backoff budget.
    """


# The exact quotaId Google returns for the free-tier daily request cap
# (distinct from GenerateRequestsPerMinutePerProjectPerModel-FreeTier,
# which IS worth retrying). Confirmed against a live 429 body during
# investigation of persistent Gemini throttling in this benchmark.
_DAILY_QUOTA_MARKER = "GenerateRequestsPerDayPerProjectPerModel"


def _is_daily_quota_exhausted(exc: Exception) -> bool:
    return _DAILY_QUOTA_MARKER in str(exc)


def _is_rate_limited(exc: Exception) -> bool:
    text = str(exc)
    return "429" in text or "RESOURCE_EXHAUSTED" in text or "rate_limit" in text.lower()


_API_ERROR_MARKERS = (
    "credit balance",
    "insufficient_quota",
    "invalid_api_key",
    "invalid x-api-key",
    "authenticationerror",
    "permission_denied",
    "unauthorized",
    "resource_exhausted",
    "connectionerror",
    "timeout",
    "503",
    "502",
    "500",
    "overloaded",
)


def classify_llm_error(exc: Exception) -> str:
    """
    Distinguish infrastructure/provider-account failures (never reached
    the model, or the provider's own service failed) from genuine model
    output failures (the model responded, but its output didn't parse
    into a valid QueryFrame). Only the latter reflects on model quality.
    """
    if isinstance(exc, DailyQuotaExhaustedError) or _is_daily_quota_exhausted(exc):
        return "daily_quota_exhausted"
    if _is_rate_limited(exc):
        return "api_rate_limit"
    text = str(exc).lower()
    if any(marker in text for marker in _API_ERROR_MARKERS):
        return "api_error"
    return "model_output_error"


def _retry_delay_seconds(exc: Exception, default: float = 20.0) -> float:
    match = _RETRY_DELAY_RE.search(str(exc))
    return float(match.group(1)) + 1.0 if match else default


def call_with_retry(fn, provider_name: str, max_retries: int = 5):
    """Returns (result, latency_ms) where latency_ms times only the final,
    successful call — rate-limit backoff sleep is excluded so it never
    pollutes the reported latency numbers."""
    for attempt in range(max_retries + 1):
        _pace(provider_name)
        start = time.perf_counter()
        try:
            result = fn()
        except Exception as exc:  # noqa: BLE001
            if _is_daily_quota_exhausted(exc):
                # Retrying cannot succeed until the provider's daily quota
                # resets, so don't burn the retry/backoff budget on it —
                # surface it distinctly and let the caller decide to stop.
                raise DailyQuotaExhaustedError(str(exc)) from exc
            if attempt < max_retries and _is_rate_limited(exc):
                delay = _retry_delay_seconds(exc)
                print(f"    (rate limited, retrying in {delay:.0f}s...)", flush=True)
                time.sleep(delay)
                continue
            raise
        return result, (time.perf_counter() - start) * 1000


# ============================================================================
# Running one case against one provider
# ============================================================================


def run_case(case: dict, provider_name: str, provider_instance) -> dict:
    result = {
        "id": case["id"],
        "category": case["category"],
        "provider": provider_name,
        "notes": case.get("notes", ""),
        "turns": [],
        "llm_error": None,
        "llm_error_type": None,
        "latencies_ms": [],
        "usage": [],
    }

    spy_calls = []

    real_parse_query = llm_parser.parse_query

    def spy(user_query, conversation_context=None):
        try:
            frame, elapsed_ms = call_with_retry(
                lambda: real_parse_query(user_query, conversation_context=conversation_context),
                provider_name,
            )
        except Exception as exc:  # noqa: BLE001 — recorded, not raised, so grading can continue
            spy_calls.append({"query": user_query, "frame": None, "error": str(exc), "latency_ms": None})
            raise
        usage = getattr(provider_instance, "last_usage", None)
        spy_calls.append({"query": user_query, "frame": frame, "error": None, "latency_ms": elapsed_ms, "usage": usage})
        if usage:
            result["usage"].append(usage)
        return frame

    agent = agent_module.QueryAgent()

    with patch("src.agent.parse_query", side_effect=spy):
        for turn in case["turns"]:
            try:
                response = agent.run(turn["message"])
            except DailyQuotaExhaustedError:
                # This case did not complete — don't record it as a result
                # (checkpointing it would mark it "done" and skip it on
                # resume, when it should be retried once quota resets).
                # Propagate so main() stops this provider's run gracefully
                # rather than continuing to burn quota on cases that cannot
                # possibly succeed today.
                raise
            except Exception as exc:  # noqa: BLE001
                result["llm_error"] = f"{type(exc).__name__}: {exc}"
                result["llm_error_type"] = classify_llm_error(exc)
                result["turns"].append({"message": turn["message"], "response": None, "level1": None})
                break

            level1 = None
            if "ideal" in turn:
                # The raw frame the LLM produced *this turn* is the last
                # spy call recorded (guardrail-blocked turns never call it).
                if spy_calls and spy_calls[-1]["frame"] is not None:
                    fields = list(turn["ideal"].keys()) if "ideal" in turn else []
                    level1_fields = turn.get("level1_fields", fields or ["intent"])
                    level1 = grade_level1_turn(spy_calls[-1]["frame"], turn["ideal"], level1_fields)

            result["turns"].append(
                {
                    "message": turn["message"],
                    "response": {
                        "reply": response.reply,
                        "error": response.error,
                        "intent": response.intent,
                        "camera": response.camera,
                        "start_datetime": response.start_datetime,
                        "end_datetime": response.end_datetime,
                        "time_start": response.time_start,
                        "time_end": response.time_end,
                        "weekday": response.weekday,
                        "sql": response.sql,
                        "row_count": response.row_count,
                    },
                    "level1": level1,
                }
            )

    result["llm_calls"] = len(spy_calls)
    result["latencies_ms"] = [c["latency_ms"] for c in spy_calls if c["frame"] is not None]

    # --- Level 2 grading ---
    final_mode = case["final_level2"]
    last_response = result["turns"][-1]["response"] if result["turns"] else None

    if final_mode == "blocked":
        result["level2"] = {
            "_pass": (
                len(spy_calls) == 0
                and last_response is not None
                and last_response["error"] is not None
                and last_response["sql"] is None
            ),
            "llm_invoked": len(spy_calls) > 0,
        }
    elif final_mode == "is_valid_false":
        result["level2"] = {
            "_pass": last_response is not None and last_response["error"] is not None,
        }
    elif final_mode == "lenient_unbounded_or_clarify":
        ok = last_response is not None and (
            (last_response["error"] is None and last_response["camera"] is None and last_response["start_datetime"] is None)
            or (last_response["intent"] == "clarification_needed")
        )
        result["level2"] = {"_pass": ok}
    elif final_mode == "resolved_match":
        ideal_frames = [t["ideal"] for t in case["turns"]]
        gold = compute_gold_resolved(ideal_frames)
        if last_response is None:
            result["level2"] = {"_pass": False, "reason": "no response (exception)"}
        else:

            class _R:
                pass

            r = _R()
            for k in ("error", "camera", "start_datetime", "end_datetime", "time_start", "time_end", "weekday"):
                setattr(r, k, last_response[k])
            result["level2"] = grade_level2_resolved_match(gold, r)
    else:
        raise ValueError(f"Unknown final_level2 mode: {final_mode}")

    return result


# ============================================================================
# Aggregate metrics
# ============================================================================


def aggregate(raw_results: list[dict], provider_name: str) -> dict:
    def rate(preds):
        preds = list(preds)
        return (sum(preds) / len(preds)) if preds else None

    level1_all_checks = []
    for r in raw_results:
        for t in r["turns"]:
            if t["level1"]:
                level1_all_checks.append(t["level1"])

    def field_rate(field):
        vals = [c[field] for c in level1_all_checks if field in c]
        return rate(vals)

    level2_pass = [r["level2"]["_pass"] for r in raw_results]
    by_category = {}
    for r in raw_results:
        by_category.setdefault(r["category"], []).append(r["level2"]["_pass"])
    category_rates = {cat: rate(v) for cat, v in by_category.items()}

    all_latencies = [ms for r in raw_results for ms in r["latencies_ms"]]
    schema_failures = sum(1 for r in raw_results if r["llm_error"])
    api_errors = sum(1 for r in raw_results if r.get("llm_error_type") in ("api_error", "api_rate_limit"))
    model_output_errors = sum(1 for r in raw_results if r.get("llm_error_type") == "model_output_error")
    failed_case_ids = {
        error_type: [r["id"] for r in raw_results if r.get("llm_error_type") == error_type]
        for error_type in ("api_error", "api_rate_limit", "model_output_error")
    }

    all_usage = [u for r in raw_results for u in r.get("usage", [])]
    total_input_tokens = sum(u["input_tokens"] for u in all_usage)
    total_output_tokens = sum(u["output_tokens"] for u in all_usage)
    pricing = PRICING_USD_PER_1M.get(provider_name)
    cost_usd = None
    if pricing and all_usage:
        cost_usd = (total_input_tokens / 1_000_000) * pricing["input"] + (
            total_output_tokens / 1_000_000
        ) * pricing["output"]

    return {
        "provider": provider_name,
        "num_cases": len(raw_results),
        "num_llm_calls": sum(r["llm_calls"] for r in raw_results),
        "schema_or_api_failures": schema_failures,
        "failures_by_type": {
            "api_error_or_rate_limit": api_errors,
            "model_output_error": model_output_errors,
        },
        "failed_case_ids": failed_case_ids,
        "level1_field_accuracy": {
            "intent": field_rate("intent"),
            "camera": field_rate("camera"),
            "date": field_rate("date"),
            "time": field_rate("time"),
            "weekday_recurring": field_rate("weekday"),
            "overall": rate([c["_pass"] for c in level1_all_checks]),
        },
        "level2_end_to_end_accuracy": {
            "overall": rate(level2_pass),
            "by_category": category_rates,
        },
        "latency_ms": {
            "count": len(all_latencies),
            "mean": (sum(all_latencies) / len(all_latencies)) if all_latencies else None,
            "min": min(all_latencies) if all_latencies else None,
            "max": max(all_latencies) if all_latencies else None,
            "p50": sorted(all_latencies)[len(all_latencies) // 2] if all_latencies else None,
        },
        "token_usage": {
            "calls_with_usage": len(all_usage),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
        },
        "cost_usd": cost_usd,
        "pricing_used_usd_per_1m": pricing,
    }


# ============================================================================
# Checkpointing
#
# One JSON Lines file per provider (benchmark/results/checkpoint_<provider>.jsonl),
# appended after every completed case and flushed immediately. Append-only
# is deliberate: a hard kill mid-write can only truncate the last line
# (skipped on reload), never corrupt previously-completed cases, unlike
# rewriting a single JSON file on every case would risk.
# ============================================================================


def _checkpoint_path(provider_name: str, run_tag: str | None = None) -> Path:
    suffix = f"_{run_tag}" if run_tag else ""
    return RESULTS_DIR / f"checkpoint_{provider_name}{suffix}.jsonl"


def _load_checkpoint(provider_name: str, run_tag: str | None = None) -> dict:
    """Returns {case_id: result} for cases already completed in a prior,
    interrupted run of this provider. A truncated trailing line (e.g. from
    a hard kill mid-write) is skipped rather than failing the whole load."""
    path = _checkpoint_path(provider_name, run_tag)
    completed = {}
    if not path.exists():
        return completed
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                continue
            completed[result["id"]] = result
    return completed


def _append_checkpoint(provider_name: str, result: dict, run_tag: str | None = None) -> None:
    with open(_checkpoint_path(provider_name, run_tag), "a", encoding="utf-8") as f:
        f.write(json.dumps(result, default=str))
        f.write("\n")
        f.flush()


def _tagged_path(stem: str, run_tag: str | None, ext: str) -> Path:
    suffix = f"_{run_tag}" if run_tag else ""
    return RESULTS_DIR / f"{stem}{suffix}.{ext}"


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Benchmark LLM providers on the CCTV query agent.")
    parser.add_argument("--provider", choices=["anthropic", "gemini"], default=None, help="Run only one provider (default: both).")
    parser.add_argument("--sample", type=int, default=None, help="Only run the first N test cases (for cheap dev iteration).")
    parser.add_argument("--categories", type=str, default=None, help="Comma-separated list of categories to run.")
    parser.add_argument(
        "--case-ids",
        type=str,
        default=None,
        help=(
            "Comma-separated list of exact case IDs to run (e.g. "
            "'cam_std_01,exact_date_02'), instead of the full 60-case benchmark. "
            "Unless --run-tag is also given, results/checkpoints for this run are "
            "written to *_subset.* files so they never mix with the full run's "
            "checkpoint_<provider>.jsonl / raw_<provider>.json / aggregate_<provider>.json."
        ),
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Suffix for checkpoint/raw/aggregate/comparison filenames (e.g. 'subset20'). "
        "Defaults to 'subset' automatically when --case-ids is used; has no effect otherwise.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore any existing checkpoint_<provider>[_<tag>].jsonl and start that provider's run from scratch.",
    )
    args = parser.parse_args()

    with open(Path(__file__).resolve().parent / "test_cases.json", encoding="utf-8") as f:
        cases = json.load(f)["cases"]

    if args.categories:
        wanted = set(c.strip() for c in args.categories.split(","))
        cases = [c for c in cases if c["category"] in wanted]

    if args.case_ids:
        wanted_ids = [c.strip() for c in args.case_ids.split(",") if c.strip()]
        cases_by_id = {c["id"]: c for c in cases}
        missing = [cid for cid in wanted_ids if cid not in cases_by_id]
        if missing:
            parser.error(f"Unknown case id(s): {', '.join(missing)}")
        cases = [cases_by_id[cid] for cid in wanted_ids]

    if args.sample:
        cases = cases[: args.sample]

    run_tag = args.run_tag or ("subset" if args.case_ids else None)

    env = _load_env()
    providers = build_providers(env, only=args.provider)

    RESULTS_DIR.mkdir(exist_ok=True)

    all_aggregates = {}
    for provider_name, provider_instance in providers.items():
        print(f"\n=== Running {len(cases)} cases against provider={provider_name} ===", flush=True)
        llm_parser._provider = provider_instance

        completed = {} if args.fresh else _load_checkpoint(provider_name, run_tag)
        if completed:
            print(
                f"  Resuming from checkpoint: {len(completed)} case(s) already "
                f"completed, skipping.",
                flush=True,
            )
        raw_results = list(completed.values())

        stopped_early = False
        for i, case in enumerate(cases, 1):
            if case["id"] in completed:
                print(f"  [{i}/{len(cases)}] {case['id']} ({case['category']}) - skipped (checkpointed)", flush=True)
                continue

            print(f"  [{i}/{len(cases)}] {case['id']} ({case['category']})", flush=True)
            try:
                result = run_case(case, provider_name, provider_instance)
            except DailyQuotaExhaustedError as exc:
                print(
                    f"\n  Daily free-tier quota exhausted at case {i}/{len(cases)} "
                    f"({case['id']}): {exc}\n"
                    f"  Stopping the {provider_name} run gracefully. "
                    f"{len(raw_results)} case(s) already completed are saved in "
                    f"{_checkpoint_path(provider_name).name} - re-run the same "
                    f"command after the quota resets to continue.",
                    flush=True,
                )
                stopped_early = True
                break
            except Exception:  # noqa: BLE001
                print(f"    !! unhandled error on {case['id']}:")
                traceback.print_exc()
                result = {
                    "id": case["id"],
                    "category": case["category"],
                    "provider": provider_name,
                    "turns": [],
                    "llm_error": "harness_error",
                    "latencies_ms": [],
                    "llm_calls": 0,
                    "level2": {"_pass": False},
                }

            raw_results.append(result)
            _append_checkpoint(provider_name, result, run_tag)

        raw_path = _tagged_path(f"raw_{provider_name}", run_tag, "json")
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(raw_results, f, indent=2, default=str)

        agg = aggregate(raw_results, provider_name)
        agg["stopped_early"] = stopped_early
        agg["cases_completed"] = len(raw_results)
        agg["cases_total"] = len(cases)
        all_aggregates[provider_name] = agg
        agg_path = _tagged_path(f"aggregate_{provider_name}", run_tag, "json")
        with open(agg_path, "w", encoding="utf-8") as f:
            json.dump(agg, f, indent=2)

        suffix = " (STOPPED EARLY - daily quota exhausted)" if stopped_early else ""
        print(f"\n--- {provider_name} summary{suffix} ---")
        print(json.dumps(agg, indent=2))

    comparison_path = _tagged_path("comparison", run_tag, "json")
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(all_aggregates, f, indent=2)

    print(f"\nRaw results, aggregates, and comparison written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
