#!/usr/bin/env python3
"""Run the preregistered synthetic provider screen for G2b."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Callable, Literal, Sequence

import litellm
from pydantic import BaseModel, ConfigDict, ValidationError


SCREEN_VERSION = "g2b-provider-screen-v1"
SCREEN_SEED = 20260827
TEMPERATURE = 0.0
TIMEOUT_SECONDS = 90.0
MAX_COMPLETION_TOKENS = 128
MIN_START_INTERVAL_SECONDS = 12.0
RETRY_DELAYS_SECONDS = (20.0, 40.0)
MAX_TOTAL_TRANSIENT_RETRIES_FOR_PASS = 2
MAX_INPUT_TOKENS_PER_CANDIDATE = 1_200_000
MAX_ESTIMATED_COST_PER_CANDIDATE_USD = 2.00


@dataclass(frozen=True)
class Candidate:
    rank: int
    name: str
    model: str
    reasoning_effort: str | None
    input_usd_per_million: float
    cached_input_usd_per_million: float
    output_usd_per_million: float


CANDIDATES = (
    Candidate(
        rank=1,
        name="primary",
        model="openai/gpt-5.4-mini-2026-03-17",
        reasoning_effort="none",
        input_usd_per_million=0.75,
        cached_input_usd_per_million=0.075,
        output_usd_per_million=4.50,
    ),
    Candidate(
        rank=2,
        name="compatibility_fallback",
        model="openai/gpt-4.1-mini-2025-04-14",
        reasoning_effort=None,
        input_usd_per_million=0.40,
        cached_input_usd_per_million=0.10,
        output_usd_per_million=1.60,
    ),
)


@dataclass(frozen=True)
class SessionSpec:
    name: str
    calls: int
    characters_per_call: int


SESSIONS = (
    SessionSpec("short", 6, 2_500),
    SessionSpec("medium", 8, 6_000),
    SessionSpec("long", 10, 19_000),
)
EXPECTED_CALLS = sum(session.calls for session in SESSIONS)


class ScreenAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["ANSWER"]
    content: str


class ScreenFailure(RuntimeError):
    """A frozen feasibility criterion failed."""

    def __init__(self, message: str, *, category: str):
        super().__init__(message)
        self.category = category


WORDS = (
    "amber bridge calm delta ember field gentle harbor ivory juniper kindle "
    "lumen meadow north orbit pebble quiet river silver timber umber valley "
    "willow xenon yellow zephyr"
).split()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def synthetic_block(session: str, turn: int, characters: int) -> str:
    """Return deterministic generic text at exactly ``characters`` characters."""
    prefix = f"SYNTHETIC TRANSPORT PAYLOAD session={session} turn={turn}. "
    pieces = [prefix]
    total = len(prefix)
    index = 0
    while total < characters:
        word = WORDS[(turn * 7 + index * 11) % len(WORDS)]
        piece = f"{word}-{turn:02d}-{index:05d} "
        pieces.append(piece)
        total += len(piece)
        index += 1
    return "".join(pieces)[:characters]


def expected_nonce(candidate: Candidate, session: SessionSpec, turn: int) -> str:
    return f"g2b-screen:{candidate.rank}:{session.name}:{turn:02d}"


def build_user_message(
    candidate: Candidate, session: SessionSpec, turn: int
) -> str:
    nonce = expected_nonce(candidate, session, turn)
    payload = synthetic_block(session.name, turn, session.characters_per_call)
    return (
        "This is a synthetic API transport check, not a benchmark question.\n"
        "Return the structured action ANSWER with content exactly equal to "
        f"{nonce!r}. Do not solve, summarize, or transform the payload.\n\n"
        f"{payload}"
    )


def _status_code(exc: BaseException) -> int | None:
    for attr in ("status_code", "http_status", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and 100 <= value <= 599:
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None) if response is not None else None
    if isinstance(value, int) and 100 <= value <= 599:
        return value
    match = re.search(
        r"(?:status[_ ]?code|http(?:\s+error)?|error\s+code)[\s:=]+(\d{3})",
        str(exc),
        re.IGNORECASE,
    )
    return int(match.group(1)) if match else None


def is_retryable(exc: BaseException) -> bool:
    status = _status_code(exc)
    if status is not None:
        return status in {408, 425, 429} or 500 <= status <= 599
    return isinstance(
        exc,
        (
            litellm.InternalServerError,
            litellm.APIConnectionError,
            litellm.Timeout,
            litellm.ServiceUnavailableError,
            litellm.RateLimitError,
            ConnectionError,
            TimeoutError,
            OSError,
        ),
    )


def is_account_failure(exc: BaseException) -> bool:
    status = _status_code(exc)
    text = str(exc).lower()
    if status in {401, 402, 403}:
        return True
    return any(
        marker in text
        for marker in (
            "api key",
            "authentication",
            "billing",
            "insufficient_quota",
            "organization",
            "usage limit",
        )
    )


def _extract_action(response: Any) -> ScreenAction:
    message = response.choices[0].message
    parsed = getattr(message, "parsed", None)
    if isinstance(parsed, ScreenAction):
        return parsed
    if isinstance(parsed, BaseModel):
        return ScreenAction.model_validate(parsed.model_dump())
    if isinstance(parsed, dict):
        return ScreenAction.model_validate(parsed)

    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        arguments = tool_calls[0].function.arguments
        if isinstance(arguments, str):
            return ScreenAction.model_validate_json(arguments)
        return ScreenAction.model_validate(arguments)

    content = getattr(message, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise ScreenFailure("provider returned empty content", category="model_failure")
    return ScreenAction.model_validate_json(content)


def _usage_payload(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")

    def get_value(name: str, fallback: str | None = None) -> Any:
        if isinstance(usage, dict):
            return usage.get(name, usage.get(fallback) if fallback else None)
        value = getattr(usage, name, None)
        return value if value is not None else getattr(usage, fallback, None)

    input_tokens = get_value("prompt_tokens", "input_tokens")
    output_tokens = get_value("completion_tokens", "output_tokens")
    if (
        isinstance(input_tokens, bool)
        or not isinstance(input_tokens, int)
        or input_tokens < 0
    ):
        raise ScreenFailure(
            "response omitted a nonnegative input-token count",
            category="model_failure",
        )
    if (
        isinstance(output_tokens, bool)
        or not isinstance(output_tokens, int)
        or output_tokens < 0
    ):
        output_tokens = 0

    details = get_value("prompt_tokens_details", "input_tokens_details")
    if isinstance(details, dict):
        cached_tokens = details.get("cached_tokens", 0)
    else:
        cached_tokens = getattr(details, "cached_tokens", 0)
    if not isinstance(cached_tokens, int) or isinstance(cached_tokens, bool):
        cached_tokens = 0
    cached_tokens = max(0, min(cached_tokens, input_tokens))
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_tokens,
        "output_tokens": output_tokens,
    }


def _response_metadata(response: Any) -> dict[str, str | None]:
    response_id = getattr(response, "id", None)
    fingerprint = getattr(response, "system_fingerprint", None)
    return {
        "response_id": str(response_id) if response_id else None,
        "system_fingerprint": str(fingerprint) if fingerprint else None,
    }


def completion_kwargs(
    candidate: Candidate, messages: list[dict[str, str]]
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": candidate.model,
        "messages": messages,
        "response_format": ScreenAction,
        "temperature": TEMPERATURE,
        "seed": SCREEN_SEED,
        "timeout": TIMEOUT_SECONDS,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "max_retries": 0,
    }
    if candidate.reasoning_effort is not None:
        kwargs["reasoning_effort"] = candidate.reasoning_effort
    return kwargs


def _pace(last_start: float | None, monotonic: Callable[[], float]) -> float:
    if last_start is None:
        return 0.0
    return max(0.0, MIN_START_INTERVAL_SECONDS - (monotonic() - last_start))


def _estimated_cost(candidate: Candidate, totals: dict[str, int]) -> float:
    uncached = totals["input_tokens"] - totals["cached_input_tokens"]
    return (
        uncached * candidate.input_usd_per_million
        + totals["cached_input_tokens"] * candidate.cached_input_usd_per_million
        + totals["output_tokens"] * candidate.output_usd_per_million
    ) / 1_000_000


def run_candidate(
    candidate: Candidate,
    *,
    completion: Callable[..., Any] = litellm.completion,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    total_retries = 0
    last_start: float | None = None
    started_at = _utc_now()
    failure: dict[str, str] | None = None

    try:
        for session in SESSIONS:
            messages: list[dict[str, str]] = []
            for turn in range(1, session.calls + 1):
                prompt = build_user_message(candidate, session, turn)
                messages.append({"role": "user", "content": prompt})
                expected = expected_nonce(candidate, session, turn)
                retries = 0
                while True:
                    delay = _pace(last_start, monotonic)
                    if delay:
                        sleep(delay)
                    last_start = monotonic()
                    progress(
                        f"candidate={candidate.name} session={session.name} "
                        f"turn={turn}/{session.calls} attempt={retries + 1}"
                    )
                    request_started = monotonic()
                    try:
                        response = completion(**completion_kwargs(candidate, messages))
                        latency = monotonic() - request_started
                        try:
                            action = _extract_action(response)
                        except (ValidationError, json.JSONDecodeError) as exc:
                            raise ScreenFailure(
                                f"schema-invalid response: {type(exc).__name__}",
                                category="model_failure",
                            ) from exc
                        if action.content != expected:
                            raise ScreenFailure(
                                "structured content did not equal the frozen nonce",
                                category="model_failure",
                            )
                        usage = _usage_payload(response)
                        metadata = _response_metadata(response)
                        records.append(
                            {
                                "session": session.name,
                                "turn": turn,
                                "attempts": retries + 1,
                                "latency_seconds": round(latency, 6),
                                "prompt_digest": _digest(messages),
                                "action_digest": _digest(action.model_dump()),
                                **usage,
                                **metadata,
                            }
                        )
                        running_totals = {
                            key: sum(int(record[key]) for record in records)
                            for key in (
                                "input_tokens",
                                "cached_input_tokens",
                                "output_tokens",
                            )
                        }
                        if (
                            running_totals["input_tokens"]
                            > MAX_INPUT_TOKENS_PER_CANDIDATE
                            or _estimated_cost(candidate, running_totals)
                            > MAX_ESTIMATED_COST_PER_CANDIDATE_USD
                        ):
                            raise ScreenFailure(
                                "frozen provider-screen cost guard crossed",
                                category="account_failure",
                            )
                        messages.append(
                            {"role": "assistant", "content": action.model_dump_json()}
                        )
                        break
                    except ScreenFailure:
                        raise
                    except Exception as exc:
                        if is_account_failure(exc):
                            raise ScreenFailure(
                                f"account/API readiness failure: {type(exc).__name__}",
                                category="account_failure",
                            ) from exc
                        if not is_retryable(exc):
                            raise ScreenFailure(
                                f"nonretryable provider failure: {type(exc).__name__}",
                                category="model_failure",
                            ) from exc
                        if retries >= len(RETRY_DELAYS_SECONDS):
                            failure_category = (
                                "account_failure"
                                if _status_code(exc) == 429
                                else "model_failure"
                            )
                            raise ScreenFailure(
                                f"transient retry budget exhausted: {type(exc).__name__}",
                                category=failure_category,
                            ) from exc
                        sleep(RETRY_DELAYS_SECONDS[retries])
                        retries += 1
                        total_retries += 1
    except ScreenFailure as exc:
        failure = {"category": exc.category, "message": str(exc)}

    totals = {
        key: sum(int(record[key]) for record in records)
        for key in ("input_tokens", "cached_input_tokens", "output_tokens")
    }
    passed = (
        failure is None
        and len(records) == EXPECTED_CALLS
        and total_retries <= MAX_TOTAL_TRANSIENT_RETRIES_FOR_PASS
    )
    if failure is None and total_retries > MAX_TOTAL_TRANSIENT_RETRIES_FOR_PASS:
        failure = {
            "category": "model_failure",
            "message": "total transient retries exceeded the frozen passing limit",
        }
    return {
        "candidate": asdict(candidate),
        "started_at": started_at,
        "completed_at": _utc_now(),
        "calls_expected": EXPECTED_CALLS,
        "calls_completed": len(records),
        "total_transient_retries": total_retries,
        "token_totals": totals,
        "estimated_cost_usd": round(_estimated_cost(candidate, totals), 6),
        "passed": passed,
        "failure": failure,
        "records": records,
    }


def run_screen(
    *,
    completion: Callable[..., Any] = litellm.completion,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    selected: Candidate | None = None
    status = "no_candidate_passed"
    for candidate in CANDIDATES:
        result = run_candidate(
            candidate,
            completion=completion,
            sleep=sleep,
            monotonic=monotonic,
            progress=progress,
        )
        results.append(result)
        if result["passed"]:
            selected = candidate
            status = "passed"
            break
        failure = result.get("failure") or {}
        if failure.get("category") == "account_failure":
            status = "account_not_ready"
            break
    return {
        "screen_version": SCREEN_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "selected_candidate": asdict(selected) if selected else None,
        "candidate_results": results,
        "real_g2_material_loaded": False,
        "g2b_task_calls_authorized": False,
        "next_gate": (
            "commit_and_tag_g2b_preregistration"
            if selected
            else "resolve_screen_failure_without_running_g2b"
        ),
    }


def public_summary(screen: dict[str, Any]) -> dict[str, Any]:
    results = []
    for candidate_result in screen["candidate_results"]:
        sanitized = dict(candidate_result)
        sanitized.pop("records", None)
        results.append(sanitized)
    return {**screen, "candidate_results": results}


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _archive(path: Path) -> None:
    if not path.exists():
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path.rename(path.with_name(f"{path.name}.archived.{stamp}"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "provider_screen",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--fresh", action="store_true")
    subparsers.add_parser("summarize")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results_dir = args.results_dir.resolve()
    private_path = results_dir / "private" / "screen.json"
    public_path = results_dir / "public" / "screen_summary.json"

    if args.command == "preflight":
        ready = bool(os.environ.get("OPENAI_API_KEY"))
        report = {
            "screen_version": SCREEN_VERSION,
            "openai_api_key_present": ready,
            "candidate_order": [candidate.model for candidate in CANDIDATES],
            "expected_calls_for_first_passing_candidate": EXPECTED_CALLS,
            "real_g2_material_loaded": False,
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if ready else 2

    if args.command == "summarize":
        if not public_path.is_file():
            raise SystemExit(f"No screen summary at {public_path}")
        print(public_path.read_text(encoding="utf-8"), end="")
        return 0

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")
    if args.fresh:
        _archive(private_path)
        _archive(public_path)
    elif private_path.exists() or public_path.exists():
        raise SystemExit("Existing provider screen found; use --fresh to archive it")

    screen = run_screen()
    _atomic_json(private_path, screen)
    summary = public_summary(screen)
    _atomic_json(public_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if screen["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
