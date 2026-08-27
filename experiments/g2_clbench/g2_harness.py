#!/usr/bin/env python3
"""Preregistered G2 matched-error structural-leverage runner.

The runner keeps raw actions and observations only in the ignored private cache.
Public summaries contain configuration, hashes, aggregate metrics, and digests.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import inspect
import json
import logging
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence


HARNESS_VERSION = "g2-v1"
FROZEN_CLBENCH_COMMIT = "5f8c50eb1e84b2eda2ef4faff757dfc812a0ea26"
FROZEN_MODEL = "groq/qwen/qwen3.6-27b"
FROZEN_PROVIDER_MODE = "litellm_chat"
FROZEN_TEMPERATURE = 0.0
FROZEN_REQUEST_SEEDS = (7, 17, 29, 41, 53)
FROZEN_TASK_SEED = 42
FROZEN_QUERY_BUDGET = 15
FROZEN_QUESTION_COUNT = 40
FROZEN_REQUEST_TIMEOUT_SECONDS = 60.0

FROZEN_DATA_HASHES = {
    "products.db": "edf8ee80ff125de0bfd6c37a1d185efa9e3037ce28eb1bd1d32ae0829bd264a6",
    "products_drifted.db": "a53d523f70604be0e4328f3722417895250ac35e1127cc504b609576aee70fad",
    "questions.json": "f98e515c14b74f4d30c29401fb79cca9ceeb15fe9ee2432b10265a4a649d1771",
    "questions_post_drift.json": "c05aa6a33def3878e6b655a1cba6454e55e1a9c16ded8c1fe53b7811986a45de",
}
FROZEN_OVERLAY_HASHES = {
    "0001-clbench-sampling-controls.patch": "20ffb6824461116580c3fab0554bb4cb6a4293024ebd855b0f45af7a12d64167",
    "0002-clbench-verdict-only-feedback.patch": "009f98693f2cb913744ca617a2fec03551e2408079964747297860664ad5ec88",
    "0003-clbench-verdict-token-flip.patch": "d78fceed1e0fcb0833c7c70a3ef38b44fd8155a195a7372d7af55f67ac696c50",
    "0004-clbench-exact-prefix-replay.patch": "f3e1d481471083df02ac5ba68bdc67fe44415c26191c71bae41ae37efd41c275",
    "0005-clbench-request-timeout-metadata.patch": "a64ec887d51777bf6c45e471a3d3504c50b06aebdd6a3aef935a5811dd4a276d",
    "0006-clbench-retry-groq-tool-use-failures.patch": "2004ccb89f46e7294fcb4c1979a6c4dc3536a4c1576af6a2fcde4be0f9c214a7",
    "0007-clbench-groq-json-object-mode.patch": "1fc58977baff299a7fe65f5e33aaadb640c780289aa22aac9c1d9ba5ffc6ad2e",
}

VERDICT_RE = re.compile(r"(EVALUATOR VERDICT:\s*)(PASS|FAIL)\b")
PRIVATE_FORMAT_VERSION = 1


@dataclass(frozen=True)
class Candidate:
    rank: int
    position: int
    high_run: int
    high_score: float
    low_run: int
    low_score: float
    gap: float
    remaining: int


CANDIDATES = (
    Candidate(1, 28, 0, 12.000, 4, 7.500, 4.500, 12),
    Candidate(2, 25, 3, 15.000, 1, 11.000, 4.000, 15),
    Candidate(3, 23, 0, 13.333, 4, 9.750, 3.583, 17),
    Candidate(4, 24, 3, 13.000, 2, 9.600, 3.400, 16),
    Candidate(5, 27, 4, 11.000, 2, 7.750, 3.250, 13),
    Candidate(6, 30, 4, 8.667, 1, 5.500, 3.167, 10),
    Candidate(7, 31, 4, 9.000, 0, 6.000, 3.000, 9),
    Candidate(8, 29, 0, 9.333, 3, 6.500, 2.833, 11),
    Candidate(9, 26, 2, 11.333, 1, 9.000, 2.333, 14),
    Candidate(10, 32, 0, 6.333, 1, 4.000, 2.333, 8),
)


class G2ValidityError(RuntimeError):
    """Raised when a frozen G2 validity gate fails."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _query_digest(query: Any) -> str:
    metadata = dict(query.metadata or {})
    metadata.pop("db_path", None)
    schema = query.response_schema.model_json_schema()
    return _digest(
        {
            "prompt": query.prompt,
            "response_schema": schema,
            "instance_id": query.instance_id,
            "instance_index": query.instance_index,
            "metadata": metadata,
        }
    )


def _observation_payload(observation: Any) -> dict[str, Any]:
    return {
        "content": observation.content,
        "instance_complete": bool(observation.instance_complete),
        "metadata": dict(observation.metadata or {}),
    }


def _state_snapshot(system: Any) -> dict[str, Any]:
    artifacts = system.get_run_artifacts() or {}
    return {
        "messages": artifacts.get("messages", list(getattr(system, "messages", []))),
        "interaction_count": artifacts.get(
            "interaction_count", getattr(system, "interaction_count", None)
        ),
        "has_truncated": artifacts.get(
            "has_truncated", getattr(system, "has_truncated_flag", None)
        ),
        "truncation_count": artifacts.get(
            "truncation_count", getattr(system, "truncation_count", None)
        ),
        "provider_state": artifacts.get("provider_state"),
    }


def _safe_usage_event(event: Any) -> dict[str, Any]:
    return {
        "call_type": event.call_type,
        "model": event.model,
        "provider": event.provider,
        "input_tokens": event.input_tokens,
        "output_tokens": event.output_tokens,
        "total_tokens": event.total_tokens,
        "reasoning_tokens": event.reasoning_tokens,
        "cached_input_tokens": event.cached_input_tokens,
        "cost_usd": event.cost_usd,
        "response_id": event.response_id,
        "metadata": dict(event.metadata or {}),
    }


def _action_payload(action: Any) -> dict[str, Any]:
    payload = action.model_dump(mode="json")
    if not isinstance(payload, dict):
        raise G2ValidityError("Structured action did not serialize to an object")
    return payload


def _question_num(query: Any) -> int:
    metadata = query.metadata or {}
    value = metadata.get("question_num")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if query.instance_index is None:
        raise G2ValidityError("Query is missing question_num and instance_index")
    return int(query.instance_index) + 1


def _extract_verdict(content: str) -> str:
    match = VERDICT_RE.search(content)
    if match is None:
        raise G2ValidityError("Terminal verdict-only feedback is malformed")
    return match.group(2)


def _assert_verdict_only_change(
    clean_content: str, counterfactual_content: str
) -> None:
    clean_match = VERDICT_RE.search(clean_content)
    counterfactual_match = VERDICT_RE.search(counterfactual_content)
    if clean_match is None or counterfactual_match is None:
        raise G2ValidityError("Intervention feedback is missing a verdict token")
    if clean_match.group(2) == counterfactual_match.group(2):
        raise G2ValidityError("Intervention verdict token was not flipped")
    clean_normalized = VERDICT_RE.sub(r"\1<VERDICT>", clean_content, count=1)
    counterfactual_normalized = VERDICT_RE.sub(
        r"\1<VERDICT>", counterfactual_content, count=1
    )
    if clean_normalized != counterfactual_normalized:
        raise G2ValidityError("More than the verdict token changed at intervention")


def _sanitize_response_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = dict(metadata or {})
    provider_state = payload.get("provider_state")
    if isinstance(provider_state, dict):
        payload["provider_state"] = {
            key: value
            for key, value in provider_state.items()
            if key not in {"previous_response_id", "encrypted_reasoning"}
        }
    return payload


@dataclass
class TurnRecord:
    question_num: int
    turn_in_question: int
    query_digest: str
    instance_id_digest: str
    action: dict[str, Any]
    action_digest: str
    assistant_record: str
    input_tokens: int
    response_metadata: dict[str, Any]
    state_before_observe_digest: str
    observation: dict[str, Any]
    observation_digest: str
    done: bool
    replayed: bool
    usage_events: list[dict[str, Any]]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TurnRecord":
        return cls(**dict(payload))


@dataclass
class QuestionOutcome:
    position: int
    instance_id_digest: str
    correct: bool
    true_verdict: str
    submitted_answer_digest: str
    num_queries: int
    num_actions: int
    regret: float
    timed_out: bool
    budget_exceeded: bool

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "QuestionOutcome":
        return cls(**dict(payload))


@dataclass
class Trajectory:
    kind: str
    request_seed: int
    run_index: int
    intervention_position: int | None
    config_digest: str
    turns: list[TurnRecord]
    outcomes: list[QuestionOutcome]
    score: float
    completed_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "private_format_version": PRIVATE_FORMAT_VERSION,
            "harness_version": HARNESS_VERSION,
            "kind": self.kind,
            "request_seed": self.request_seed,
            "run_index": self.run_index,
            "intervention_position": self.intervention_position,
            "config_digest": self.config_digest,
            "turns": [asdict(turn) for turn in self.turns],
            "outcomes": [asdict(outcome) for outcome in self.outcomes],
            "score": self.score,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Trajectory":
        if payload.get("private_format_version") != PRIVATE_FORMAT_VERSION:
            raise G2ValidityError("Private cache format version does not match")
        if payload.get("harness_version") != HARNESS_VERSION:
            raise G2ValidityError("Private cache harness version does not match")
        return cls(
            kind=str(payload["kind"]),
            request_seed=int(payload["request_seed"]),
            run_index=int(payload["run_index"]),
            intervention_position=(
                None
                if payload.get("intervention_position") is None
                else int(payload["intervention_position"])
            ),
            config_digest=str(payload["config_digest"]),
            turns=[TurnRecord.from_dict(item) for item in payload["turns"]],
            outcomes=[QuestionOutcome.from_dict(item) for item in payload["outcomes"]],
            score=float(payload["score"]),
            completed_at=str(payload["completed_at"]),
        )


def _config_payload(request_seed: int, run_index: int) -> dict[str, Any]:
    return {
        "harness_version": HARNESS_VERSION,
        "upstream_commit": FROZEN_CLBENCH_COMMIT,
        "model": FROZEN_MODEL,
        "provider_mode": FROZEN_PROVIDER_MODE,
        "temperature": FROZEN_TEMPERATURE,
        "request_seed": request_seed,
        "task_seed": FROZEN_TASK_SEED,
        "query_budget": FROZEN_QUERY_BUDGET,
        "question_count": FROZEN_QUESTION_COUNT,
        "request_timeout_seconds": FROZEN_REQUEST_TIMEOUT_SECONDS,
        "feedback_mode": "verdict_only",
    }


def _extract_outcomes(task_result: Any) -> list[QuestionOutcome]:
    history = task_result.metrics.get("question_history")
    if not isinstance(history, list):
        raise G2ValidityError("Task result is missing question_history")
    outcomes: list[QuestionOutcome] = []
    for position, record in enumerate(history, start=1):
        correct = bool(record["correct"])
        outcomes.append(
            QuestionOutcome(
                position=position,
                instance_id_digest=_digest(str(record["question_id"])),
                correct=correct,
                true_verdict="PASS" if correct else "FAIL",
                submitted_answer_digest=_digest(
                    str(record.get("submitted_answer", ""))
                ),
                num_queries=int(record["num_queries"]),
                num_actions=int(
                    record.get("num_actions", int(record["num_queries"]) + 1)
                ),
                regret=float(record.get("regret", record["num_queries"])),
                timed_out=bool(record.get("timed_out", False)),
                budget_exceeded=bool(record.get("budget_exceeded", False)),
            )
        )
    return outcomes


def _live_turn(
    task: Any, system: Any, query: Any, turn_in_question: int
) -> tuple[TurnRecord, Any]:
    stale_events = system.consume_usage_events()
    if stale_events:
        raise G2ValidityError("Usage buffer was not empty before a live turn")
    response = system.respond(query)
    usage_events = system.consume_usage_events()
    if len(usage_events) != 1:
        raise G2ValidityError(
            f"Expected one billable usage event, observed {len(usage_events)}"
        )
    input_tokens = usage_events[0].input_tokens
    if isinstance(input_tokens, bool) or not isinstance(input_tokens, int):
        raise G2ValidityError("Live response is missing API-reported input_tokens")
    if not getattr(system, "messages", None):
        raise G2ValidityError("ICL system did not persist the assistant record")
    assistant_record = system.messages[-1]["content"]
    action = _action_payload(response.action)
    state_digest = _digest(_state_snapshot(system))
    step_result = task.step(response)
    observation = _observation_payload(step_result.observation)
    record = TurnRecord(
        question_num=_question_num(query),
        turn_in_question=turn_in_question,
        query_digest=_query_digest(query),
        instance_id_digest=_digest(str(query.instance_id)),
        action=action,
        action_digest=_digest(action),
        assistant_record=assistant_record,
        input_tokens=input_tokens,
        response_metadata=_sanitize_response_metadata(response.metadata),
        state_before_observe_digest=state_digest,
        observation=observation,
        observation_digest=_digest(observation),
        done=bool(step_result.done),
        replayed=False,
        usage_events=[_safe_usage_event(event) for event in usage_events],
    )
    system.observe(step_result.observation, step_result.next_query)
    return record, step_result


def run_clean_trajectory(
    task: Any,
    system: Any,
    *,
    request_seed: int,
    run_index: int,
    progress: Callable[[str], None] = print,
) -> Trajectory:
    """Run one complete truthful trajectory and return its private replay cache."""
    query = task.reset()
    system.reset()
    system.consume_usage_events()
    turns: list[TurnRecord] = []
    turn_counts: dict[int, int] = {}
    done = False
    while not done:
        question_num = _question_num(query)
        turn_counts[question_num] = turn_counts.get(question_num, 0) + 1
        progress(
            f"seed={request_seed} clean_run={run_index} "
            f"question={question_num}/{FROZEN_QUESTION_COUNT} "
            f"turn={turn_counts[question_num]}"
        )
        record, step_result = _live_turn(task, system, query, turn_counts[question_num])
        turns.append(record)
        done = bool(step_result.done)
        query = step_result.next_query
        if not done and query is None:
            raise G2ValidityError("Task returned no next query before completion")
    task_result = task.evaluate()
    config_digest = _digest(_config_payload(request_seed, run_index))
    return Trajectory(
        kind="clean",
        request_seed=request_seed,
        run_index=run_index,
        intervention_position=None,
        config_digest=config_digest,
        turns=turns,
        outcomes=_extract_outcomes(task_result),
        score=float(task_result.score),
        completed_at=_utc_now(),
    )


def _make_response(query: Any, turn: TurnRecord) -> Any:
    from src.interface import Response

    action = query.response_schema.model_validate(turn.action)
    return Response(action=action, metadata=dict(turn.response_metadata))


def _replay_turn(
    task: Any,
    system: Any,
    query: Any,
    clean_turn: TurnRecord,
    *,
    intervention_position: int,
) -> tuple[TurnRecord, Any]:
    if _query_digest(query) != clean_turn.query_digest:
        raise G2ValidityError("Replay query digest differs from the clean prefix")
    if system.consume_usage_events():
        raise G2ValidityError("Usage buffer was not empty before replay")
    system.replay_response(
        query,
        assistant_record=clean_turn.assistant_record,
        input_tokens=clean_turn.input_tokens,
    )
    if system.consume_usage_events():
        raise G2ValidityError("Replay made or recorded a provider call")
    state_digest = _digest(_state_snapshot(system))
    if state_digest != clean_turn.state_before_observe_digest:
        raise G2ValidityError("Replayed ICL state differs from the clean state")
    response = _make_response(query, clean_turn)
    step_result = task.step(response)
    observation = _observation_payload(step_result.observation)
    is_intervention_terminal = (
        clean_turn.question_num == intervention_position
        and clean_turn.observation["instance_complete"]
    )
    if is_intervention_terminal:
        if observation["metadata"] != clean_turn.observation["metadata"]:
            raise G2ValidityError("Intervention changed task observation metadata")
        _assert_verdict_only_change(
            clean_turn.observation["content"], observation["content"]
        )
    elif observation != clean_turn.observation:
        raise G2ValidityError("A pre-intervention observation differs from clean")
    record = TurnRecord(
        question_num=clean_turn.question_num,
        turn_in_question=clean_turn.turn_in_question,
        query_digest=clean_turn.query_digest,
        instance_id_digest=clean_turn.instance_id_digest,
        action=dict(clean_turn.action),
        action_digest=clean_turn.action_digest,
        assistant_record=clean_turn.assistant_record,
        input_tokens=clean_turn.input_tokens,
        response_metadata=dict(clean_turn.response_metadata),
        state_before_observe_digest=state_digest,
        observation=observation,
        observation_digest=_digest(observation),
        done=bool(step_result.done),
        replayed=True,
        usage_events=[],
    )
    system.observe(step_result.observation, step_result.next_query)
    return record, step_result


def run_counterfactual_trajectory(
    task: Any,
    system: Any,
    clean: Trajectory,
    *,
    intervention_position: int,
    progress: Callable[[str], None] = print,
) -> Trajectory:
    """Replay through the intervention response, then continue live."""
    query = task.reset()
    system.reset()
    system.consume_usage_events()
    turns: list[TurnRecord] = []
    turn_counts: dict[int, int] = {}
    clean_prefix = [
        turn for turn in clean.turns if turn.question_num <= intervention_position
    ]
    if not clean_prefix or not (
        clean_prefix[-1].question_num == intervention_position
        and clean_prefix[-1].observation["instance_complete"]
    ):
        raise G2ValidityError("Clean cache does not include the intervention response")

    done = False
    for clean_turn in clean_prefix:
        if done or query is None:
            raise G2ValidityError("Task ended before the clean prefix was replayed")
        progress(
            f"seed={clean.request_seed} counterfactual_run={clean.run_index} "
            f"question={clean_turn.question_num}/{FROZEN_QUESTION_COUNT} "
            "mode=replay"
        )
        record, step_result = _replay_turn(
            task,
            system,
            query,
            clean_turn,
            intervention_position=intervention_position,
        )
        turns.append(record)
        done = bool(step_result.done)
        query = step_result.next_query
        turn_counts[clean_turn.question_num] = clean_turn.turn_in_question

    while not done:
        if query is None:
            raise G2ValidityError("Task returned no next query after intervention")
        question_num = _question_num(query)
        turn_counts[question_num] = turn_counts.get(question_num, 0) + 1
        progress(
            f"seed={clean.request_seed} counterfactual_run={clean.run_index} "
            f"question={question_num}/{FROZEN_QUESTION_COUNT} "
            f"turn={turn_counts[question_num]} mode=live"
        )
        record, step_result = _live_turn(task, system, query, turn_counts[question_num])
        turns.append(record)
        done = bool(step_result.done)
        query = step_result.next_query

    task_result = task.evaluate()
    return Trajectory(
        kind="counterfactual",
        request_seed=clean.request_seed,
        run_index=clean.run_index,
        intervention_position=intervention_position,
        config_digest=clean.config_digest,
        turns=turns,
        outcomes=_extract_outcomes(task_result),
        score=float(task_result.score),
        completed_at=_utc_now(),
    )


def _terminal_turn(trajectory: Trajectory, position: int) -> TurnRecord:
    matches = [
        turn
        for turn in trajectory.turns
        if turn.question_num == position and turn.observation["instance_complete"]
    ]
    if len(matches) != 1:
        raise G2ValidityError(
            f"Expected one terminal turn at position {position}, found {len(matches)}"
        )
    return matches[0]


def _outcome(trajectory: Trajectory, position: int) -> QuestionOutcome:
    matches = [item for item in trajectory.outcomes if item.position == position]
    if len(matches) != 1:
        raise G2ValidityError(
            f"Expected one outcome at position {position}, found {len(matches)}"
        )
    return matches[0]


def intervention_status(trajectory: Trajectory, position: int) -> dict[str, Any]:
    """Return only the predeclared fields allowed during candidate selection."""
    outcome = _outcome(trajectory, position)
    terminal = _terminal_turn(trajectory, position)
    true_verdict = outcome.true_verdict
    observation_true_verdict = terminal.observation["metadata"].get("true_verdict")
    if observation_true_verdict != true_verdict:
        raise G2ValidityError("Task outcome and observation true verdict disagree")
    return {
        "normal_completion": not outcome.timed_out and not outcome.budget_exceeded,
        "true_verdict": true_verdict,
    }


def candidate_is_eligible(
    candidate: Candidate,
    high_clean: Trajectory,
    low_clean: Trajectory,
) -> tuple[bool, str]:
    high = intervention_status(high_clean, candidate.position)
    low = intervention_status(low_clean, candidate.position)
    if not high["normal_completion"] or not low["normal_completion"]:
        return False, "intervention_timeout_or_budget_exhaustion"
    if high["true_verdict"] != low["true_verdict"]:
        return False, "true_verdict_mismatch"
    return True, "eligible"


def select_first_eligible_candidate(
    clean_by_run: Mapping[int, Trajectory],
    candidates: Sequence[Candidate] = CANDIDATES,
) -> tuple[Candidate | None, list[dict[str, Any]]]:
    """Select using intervention validity and true verdict only."""
    checks: list[dict[str, Any]] = []
    for candidate in candidates:
        high = clean_by_run.get(candidate.high_run)
        low = clean_by_run.get(candidate.low_run)
        if high is None or low is None:
            checks.append({"rank": candidate.rank, "reason": "clean_run_missing"})
            continue
        eligible, reason = candidate_is_eligible(candidate, high, low)
        checks.append({"rank": candidate.rank, "reason": reason})
        if eligible:
            return candidate, checks
    return None, checks


def _action_sequences(trajectory: Trajectory) -> dict[int, list[str]]:
    result: dict[int, list[str]] = {}
    for turn in trajectory.turns:
        result.setdefault(turn.question_num, []).append(turn.action_digest)
    return result


def _terminal_context(trajectory: Trajectory) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for turn in trajectory.turns:
        if not turn.observation["instance_complete"]:
            continue
        result[turn.question_num] = {
            "context_tokens": turn.response_metadata.get("context_tokens"),
            "truncation_count": turn.response_metadata.get("truncation_count"),
        }
    return result


def audit_counterfactual(
    clean: Trajectory,
    counterfactual: Trajectory,
    intervention_position: int,
    *,
    expected_questions: int = FROZEN_QUESTION_COUNT,
) -> dict[str, Any]:
    reasons: list[str] = []
    if len(clean.outcomes) != expected_questions:
        reasons.append("clean_question_count")
    if len(counterfactual.outcomes) != expected_questions:
        reasons.append("counterfactual_question_count")

    clean_prefix = [
        turn for turn in clean.turns if turn.question_num <= intervention_position
    ]
    counterfactual_prefix = [
        turn
        for turn in counterfactual.turns
        if turn.question_num <= intervention_position
    ]
    if len(clean_prefix) != len(counterfactual_prefix):
        reasons.append("prefix_length")
    else:
        for clean_turn, counterfactual_turn in zip(
            clean_prefix, counterfactual_prefix, strict=True
        ):
            if clean_turn.query_digest != counterfactual_turn.query_digest:
                reasons.append("prefix_query")
                break
            if clean_turn.action_digest != counterfactual_turn.action_digest:
                reasons.append("prefix_action")
                break
            if clean_turn.assistant_record != counterfactual_turn.assistant_record:
                reasons.append("prefix_assistant_record")
                break
            if (
                clean_turn.state_before_observe_digest
                != counterfactual_turn.state_before_observe_digest
            ):
                reasons.append("prefix_system_state")
                break
            is_intervention_terminal = (
                clean_turn.question_num == intervention_position
                and clean_turn.observation["instance_complete"]
            )
            if (
                not is_intervention_terminal
                and clean_turn.observation_digest
                != counterfactual_turn.observation_digest
            ):
                reasons.append("prefix_observation")
                break
            if not counterfactual_turn.replayed or counterfactual_turn.usage_events:
                reasons.append("prefix_not_zero_call_replay")
                break

    try:
        clean_intervention = _terminal_turn(clean, intervention_position)
        counterfactual_intervention = _terminal_turn(
            counterfactual, intervention_position
        )
        _assert_verdict_only_change(
            clean_intervention.observation["content"],
            counterfactual_intervention.observation["content"],
        )
        if (
            clean_intervention.observation["metadata"]
            != counterfactual_intervention.observation["metadata"]
        ):
            reasons.append("intervention_metadata")
        if asdict(_outcome(clean, intervention_position)) != asdict(
            _outcome(counterfactual, intervention_position)
        ):
            reasons.append("intervention_scoring")
    except G2ValidityError:
        reasons.append("intervention_verdict_only")

    mismatches: list[tuple[int, str, str]] = []
    clean_mismatches: list[int] = []
    for outcome in clean.outcomes:
        try:
            delivered = _extract_verdict(
                _terminal_turn(clean, outcome.position).observation["content"]
            )
        except G2ValidityError:
            reasons.append("malformed_clean_terminal_verdict")
            continue
        if delivered != outcome.true_verdict:
            clean_mismatches.append(outcome.position)
    if clean_mismatches:
        reasons.append("clean_evaluator_not_truthful")

    for outcome in counterfactual.outcomes:
        try:
            delivered = _extract_verdict(
                _terminal_turn(counterfactual, outcome.position).observation["content"]
            )
        except G2ValidityError:
            reasons.append("malformed_terminal_verdict")
            continue
        if delivered != outcome.true_verdict:
            mismatches.append((outcome.position, outcome.true_verdict, delivered))
    if len(mismatches) != 1 or mismatches[0][0] != intervention_position:
        reasons.append("evaluator_error_count_or_location")

    confusion_type = None
    if len(mismatches) == 1:
        _, truth, delivered = mismatches[0]
        if truth == "PASS" and delivered == "FAIL":
            confusion_type = "false_negative"
        elif truth == "FAIL" and delivered == "PASS":
            confusion_type = "false_positive"
        else:
            reasons.append("invalid_confusion_type")

    return {
        "valid": not reasons,
        "reasons": sorted(set(reasons)),
        "exact_prefix_replay": not any(
            reason.startswith("prefix_") for reason in reasons
        ),
        "provider_calls_in_replayed_prefix": sum(
            len(turn.usage_events) for turn in counterfactual_prefix
        ),
        "clean_evaluator_errors": len(clean_mismatches),
        "evaluator_errors": len(mismatches),
        "evaluator_accuracy": (
            (expected_questions - len(mismatches)) / expected_questions
            if expected_questions
            else 0.0
        ),
        "confusion_type": confusion_type,
    }


def arm_effect(
    clean: Trajectory,
    counterfactual: Trajectory,
    intervention_position: int,
) -> dict[str, Any]:
    clean_outcomes = {item.position: item for item in clean.outcomes}
    counterfactual_outcomes = {item.position: item for item in counterfactual.outcomes}
    positions = sorted(
        position
        for position in clean_outcomes
        if position > intervention_position and position in counterfactual_outcomes
    )
    if not positions:
        raise G2ValidityError("No downstream outcomes are available")
    clean_actions = _action_sequences(clean)
    counterfactual_actions = _action_sequences(counterfactual)
    clean_context = _terminal_context(clean)
    counterfactual_context = _terminal_context(counterfactual)

    regret_deltas = [
        counterfactual_outcomes[position].regret - clean_outcomes[position].regret
        for position in positions
    ]
    accuracy_deltas = [
        float(clean_outcomes[position].correct)
        - float(counterfactual_outcomes[position].correct)
        for position in positions
    ]
    query_deltas = [
        counterfactual_outcomes[position].num_queries
        - clean_outcomes[position].num_queries
        for position in positions
    ]
    divergences = [
        clean_actions.get(position, []) != counterfactual_actions.get(position, [])
        for position in positions
    ]
    first_divergence = next(
        (
            position - intervention_position
            for position, value in zip(positions, divergences, strict=True)
            if value
        ),
        None,
    )

    context_deltas: list[float] = []
    truncation_deltas: list[float] = []
    for position in positions:
        clean_item = clean_context.get(position, {})
        counterfactual_item = counterfactual_context.get(position, {})
        clean_tokens = clean_item.get("context_tokens")
        counterfactual_tokens = counterfactual_item.get("context_tokens")
        if isinstance(clean_tokens, (int, float)) and isinstance(
            counterfactual_tokens, (int, float)
        ):
            context_deltas.append(float(counterfactual_tokens - clean_tokens))
        clean_truncation = clean_item.get("truncation_count")
        counterfactual_truncation = counterfactual_item.get("truncation_count")
        if isinstance(clean_truncation, (int, float)) and isinstance(
            counterfactual_truncation, (int, float)
        ):
            truncation_deltas.append(
                float(counterfactual_truncation - clean_truncation)
            )

    return {
        "downstream_questions": len(positions),
        "signed_regret_harm": statistics.mean(regret_deltas) / FROZEN_QUERY_BUDGET,
        "downstream_accuracy_harm": statistics.mean(accuracy_deltas),
        "normalized_exploratory_query_excess": statistics.mean(query_deltas)
        / FROZEN_QUERY_BUDGET,
        "first_downstream_action_divergence": first_divergence,
        "downstream_action_divergence_rate": statistics.mean(
            float(value) for value in divergences
        ),
        "mean_context_token_excess": (
            statistics.mean(context_deltas) if context_deltas else None
        ),
        "mean_truncation_count_excess": (
            statistics.mean(truncation_deltas) if truncation_deltas else None
        ),
        "persistence": [
            {
                "distance": position - intervention_position,
                "normalized_regret_delta": regret_delta / FROZEN_QUERY_BUDGET,
                "action_diverged": action_diverged,
            }
            for position, regret_delta, action_diverged in zip(
                positions, regret_deltas, divergences, strict=True
            )
        ],
    }


def paired_metrics(
    candidate: Candidate,
    high_clean: Trajectory,
    high_counterfactual: Trajectory,
    low_clean: Trajectory,
    low_counterfactual: Trajectory,
) -> dict[str, Any]:
    high_audit = audit_counterfactual(
        high_clean, high_counterfactual, candidate.position
    )
    low_audit = audit_counterfactual(low_clean, low_counterfactual, candidate.position)
    reasons: list[str] = []
    if not high_audit["valid"]:
        reasons.extend(f"high:{reason}" for reason in high_audit["reasons"])
    if not low_audit["valid"]:
        reasons.extend(f"low:{reason}" for reason in low_audit["reasons"])
    if high_audit["confusion_type"] != low_audit["confusion_type"]:
        reasons.append("matched_confusion_type")
    if candidate.remaining != FROZEN_QUESTION_COUNT - candidate.position:
        reasons.append("candidate_horizon")

    high = arm_effect(high_clean, high_counterfactual, candidate.position)
    low = arm_effect(low_clean, low_counterfactual, candidate.position)
    if high["downstream_questions"] != candidate.remaining:
        reasons.append("high_horizon")
    if low["downstream_questions"] != candidate.remaining:
        reasons.append("low_horizon")

    return {
        "valid": not reasons,
        "reasons": reasons,
        "candidate": asdict(candidate),
        "confusion_type": high_audit["confusion_type"],
        "high_audit": high_audit,
        "low_audit": low_audit,
        "high": high,
        "low": low,
        "matched_difference_D": high["signed_regret_harm"] - low["signed_regret_harm"],
    }


def _git_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _import_clbench(clbench_root: Path) -> tuple[Any, Any]:
    root_string = str(clbench_root.resolve())
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    from src.systems.icl.system import ICLSystem
    from src.tasks.database_exploration.task import DatabaseExploration

    source_path = Path(inspect.getfile(ICLSystem)).resolve()
    if not source_path.is_relative_to(clbench_root.resolve()):
        raise G2ValidityError(f"Imported CL-Bench from unexpected path: {source_path}")
    return DatabaseExploration, ICLSystem


def preflight(clbench_root: Path) -> dict[str, Any]:
    clbench_root = clbench_root.resolve()
    reasons: list[str] = []
    try:
        commit = _git_head(clbench_root)
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
        reasons.append("upstream_git_commit_unavailable")
    if commit != FROZEN_CLBENCH_COMMIT:
        reasons.append("upstream_commit_mismatch")

    data_dir = clbench_root / "data" / "database_exploration"
    observed_hashes: dict[str, str | None] = {}
    for filename, expected in FROZEN_DATA_HASHES.items():
        path = data_dir / filename
        observed = _sha256_file(path) if path.is_file() else None
        observed_hashes[filename] = observed
        if observed != expected:
            reasons.append(f"data_hash:{filename}")

    patch_dir = Path(__file__).resolve().parent / "patches"
    observed_overlay_hashes: dict[str, str | None] = {}
    for filename, expected in FROZEN_OVERLAY_HASHES.items():
        path = patch_dir / filename
        observed = _sha256_file(path) if path.is_file() else None
        observed_overlay_hashes[filename] = observed
        if observed != expected:
            reasons.append(f"overlay_hash:{filename}")

    overlay_api: dict[str, Any] = {}
    try:
        DatabaseExploration, ICLSystem = _import_clbench(clbench_root)
        task_params = inspect.signature(DatabaseExploration.__init__).parameters
        system_params = inspect.signature(ICLSystem.__init__).parameters
        overlay_api = {
            "feedback_mode": "feedback_mode" in task_params,
            "verdict_flip_question_num": "verdict_flip_question_num" in task_params,
            "temperature": "temperature" in system_params,
            "seed": "seed" in system_params,
            "request_timeout_seconds": "request_timeout_seconds" in system_params,
            "replay_response": callable(getattr(ICLSystem, "replay_response", None)),
        }
        for name, present in overlay_api.items():
            if not present:
                reasons.append(f"overlay_api:{name}")
    except Exception as exc:
        overlay_api = {"import_error": f"{type(exc).__name__}: {exc}"}
        reasons.append("overlay_import")

    return {
        "valid": not reasons,
        "reasons": reasons,
        "clbench_root": str(clbench_root),
        "observed_commit": commit,
        "expected_commit": FROZEN_CLBENCH_COMMIT,
        "observed_data_hashes": observed_hashes,
        "expected_data_hashes": FROZEN_DATA_HASHES,
        "observed_overlay_hashes": observed_overlay_hashes,
        "expected_overlay_hashes": FROZEN_OVERLAY_HASHES,
        "overlay_api": overlay_api,
    }


def make_task_system(
    clbench_root: Path,
    *,
    request_seed: int,
    run_index: int,
    verdict_flip_question_num: int | None,
) -> tuple[Any, Any]:
    DatabaseExploration, ICLSystem = _import_clbench(clbench_root)
    task = DatabaseExploration(
        schedule="default",
        run_index=run_index,
        max_queries_per_question=FROZEN_QUERY_BUDGET,
        seed=FROZEN_TASK_SEED,
        response_timeout_seconds=FROZEN_REQUEST_TIMEOUT_SECONDS,
        feedback_mode="verdict_only",
        verdict_flip_question_num=verdict_flip_question_num,
    )
    system = ICLSystem(
        model=FROZEN_MODEL,
        provider_mode=FROZEN_PROVIDER_MODE,
        temperature=FROZEN_TEMPERATURE,
        seed=request_seed,
        request_timeout_seconds=FROZEN_REQUEST_TIMEOUT_SECONDS,
    )
    return task, system


def _trajectory_path(
    results_dir: Path,
    request_seed: int,
    kind: str,
    run_index: int,
    intervention_position: int | None = None,
) -> Path:
    suffix = (
        f"_position_{intervention_position}"
        if intervention_position is not None
        else ""
    )
    return (
        results_dir
        / "private"
        / f"seed_{request_seed}"
        / f"{kind}_run_{run_index}{suffix}.json"
    )


def _load_trajectory(path: Path, expected_config_digest: str) -> Trajectory:
    payload = json.loads(path.read_text(encoding="utf-8"))
    trajectory = Trajectory.from_dict(payload)
    if trajectory.config_digest != expected_config_digest:
        raise G2ValidityError(
            f"Cached trajectory configuration differs: {path}. Use --fresh."
        )
    return trajectory


def _archive_seed_cache(results_dir: Path, request_seed: int) -> None:
    root = results_dir / "private" / f"seed_{request_seed}"
    if not root.exists():
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = root.with_name(f"{root.name}.archived.{stamp}")
    root.rename(destination)


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _get_or_run_clean(
    clbench_root: Path,
    results_dir: Path,
    request_seed: int,
    run_index: int,
    progress: Callable[[str], None],
) -> Trajectory:
    path = _trajectory_path(results_dir, request_seed, "clean", run_index)
    config_digest = _digest(_config_payload(request_seed, run_index))
    if path.is_file():
        progress(f"seed={request_seed} clean_run={run_index} cache=hit")
        return _load_trajectory(path, config_digest)
    task, system = make_task_system(
        clbench_root,
        request_seed=request_seed,
        run_index=run_index,
        verdict_flip_question_num=None,
    )
    trajectory = run_clean_trajectory(
        task,
        system,
        request_seed=request_seed,
        run_index=run_index,
        progress=progress,
    )
    _atomic_json(path, trajectory.to_dict())
    return trajectory


def _get_or_run_counterfactual(
    clbench_root: Path,
    results_dir: Path,
    clean: Trajectory,
    intervention_position: int,
    progress: Callable[[str], None],
) -> Trajectory:
    path = _trajectory_path(
        results_dir,
        clean.request_seed,
        "counterfactual",
        clean.run_index,
        intervention_position,
    )
    if path.is_file():
        progress(
            f"seed={clean.request_seed} counterfactual_run={clean.run_index} cache=hit"
        )
        return _load_trajectory(path, clean.config_digest)
    task, system = make_task_system(
        clbench_root,
        request_seed=clean.request_seed,
        run_index=clean.run_index,
        verdict_flip_question_num=intervention_position,
    )
    trajectory = run_counterfactual_trajectory(
        task,
        system,
        clean,
        intervention_position=intervention_position,
        progress=progress,
    )
    _atomic_json(path, trajectory.to_dict())
    return trajectory


def _run_seed_in_clbench_directory(
    clbench_root: Path,
    results_dir: Path,
    request_seed: int,
    *,
    fresh: bool = False,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    if request_seed not in FROZEN_REQUEST_SEEDS:
        raise ValueError(f"Seed {request_seed} is not preregistered")
    if fresh:
        _archive_seed_cache(results_dir, request_seed)

    clean_by_run: dict[int, Trajectory] = {}
    candidate_checks: list[dict[str, Any]] = []
    selected: Candidate | None = None
    for candidate in CANDIDATES:
        for run_index in (candidate.high_run, candidate.low_run):
            if run_index not in clean_by_run:
                clean_by_run[run_index] = _get_or_run_clean(
                    clbench_root,
                    results_dir,
                    request_seed,
                    run_index,
                    progress,
                )
        eligible, reason = candidate_is_eligible(
            candidate,
            clean_by_run[candidate.high_run],
            clean_by_run[candidate.low_run],
        )
        candidate_checks.append({"rank": candidate.rank, "reason": reason})
        if eligible:
            selected = candidate
            break

    public_path = results_dir / "public" / f"seed_{request_seed}.json"
    if selected is None:
        summary = {
            "harness_version": HARNESS_VERSION,
            "request_seed": request_seed,
            "eligible": False,
            "valid": False,
            "candidate_checks": candidate_checks,
            "result": "ineligible",
            "completed_at": _utc_now(),
        }
        _atomic_json(public_path, summary)
        return summary

    high_clean = clean_by_run[selected.high_run]
    low_clean = clean_by_run[selected.low_run]
    high_counterfactual = _get_or_run_counterfactual(
        clbench_root,
        results_dir,
        high_clean,
        selected.position,
        progress,
    )
    low_counterfactual = _get_or_run_counterfactual(
        clbench_root,
        results_dir,
        low_clean,
        selected.position,
        progress,
    )
    metrics = paired_metrics(
        selected,
        high_clean,
        high_counterfactual,
        low_clean,
        low_counterfactual,
    )
    summary = {
        "harness_version": HARNESS_VERSION,
        "request_seed": request_seed,
        "eligible": True,
        "valid": metrics["valid"],
        "candidate_checks": candidate_checks,
        "configuration": _config_payload(request_seed, selected.high_run),
        "metrics": metrics,
        "trajectory_digests": {
            "high_clean": _digest(high_clean.to_dict()),
            "high_counterfactual": _digest(high_counterfactual.to_dict()),
            "low_clean": _digest(low_clean.to_dict()),
            "low_counterfactual": _digest(low_counterfactual.to_dict()),
        },
        "completed_at": _utc_now(),
    }
    _atomic_json(public_path, summary)
    return summary


def run_seed(
    clbench_root: Path,
    results_dir: Path,
    request_seed: int,
    *,
    fresh: bool = False,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Run one frozen seed with CL-Bench-relative fixture paths resolved safely."""
    with _working_directory(clbench_root.resolve()):
        return _run_seed_in_clbench_directory(
            clbench_root,
            results_dir,
            request_seed,
            fresh=fresh,
            progress=progress,
        )


def aggregate_results(results_dir: Path) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    for seed in FROZEN_REQUEST_SEEDS:
        path = results_dir / "public" / f"seed_{seed}.json"
        if path.is_file():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    valid = [item for item in summaries if item.get("eligible") and item.get("valid")]
    differences = [float(item["metrics"]["matched_difference_D"]) for item in valid]
    high_harms = [
        float(item["metrics"]["high"]["signed_regret_harm"]) for item in valid
    ]
    eligible_count = sum(bool(item.get("eligible")) for item in summaries)
    all_five_eligible = eligible_count == len(FROZEN_REQUEST_SEEDS)
    all_five_valid = len(valid) == len(FROZEN_REQUEST_SEEDS)
    positive_d_count = sum(value > 0 for value in differences)
    decision_positive = (
        all_five_eligible
        and all_five_valid
        and statistics.mean(differences) > 0
        and positive_d_count >= 4
        and statistics.mean(high_harms) > 0
    )
    if not all_five_eligible:
        decision = "inconclusive_fewer_than_five_eligible"
    elif not all_five_valid:
        decision = "invalid"
    elif decision_positive:
        decision = "positive"
    else:
        decision = "negative_or_inconclusive"
    aggregate = {
        "harness_version": HARNESS_VERSION,
        "seeds_present": [item["request_seed"] for item in summaries],
        "eligible_seeds": eligible_count,
        "valid_seeds": len(valid),
        "mean_D": statistics.mean(differences) if differences else None,
        "seeds_with_D_positive": positive_d_count,
        "mean_H_high": statistics.mean(high_harms) if high_harms else None,
        "decision": decision,
        "preregistered_rule_met": decision_positive,
        "generated_at": _utc_now(),
    }
    _atomic_json(results_dir / "public" / "g2_aggregate.json", aggregate)
    return aggregate


def _default_clbench_root() -> Path:
    return Path(
        os.environ.get(
            "CLBENCH_ROOT", str(Path.home() / "Downloads" / "continual-learning-bench")
        )
    )


def _default_results_dir() -> Path:
    return Path(__file__).resolve().parent / "results"


def _configure_logging(results_dir: Path) -> None:
    log_path = results_dir / "private" / "g2_runtime.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clbench-root", type=Path, default=_default_clbench_root())
    parser.add_argument("--results-dir", type=Path, default=_default_results_dir())
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight", help="Verify frozen commit, data, and overlays")
    run_parser = subparsers.add_parser("run", help="Run frozen G2 seeds")
    run_parser.add_argument(
        "--seed",
        choices=["all", *(str(seed) for seed in FROZEN_REQUEST_SEEDS)],
        default="all",
    )
    run_parser.add_argument(
        "--fresh",
        action="store_true",
        help="Archive existing private cache for the selected seed(s)",
    )
    subparsers.add_parser("summarize", help="Aggregate existing public seed summaries")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    clbench_root = args.clbench_root.resolve()
    results_dir = args.results_dir.resolve()
    if args.command == "preflight":
        report = preflight(clbench_root)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["valid"] else 2
    if args.command == "summarize":
        print(json.dumps(aggregate_results(results_dir), indent=2, sort_keys=True))
        return 0

    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit("GROQ_API_KEY is not set")
    report = preflight(clbench_root)
    if not report["valid"]:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2
    _configure_logging(results_dir)
    seeds: Iterable[int]
    if args.seed == "all":
        seeds = FROZEN_REQUEST_SEEDS
    else:
        seeds = (int(args.seed),)
    for seed in seeds:
        summary = run_seed(
            clbench_root,
            results_dir,
            seed,
            fresh=args.fresh,
        )
        print(f"seed={seed} eligible={summary['eligible']} valid={summary['valid']}")
    print(json.dumps(aggregate_results(results_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
