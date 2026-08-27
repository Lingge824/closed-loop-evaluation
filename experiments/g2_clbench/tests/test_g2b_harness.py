from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest
from pydantic import BaseModel

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from g2b_harness import (  # noqa: E402
    Candidate,
    FROZEN_MODEL,
    FROZEN_PROVIDER_MODE,
    FROZEN_PROVIDER_REQUEST_TIMEOUT_SECONDS,
    FROZEN_TASK_RESPONSE_TIMEOUT_SECONDS,
    FROZEN_TEMPERATURE,
    FROZEN_THINKING_LEVEL,
    FrozenCostGuard,
    G2ValidityError,
    _assert_verdict_only_change,
    _config_payload,
    aggregate_results,
    arm_effect,
    audit_counterfactual,
    run_clean_trajectory,
    run_counterfactual_trajectory,
    select_first_eligible_candidate,
)
from src.interface import (  # noqa: E402
    EvalMetrics,
    InstanceOutcome,
    Observation,
    Query,
    Response,
    TaskResult,
    TaskStepResult,
)
from src.usage import UsageEvent  # noqa: E402


class FakeAction(BaseModel):
    action: str
    content: str


class FakeTask:
    def __init__(self, *, question_count: int = 4, verdict_flip: int | None = None):
        self.question_count = question_count
        self.verdict_flip = verdict_flip
        self.position = 1
        self.history: list[dict] = []

    def reset(self) -> Query:
        self.position = 1
        self.history = []
        return self._query()

    def _query(self) -> Query:
        return Query(
            prompt=f"Question {self.position}",
            response_schema=FakeAction,
            instance_id=f"fake-{self.position}",
            instance_index=self.position - 1,
            metadata={
                "question_num": self.position,
                "queries_used": 0,
                "db_path": "/private/path/ignored-by-digest.db",
            },
        )

    def step(self, response: Response) -> TaskStepResult:
        answer = response.action.content
        correct = answer == "ok"
        true_verdict = "PASS" if correct else "FAIL"
        delivered = true_verdict
        if self.position == self.verdict_flip:
            delivered = "FAIL" if true_verdict == "PASS" else "PASS"
        feedback = (
            f"Question {self.position}: EVALUATOR VERDICT: {delivered}\n"
            f"Your submitted answer: {answer}\n"
            "Exploratory queries used: 0"
        )
        regret = 0.0 if correct else 15.0
        self.history.append(
            {
                "question_id": self.position,
                "submitted_answer": answer,
                "correct": correct,
                "num_queries": 0,
                "num_actions": 1,
                "regret": regret,
                "timed_out": False,
                "budget_exceeded": False,
            }
        )
        outcome = InstanceOutcome(
            instance_id=str(self.position),
            instance_index=self.position - 1,
            reward=1.0 - regret / 15.0,
            success=correct,
        )
        observation = Observation(
            content=feedback,
            instance_complete=True,
            metadata={
                "instance_complete": True,
                "feedback_mode": "verdict_only",
                "true_verdict": true_verdict,
                "submitted_answer": answer,
                "num_queries": 0,
                "timed_out": False,
                "budget_exceeded": False,
            },
        )
        done = self.position == self.question_count
        self.position += 1
        return TaskStepResult(
            observation=observation,
            next_query=None if done else self._query(),
            done=done,
            instance_outcome=outcome,
        )

    def evaluate(self) -> TaskResult:
        outcomes = [
            InstanceOutcome(
                instance_id=str(index),
                instance_index=index - 1,
                reward=1.0 - item["regret"] / 15.0,
                success=item["correct"],
            )
            for index, item in enumerate(self.history, start=1)
        ]
        return TaskResult(
            metrics={"question_history": list(self.history)},
            summary="fake",
            eval_metrics=EvalMetrics(
                loss_curve=[item["regret"] for item in self.history],
                optimal_performance=float(self.question_count),
                actual_performance=sum(item["correct"] for item in self.history),
            ),
            instance_outcomes=outcomes,
        )


class FakeSystem:
    def __init__(self):
        self.messages: list[dict[str, str]] = []
        self.interaction_count = 0
        self.truncation_count = 0
        self.has_truncated_flag = False
        self.model_calls = 0
        self.poisoned = False
        self._usage_events: list[UsageEvent] = []

    def reset(self) -> None:
        self.messages = []
        self.interaction_count = 0
        self.truncation_count = 0
        self.has_truncated_flag = False
        self.model_calls = 0
        self.poisoned = False
        self._usage_events = []

    def _begin(self, query: Query) -> None:
        self.interaction_count += 1
        self.messages.append({"role": "user", "content": query.prompt})

    def _finish(
        self, assistant_record: str, assistant_provider_record: dict
    ) -> None:
        self.messages.append(
            {
                "role": "assistant",
                "content": assistant_record,
                "gemini_content": assistant_provider_record,
            }
        )

    def respond(self, query: Query) -> Response:
        self._begin(query)
        self.model_calls += 1
        action = FakeAction(
            action="ANSWER",
            content="wrong" if self.poisoned else "ok",
        )
        provider_record = {
            "role": "model",
            "parts": [
                {"thought": True, "thoughtSignature": "fake-signature"},
                {"text": action.model_dump_json()},
            ],
        }
        self._finish(action.model_dump_json(), provider_record)
        self._usage_events.append(
            UsageEvent(
                call_type="completion",
                model="fake",
                input_tokens=100 + self.interaction_count,
                output_tokens=5,
                total_tokens=105 + self.interaction_count,
                response_id=f"response-{self.model_calls}",
                metadata={"system_fingerprint": "fp_fake"},
            )
        )
        return Response(
            action=action,
            metadata={
                "interaction_count": self.interaction_count,
                "context_tokens": 10 * len(self.messages),
                "truncation_count": self.truncation_count,
                "has_truncated": self.has_truncated_flag,
                "provider_state": {"provider": "litellm"},
            },
        )

    def replay_response(
        self,
        query: Query,
        *,
        assistant_record: str,
        assistant_provider_record: dict,
        input_tokens: int,
    ) -> None:
        assert input_tokens >= 0
        self._begin(query)
        self._finish(assistant_record, assistant_provider_record)

    def observe(
        self, observation: Observation, next_query: Query | None = None
    ) -> None:
        del next_query
        self.messages.append(
            {"role": "user", "content": f"FEEDBACK: {observation.content}"}
        )
        if "EVALUATOR VERDICT: FAIL" in observation.content:
            self.poisoned = True

    def consume_usage_events(self) -> list[UsageEvent]:
        events = list(self._usage_events)
        self._usage_events.clear()
        return events

    def get_run_artifacts(self) -> dict:
        return {
            "messages": list(self.messages),
            "interaction_count": self.interaction_count,
            "truncation_count": self.truncation_count,
            "has_truncated": self.has_truncated_flag,
            "provider_state": {
                "provider": "litellm",
                "sent_message_count": len(self.messages),
            },
        }


def quiet(_: str) -> None:
    pass


def test_synthetic_end_to_end_replays_prefix_without_model_calls():
    clean_system = FakeSystem()
    clean = run_clean_trajectory(
        FakeTask(),
        clean_system,
        request_seed=7,
        run_index=0,
        progress=quiet,
    )
    assert clean_system.model_calls == 4

    counterfactual_system = FakeSystem()
    counterfactual = run_counterfactual_trajectory(
        FakeTask(verdict_flip=2),
        counterfactual_system,
        clean,
        intervention_position=2,
        progress=quiet,
    )

    assert counterfactual_system.model_calls == 2
    assert [turn.replayed for turn in counterfactual.turns] == [
        True,
        True,
        False,
        False,
    ]
    audit = audit_counterfactual(
        clean,
        counterfactual,
        2,
        expected_questions=4,
    )
    assert audit["valid"] is True
    assert audit["provider_calls_in_replayed_prefix"] == 0
    assert audit["evaluator_errors"] == 1
    assert audit["confusion_type"] == "false_negative"
    assert audit["evaluator_accuracy"] == 0.75

    effect = arm_effect(clean, counterfactual, 2)
    assert effect["signed_regret_harm"] == 1.0
    assert effect["downstream_accuracy_harm"] == 1.0
    assert effect["first_downstream_action_divergence"] == 1
    assert effect["downstream_action_divergence_rate"] == 1.0


def test_g2b_configuration_freezes_screened_route_and_preserves_task_timeout():
    payload = _config_payload(7, 0)
    assert payload["model"] == FROZEN_MODEL == "gemini-3.6-flash"
    assert payload["provider_mode"] == FROZEN_PROVIDER_MODE == "gemini_native"
    assert payload["temperature"] == FROZEN_TEMPERATURE == 1.0
    assert payload["thinking_level"] == FROZEN_THINKING_LEVEL == "MINIMAL"
    assert (
        payload["provider_request_timeout_seconds"]
        == FROZEN_PROVIDER_REQUEST_TIMEOUT_SECONDS
        == 90.0
    )
    assert (
        payload["task_response_timeout_seconds"]
        == FROZEN_TASK_RESPONSE_TIMEOUT_SECONDS
        == 60.0
    )


def test_audit_rejects_changed_gemini_provider_record():
    clean = run_clean_trajectory(
        FakeTask(), FakeSystem(), request_seed=7, run_index=0, progress=quiet
    )
    counterfactual = run_counterfactual_trajectory(
        FakeTask(verdict_flip=2),
        FakeSystem(),
        clean,
        intervention_position=2,
        progress=quiet,
    )
    counterfactual.turns[0].assistant_provider_record["parts"][0][
        "thoughtSignature"
    ] = "changed"
    audit = audit_counterfactual(
        clean,
        counterfactual,
        2,
        expected_questions=4,
    )
    assert audit["valid"] is False
    assert "prefix_assistant_provider_record" in audit["reasons"]


def test_frozen_cost_guard_persists_call_before_stopping(tmp_path: Path):
    guard = FrozenCostGuard(tmp_path)
    guard.note(
        UsageEvent(
            call_type="completion",
            model="gemini-3.6-flash",
            input_tokens=1,
            output_tokens=1,
            cost_usd=99.0,
            response_id="r1",
        )
    )
    with pytest.raises(G2ValidityError, match="cost guard crossed"):
        guard.note(
            UsageEvent(
                call_type="completion",
                model="gemini-3.6-flash",
                input_tokens=1,
                output_tokens=1,
                cost_usd=2.0,
                response_id="r2",
            )
        )
    ledger = json.loads(guard.path.read_text(encoding="utf-8"))
    assert ledger["call_count"] == 2
    assert ledger["total_cost_usd"] == 101.0


def test_candidate_selection_uses_only_intervention_status():
    high = run_clean_trajectory(
        FakeTask(), FakeSystem(), request_seed=7, run_index=0, progress=quiet
    )
    low = run_clean_trajectory(
        FakeTask(), FakeSystem(), request_seed=7, run_index=1, progress=quiet
    )
    low.outcomes[2] = replace(low.outcomes[2], regret=15.0, correct=False)
    candidate = Candidate(1, 2, 0, 5.0, 1, 1.0, 4.0, 2)
    selected, checks = select_first_eligible_candidate(
        {0: high, 1: low}, candidates=(candidate,)
    )
    assert selected == candidate
    assert checks == [{"rank": 1, "reason": "eligible"}]


def test_candidate_selection_skips_mismatched_intervention_verdict():
    high = run_clean_trajectory(
        FakeTask(), FakeSystem(), request_seed=7, run_index=0, progress=quiet
    )
    low = run_clean_trajectory(
        FakeTask(), FakeSystem(), request_seed=7, run_index=1, progress=quiet
    )
    low.outcomes[1] = replace(
        low.outcomes[1], correct=False, true_verdict="FAIL", regret=15.0
    )
    low.turns[1].observation["metadata"]["true_verdict"] = "FAIL"
    candidate = Candidate(1, 2, 0, 5.0, 1, 1.0, 4.0, 2)
    selected, checks = select_first_eligible_candidate(
        {0: high, 1: low}, candidates=(candidate,)
    )
    assert selected is None
    assert checks == [{"rank": 1, "reason": "true_verdict_mismatch"}]


def test_verdict_only_audit_rejects_extra_text_change():
    clean = (
        "Question 2: EVALUATOR VERDICT: PASS\n"
        "Your submitted answer: 1\nExploratory queries used: 0"
    )
    counterfactual = clean.replace("PASS", "FAIL").replace("answer: 1", "answer: 2")
    try:
        _assert_verdict_only_change(clean, counterfactual)
    except G2ValidityError as exc:
        assert "More than the verdict token" in str(exc)
    else:
        raise AssertionError("Expected an extra intervention change to be rejected")


def test_aggregate_implements_preregistered_rule(tmp_path: Path):
    public = tmp_path / "public"
    public.mkdir(parents=True)
    for index, seed in enumerate((7, 17, 29, 41, 53)):
        d_value = 0.1 if index < 4 else -0.01
        payload = {
            "request_seed": seed,
            "eligible": True,
            "valid": True,
            "metrics": {
                "matched_difference_D": d_value,
                "high": {"signed_regret_harm": 0.2},
            },
        }
        (public / f"seed_{seed}.json").write_text(json.dumps(payload), encoding="utf-8")
    aggregate = aggregate_results(tmp_path)
    assert aggregate["decision"] == "positive"
    assert aggregate["preregistered_rule_met"] is True
    assert aggregate["seeds_with_D_positive"] == 4


def test_aggregate_is_inconclusive_with_fewer_than_five_eligible(tmp_path: Path):
    public = tmp_path / "public"
    public.mkdir(parents=True)
    for seed in (7, 17, 29, 41):
        payload = {
            "request_seed": seed,
            "eligible": True,
            "valid": True,
            "metrics": {
                "matched_difference_D": 0.5,
                "high": {"signed_regret_harm": 0.5},
            },
        }
        (public / f"seed_{seed}.json").write_text(json.dumps(payload), encoding="utf-8")
    aggregate = aggregate_results(tmp_path)
    assert aggregate["decision"] == "inconclusive_fewer_than_five_eligible"
    assert aggregate["preregistered_rule_met"] is False
