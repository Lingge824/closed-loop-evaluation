import tempfile
import unittest
import os
from pathlib import Path
from unittest import mock

import pilot


SOLUTIONS = {
    "dedupe(items)": (
        "def dedupe(items):\n"
        "    return list(dict.fromkeys(items))\n"
    ),
    "first_unique(items)": (
        "def first_unique(items):\n"
        "    counts = {item: items.count(item) for item in items}\n"
        "    return next((item for item in items if counts[item] == 1), None)\n"
    ),
    "stable_intersection(a, b)": (
        "def stable_intersection(a, b):\n"
        "    result = []\n"
        "    for item in a:\n"
        "        if item in b and item not in result:\n"
        "            result.append(item)\n"
        "    return result\n"
    ),
}


def fake_chat(messages, seed):
    del seed
    user = messages[-1]["content"]
    if "PROXY EVALUATOR VERDICT:" in user:
        verdict = user.split(
            "PROXY EVALUATOR VERDICT:",
            1,
        )[1].strip().splitlines()[0]
        return f"- persistent verdict lesson: {verdict}"
    for marker, solution in SOLUTIONS.items():
        if marker in user:
            return f"```python\n{solution}```"
    raise AssertionError(
        f"Unexpected prompt: {user[:160]}"
    )


class PilotTests(unittest.TestCase):
    def test_groq_endpoint_prefers_groq_key(self):
        with mock.patch.object(
            pilot,
            "BASE_URL",
            "https://api.groq.com/openai/v1",
        ), mock.patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "wrong-provider-key",
                "GROQ_API_KEY": "correct-groq-key",
            },
            clear=False,
        ):
            self.assertEqual(
                pilot.resolve_api_key(),
                "correct-groq-key",
            )

    def test_extract_python_removes_thinking_prefix(self):
        response = (
            "<think>private reasoning</think>\n"
            "```python\ndef f():\n    return 1\n```"
        )
        self.assertEqual(
            pilot.extract_python(response),
            "def f():\n    return 1\n",
        )

    def test_extract_memory_removes_reasoning_and_enforces_bullets(self):
        response = (
            "<think>private reasoning with misleading bullets:\n"
            "- do not retain this\n</think>\n"
            "- Keep the first reusable lesson.\n"
            "* Keep the second reusable lesson.\n"
            "Unstructured prose must be discarded.\n"
        )
        self.assertEqual(
            pilot.extract_memory(response),
            "- Keep the first reusable lesson.\n"
            "- Keep the second reusable lesson.\n",
        )

    def test_extract_memory_caps_and_rejects_invalid_output(self):
        response = "\n".join(
            f"- lesson {index}"
            for index in range(15)
        )
        memory = pilot.extract_memory(response)
        self.assertEqual(
            len(memory.strip().splitlines()),
            12,
        )
        self.assertNotIn(
            "lesson 12",
            memory,
        )
        with self.assertRaisesRegex(
            ValueError,
            "no valid bullet points",
        ):
            pilot.extract_memory(
                "<think>reasoning only</think>"
            )
        with self.assertRaisesRegex(
            ValueError,
            "no valid bullet points",
        ):
            pilot.extract_memory(
                "<think>unfinished reasoning\n- fake lesson"
            )

    def test_test_runner_preserves_failure_diagnostics(self):
        result = pilot.run_tests_detailed(
            "def f():\n    return 1\n",
            "from solution import f\nassert f() == 2\n",
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.returncode, 1)
        self.assertIn(
            "AssertionError",
            result.stderr,
        )
        self.assertFalse(result.timed_out)

    def test_exact_prefix_and_intervention_solution_are_reused(self):
        tasks = pilot.TASKS[:3]
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.object(
                pilot,
                "RESULTS_DIR",
                Path(temp_dir),
            ):
                clean = pilot.run_clean_trajectory(
                    tasks,
                    "clean",
                    base_seed=101,
                    chat_fn=fake_chat,
                )
                branch = pilot.run_counterfactual_trajectory(
                    tasks,
                    clean_reference=clean,
                    flip_rounds=(1,),
                    name="flip_1",
                    base_seed=101,
                    chat_fn=fake_chat,
                )

        self.assertEqual(
            clean[0],
            branch[0],
        )
        fixed_keys = [
            "skills_before",
            "solution",
            "proxy_pass",
            "oracle_pass",
            "proxy_test",
            "oracle_test",
            "solve_seed",
            "memory_seed",
        ]
        for key in fixed_keys:
            self.assertEqual(
                clean[1][key],
                branch[1][key],
                key,
            )
        self.assertNotEqual(
            clean[1]["delivered_proxy_pass"],
            branch[1]["delivered_proxy_pass"],
        )
        self.assertNotEqual(
            clean[1]["skills_after"],
            branch[1]["skills_after"],
        )

    def test_flip_must_leave_a_downstream_task(self):
        tasks = pilot.TASKS[:2]
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.object(
                pilot,
                "RESULTS_DIR",
                Path(temp_dir),
            ):
                clean = pilot.run_clean_trajectory(
                    tasks,
                    "clean",
                    chat_fn=fake_chat,
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "downstream task",
                ):
                    pilot.run_counterfactual_trajectory(
                        tasks,
                        clean_reference=clean,
                        flip_rounds=(1,),
                        name="flip_1",
                        chat_fn=fake_chat,
                    )

    def test_interaction_components_can_share_one_outcome_horizon(self):
        def records(outcomes, prefix):
            return [
                {
                    "oracle_pass": outcome,
                    "skills_after": f"{prefix}-{index}",
                    "solution": f"{prefix}-{index}",
                    "family": "test",
                    "task_id": f"t{index}",
                }
                for index, outcome in enumerate(outcomes)
            ]

        clean = records([1, 1, 1, 1], "clean")
        branch_a = records([1, 0, 1, 0], "a")
        branch_b = records([1, 1, 0, 0], "b")
        branch_ab = records([1, 0, 0, 0], "ab")
        common_start = 3
        metric_a = pilot.summarize_pair(
            clean,
            branch_a,
            (0,),
            outcome_start=common_start,
        )
        metric_b = pilot.summarize_pair(
            clean,
            branch_b,
            (2,),
            outcome_start=common_start,
        )
        metric_ab = pilot.summarize_pair(
            clean,
            branch_ab,
            (0, 2),
            outcome_start=common_start,
        )
        self.assertEqual(
            metric_a["outcome_start"],
            metric_b["outcome_start"],
        )
        self.assertEqual(
            metric_b["outcome_start"],
            metric_ab["outcome_start"],
        )


if __name__ == "__main__":
    unittest.main()
