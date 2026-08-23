import unittest

import pilot


FAST_FIRST_UNIQUE = """from collections import Counter

def first_unique(items):
    counts = Counter(items)
    for item in items:
        if counts[item] == 1:
            return item
    return None
"""


SLOW_FIRST_UNIQUE = """def first_unique(items):
    for item in items:
        if items.count(item) == 1:
            return item
    return None
"""


FAST_STABLE_INTERSECTION = """def stable_intersection(a, b):
    b_set = set(b)
    seen = set()
    result = []
    for item in a:
        if item in b_set and item not in seen:
            seen.add(item)
            result.append(item)
    return result
"""


SLOW_STABLE_INTERSECTION = """def stable_intersection(a, b):
    result = []
    for item in a:
        if item in b and item not in result:
            result.append(item)
    return result
"""


class ResourceOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = {
            task.task_id: task
            for task in pilot.G1_1_TASKS
        }

    def test_resource_contract_is_explicit_in_prompts(self):
        first_prompt = self.tasks[
            "first_unique_1"
        ].prompt
        intersection_prompt = self.tasks[
            "stable_intersection_1"
        ].prompt

        self.assertIn(
            "All elements are hashable",
            first_prompt,
        )
        self.assertIn(
            "expected O(n) time",
            first_prompt,
        )
        self.assertIn(
            "All elements are hashable",
            intersection_prompt,
        )
        self.assertIn(
            "expected O(len(a) + len(b)) time",
            intersection_prompt,
        )

    def test_exploratory_g1_tasks_remain_unchanged(self):
        original = {
            task.task_id: task
            for task in pilot.TASKS[:3]
        }
        self.assertIsNone(
            original["first_unique_1"].resource_tests
        )
        self.assertIsNone(
            original["stable_intersection_1"].resource_tests
        )
        self.assertNotIn(
            "expected O(n) time",
            original["first_unique_1"].prompt,
        )

    def assert_oracle_separates(
        self,
        task_id,
        fast_solution,
        slow_solution,
    ):
        task = self.tasks[task_id]
        fast = pilot.evaluate_oracle(
            fast_solution,
            task,
        )
        repeated_fast = pilot.evaluate_oracle(
            fast_solution,
            task,
        )
        slow = pilot.evaluate_oracle(
            slow_solution,
            task,
        )

        self.assertTrue(
            fast.correctness_test.passed
        )
        self.assertTrue(
            fast.resource_test.passed
        )
        self.assertTrue(fast.passed)
        self.assertEqual(
            fast.resource_metrics,
            repeated_fast.resource_metrics,
        )

        self.assertTrue(
            slow.correctness_test.passed
        )
        self.assertFalse(
            slow.resource_test.passed
        )
        self.assertFalse(slow.passed)
        self.assertGreater(
            slow.resource_metrics[
                "growth_ratio"
            ],
            3.0,
        )
        self.assertFalse(
            slow.resource_metrics[
                "resource_pass"
            ]
        )

    def test_first_unique_oracle_separates_linear_and_quadratic(self):
        self.assert_oracle_separates(
            "first_unique_1",
            FAST_FIRST_UNIQUE,
            SLOW_FIRST_UNIQUE,
        )

    def test_intersection_oracle_separates_linear_and_quadratic(self):
        self.assert_oracle_separates(
            "stable_intersection_1",
            FAST_STABLE_INTERSECTION,
            SLOW_STABLE_INTERSECTION,
        )

    def test_non_resource_task_uses_correctness_only(self):
        task = self.tasks["dedupe_1"]
        result = pilot.evaluate_oracle(
            "def dedupe(items):\n"
            "    return list(dict.fromkeys(items))\n",
            task,
        )
        self.assertTrue(result.passed)
        self.assertIsNone(result.resource_test)
        self.assertIsNone(result.resource_metrics)

    def test_pair_summary_reports_resource_harm_and_cost_excess(self):
        def record(
            index,
            oracle_pass,
            resource_pass,
            normalized_work,
            prefix,
        ):
            metrics = (
                {
                    "normalized_work": normalized_work,
                }
                if normalized_work is not None
                else None
            )
            return {
                "oracle_pass": oracle_pass,
                "resource_pass": resource_pass,
                "resource_metrics": metrics,
                "skills_after": f"{prefix}-memory-{index}",
                "solution": f"{prefix}-solution-{index}",
                "family": "order_preserving",
                "task_id": f"task-{index}",
            }

        clean = [
            record(0, True, None, None, "clean"),
            record(1, True, True, 4.0, "clean"),
            record(2, True, True, 2.0, "clean"),
        ]
        branch = [
            record(0, True, None, None, "branch"),
            record(1, False, False, 20.0, "branch"),
            record(2, True, True, 4.0, "branch"),
        ]

        summary = pilot.summarize_pair(
            clean,
            branch,
            (0,),
        )

        self.assertEqual(
            summary["future_resource_tasks"],
            2,
        )
        self.assertEqual(
            summary["future_clean_resource_pass_rate"],
            1.0,
        )
        self.assertEqual(
            summary["future_branch_resource_pass_rate"],
            0.5,
        )
        self.assertEqual(
            summary["future_resource_harm"],
            0.5,
        )
        self.assertEqual(
            summary[
                "mean_paired_normalized_operation_excess"
            ],
            9.0,
        )


if __name__ == "__main__":
    unittest.main()
