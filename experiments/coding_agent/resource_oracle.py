import json


RESOURCE_METRICS_PREFIX = "RESOURCE_METRICS="


FIRST_UNIQUE_RESOURCE_TESTS = r"""
import json

from solution import first_unique


class TrackedValue:
    equality_calls = 0
    hash_calls = 0

    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        type(self).equality_calls += 1
        return (
            isinstance(other, TrackedValue)
            and self.value == other.value
        )

    def __hash__(self):
        type(self).hash_calls += 1
        return hash(self.value)


def measured_work(size):
    items = (
        [TrackedValue(i) for i in range(size)]
        + [TrackedValue(i) for i in range(size)]
        + [TrackedValue(size)]
    )
    TrackedValue.equality_calls = 0
    TrackedValue.hash_calls = 0
    result = first_unique(items)
    assert result is not None
    assert result.value == size
    work = (
        TrackedValue.equality_calls
        + TrackedValue.hash_calls
    )
    return work, len(items)


small_work, small_items = measured_work(64)
large_work, large_items = measured_work(128)
growth_ratio = large_work / max(small_work, 1)
normalized_work = large_work / large_items
resource_pass = (
    growth_ratio <= 3.0
    and normalized_work <= 100.0
)

metrics = {
    "task_id": "first_unique_1",
    "small_work": small_work,
    "large_work": large_work,
    "growth_ratio": growth_ratio,
    "normalized_work": normalized_work,
    "resource_pass": resource_pass,
}
print(
    "RESOURCE_METRICS="
    + json.dumps(metrics, sort_keys=True),
    flush=True,
)
assert resource_pass, metrics
"""


STABLE_INTERSECTION_RESOURCE_TESTS = r"""
import json

from solution import stable_intersection


class TrackedValue:
    equality_calls = 0
    hash_calls = 0

    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        type(self).equality_calls += 1
        return (
            isinstance(other, TrackedValue)
            and self.value == other.value
        )

    def __hash__(self):
        type(self).hash_calls += 1
        return hash(self.value)


def measured_work(size):
    a = (
        [TrackedValue(i) for i in range(size)]
        + [TrackedValue(i) for i in range(size // 2)]
    )
    b = [
        TrackedValue(i)
        for i in range(size // 2, size + size // 2)
    ]
    TrackedValue.equality_calls = 0
    TrackedValue.hash_calls = 0
    result = stable_intersection(a, b)
    assert [item.value for item in result] == list(
        range(size // 2, size)
    )
    work = (
        TrackedValue.equality_calls
        + TrackedValue.hash_calls
    )
    return work, len(a) + len(b)


small_work, small_items = measured_work(64)
large_work, large_items = measured_work(128)
growth_ratio = large_work / max(small_work, 1)
normalized_work = large_work / large_items
resource_pass = (
    growth_ratio <= 3.0
    and normalized_work <= 100.0
)

metrics = {
    "task_id": "stable_intersection_1",
    "small_work": small_work,
    "large_work": large_work,
    "growth_ratio": growth_ratio,
    "normalized_work": normalized_work,
    "resource_pass": resource_pass,
}
print(
    "RESOURCE_METRICS="
    + json.dumps(metrics, sort_keys=True),
    flush=True,
)
assert resource_pass, metrics
"""


RESOURCE_TESTS = {
    "first_unique_1": FIRST_UNIQUE_RESOURCE_TESTS,
    "stable_intersection_1": STABLE_INTERSECTION_RESOURCE_TESTS,
}


def resource_tests_for(task_id):
    return RESOURCE_TESTS.get(task_id)


def parse_resource_metrics(stdout):
    for line in stdout.splitlines():
        if not line.startswith(RESOURCE_METRICS_PREFIX):
            continue

        payload = json.loads(
            line[len(RESOURCE_METRICS_PREFIX):]
        )
        required = {
            "task_id",
            "small_work",
            "large_work",
            "growth_ratio",
            "normalized_work",
            "resource_pass",
        }
        missing = required.difference(payload)
        if missing:
            raise ValueError(
                "Resource metrics missing fields: "
                + ", ".join(sorted(missing))
            )
        return payload

    return None
