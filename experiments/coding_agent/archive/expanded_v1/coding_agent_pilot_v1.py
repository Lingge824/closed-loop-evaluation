import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Tuple

from openai import OpenAI


# ============================================================
# CONFIG
# ============================================================

MODEL = os.environ.get("PILOT_MODEL", "")
BASE_URL = os.environ.get("OPENAI_BASE_URL")
API_KEY = os.environ.get("OPENAI_API_KEY", "dummy")

RESULTS_DIR = Path("results_v1")

INITIAL_SKILLS = """- Follow the current specification exactly.
- Think explicitly about edge cases before coding.
- Prefer simple, general Python implementations.
"""


# ============================================================
# TASKS
# ============================================================

@dataclass(frozen=True)
class Task:
    task_id: str
    family: str
    prompt: str
    visible_tests: str
    hidden_tests: str


TASKS: List[Task] = [
    # --------------------------------------------------------
    # FAMILY 1: order preservation
    # --------------------------------------------------------
    Task(
        "dedupe_1",
        "order_preserving",
        """Implement `dedupe(items)` in solution.py.
Return a list with duplicates removed while preserving the first occurrence order.
Do not mutate the input.""",
        r"""
from solution import dedupe
assert dedupe([1,1,2,2,3]) == [1,2,3]
assert dedupe(["a","b","a"]) == ["a","b"]
""",
        r"""
from solution import dedupe
assert dedupe([]) == []
assert dedupe([3,2,3,1,2]) == [3,2,1]
x=[1,2,1]
before=list(x)
assert dedupe(x)==[1,2]
assert x==before
"""
    ),
    Task(
        "first_unique_1",
        "order_preserving",
        """Implement `first_unique(items)` in solution.py.
Return the first element that occurs exactly once.
Return None if no such element exists.""",
        r"""
from solution import first_unique
assert first_unique([1,2,1,3]) == 2
assert first_unique(["a","a","b"]) == "b"
""",
        r"""
from solution import first_unique
assert first_unique([]) is None
assert first_unique([1,1,2,2]) is None
assert first_unique(["x","y","x","z","y"]) == "z"
"""
    ),
    Task(
        "stable_intersection_1",
        "order_preserving",
        """Implement `stable_intersection(a, b)` in solution.py.
Return distinct values occurring in both sequences, ordered by first occurrence in `a`.""",
        r"""
from solution import stable_intersection
assert stable_intersection([1,2,3],[2,3,4]) == [2,3]
assert stable_intersection(["b","a","b"],["a","b"]) == ["b","a"]
""",
        r"""
from solution import stable_intersection
assert stable_intersection([], [1]) == []
assert stable_intersection([3,1,3,2],[2,3]) == [3,2]
"""
    ),

    # --------------------------------------------------------
    # FAMILY 2: boundary semantics
    # --------------------------------------------------------
    Task(
        "chunks_1",
        "boundaries",
        """Implement `chunks(seq, size)` in solution.py.
Split into consecutive chunks of length size.
Final chunk may be shorter.
Raise ValueError if size <= 0.
Return a list of lists.""",
        r"""
from solution import chunks
assert chunks([1,2,3,4],2) == [[1,2],[3,4]]
assert chunks([1,2,3],2) == [[1,2],[3]]
""",
        r"""
from solution import chunks
assert chunks([],3) == []
assert chunks([1],5) == [[1]]
for bad in [0,-1]:
    try:
        chunks([1,2],bad)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
"""
    ),
    Task(
        "sliding_windows_1",
        "boundaries",
        """Implement `sliding_windows(seq, width)` in solution.py.
Return every contiguous window of exactly width elements as a list of lists.
Raise ValueError if width <= 0.
If width > len(seq), return [].""",
        r"""
from solution import sliding_windows
assert sliding_windows([1,2,3,4],2) == [[1,2],[2,3],[3,4]]
assert sliding_windows([1,2,3],3) == [[1,2,3]]
""",
        r"""
from solution import sliding_windows
assert sliding_windows([],1) == []
assert sliding_windows([1,2],3) == []
for bad in [0,-2]:
    try:
        sliding_windows([1],bad)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
"""
    ),
    Task(
        "clamp_slice_1",
        "boundaries",
        """Implement `clamp_slice(seq, start, stop)` in solution.py.
Interpret [start, stop) as half-open, clamp endpoints into [0, len(seq)].
If clamped stop < clamped start, return [].
Return a list.""",
        r"""
from solution import clamp_slice
assert clamp_slice([0,1,2,3],1,3) == [1,2]
assert clamp_slice([0,1,2],-5,2) == [0,1]
""",
        r"""
from solution import clamp_slice
assert clamp_slice([0,1,2],2,99) == [2]
assert clamp_slice([0,1,2],3,1) == []
assert clamp_slice([],-1,9) == []
"""
    ),

    # --------------------------------------------------------
    # FAMILY 3: numeric semantics
    # --------------------------------------------------------
    Task(
        "trunc_div_1",
        "numeric_semantics",
        """Implement `trunc_div(a, b)` in solution.py for integers.
Return integer division truncated toward zero, not floor division.
Raise ZeroDivisionError for b == 0.
Do not convert through floating point.""",
        r"""
from solution import trunc_div
assert trunc_div(7,3) == 2
assert trunc_div(8,-3) == -2
""",
        r"""
from solution import trunc_div
assert trunc_div(-8,3) == -2
assert trunc_div(-8,-3) == 2
assert trunc_div(1,2) == 0
assert trunc_div(10**40+1,3) == (10**40+1)//3
try:
    trunc_div(1,0)
    raise AssertionError("expected ZeroDivisionError")
except ZeroDivisionError:
    pass
"""
    ),
    Task(
        "round_half_away_1",
        "numeric_semantics",
        """Implement `round_half_away(x)` in solution.py.
For a finite Python float, round to nearest integer.
Exact halves round away from zero: 1.5 -> 2, -1.5 -> -2.
Return int.""",
        r"""
from solution import round_half_away
assert round_half_away(1.2) == 1
assert round_half_away(1.5) == 2
assert round_half_away(2.6) == 3
""",
        r"""
from solution import round_half_away
assert round_half_away(-1.2) == -1
assert round_half_away(-1.5) == -2
assert round_half_away(-2.6) == -3
assert round_half_away(0.5) == 1
assert round_half_away(-0.5) == -1
"""
    ),
    Task(
        "signed_remainder_1",
        "numeric_semantics",
        """Implement `signed_remainder(a, b)` in solution.py for integers.
Use truncated-toward-zero division semantics:
a = q*b + remainder, where q is trunc_div(a,b).
Remainder has same sign as a or is zero.
Raise ZeroDivisionError if b == 0.""",
        r"""
from solution import signed_remainder
assert signed_remainder(8,3) == 2
assert signed_remainder(8,-3) == 2
""",
        r"""
from solution import signed_remainder
assert signed_remainder(-8,3) == -2
assert signed_remainder(-8,-3) == -2
assert signed_remainder(1,2) == 1
try:
    signed_remainder(1,0)
    raise AssertionError("expected ZeroDivisionError")
except ZeroDivisionError:
    pass
"""
    ),

    # --------------------------------------------------------
    # FAMILY 4: text normalization
    # --------------------------------------------------------
    Task(
        "normalize_spaces_1",
        "text_normalization",
        """Implement `normalize_spaces(text)` in solution.py.
Strip leading/trailing whitespace and collapse each maximal run of Unicode whitespace
inside the string to one ASCII space.""",
        r"""
from solution import normalize_spaces
assert normalize_spaces("  hello   world  ") == "hello world"
assert normalize_spaces("a b") == "a b"
""",
        r"""
from solution import normalize_spaces
assert normalize_spaces("") == ""
assert normalize_spaces("\talpha\nbeta\r\n gamma") == "alpha beta gamma"
assert normalize_spaces("a\u00a0\u00a0b") == "a b"
"""
    ),
    Task(
        "split_once_1",
        "text_normalization",
        """Implement `split_once(text, sep)` in solution.py.
Split only on the first occurrence of sep and return (left, right).
If sep absent, return (text, "").
Raise ValueError if sep == "".""",
        r"""
from solution import split_once
assert split_once("a:b:c",":") == ("a","b:c")
assert split_once("abc",":") == ("abc","")
""",
        r"""
from solution import split_once
assert split_once(":abc",":") == ("","abc")
assert split_once("abc:",":") == ("abc","")
assert split_once("a--b--c","--") == ("a","b--c")
try:
    split_once("abc","")
    raise AssertionError("expected ValueError")
except ValueError:
    pass
"""
    ),
    Task(
        "canonical_key_1",
        "text_normalization",
        """Implement `canonical_key(text)` in solution.py.
1) strip leading/trailing whitespace,
2) case-fold using str.casefold(),
3) collapse internal whitespace runs to one ASCII space.""",
        r"""
from solution import canonical_key
assert canonical_key("  Hello   WORLD ") == "hello world"
assert canonical_key("ABC") == "abc"
""",
        r"""
from solution import canonical_key
assert canonical_key("Straße") == "strasse"
assert canonical_key("\tA\nB") == "a b"
assert canonical_key("") == ""
"""
    ),
]


# ============================================================
# API
# ============================================================

def client():
    kwargs = {}
    if BASE_URL:
        kwargs["base_url"] = BASE_URL

    return OpenAI(
        api_key=API_KEY,
        **kwargs,
    )


def chat(messages):
    if not MODEL:
        raise RuntimeError(
            "Set PILOT_MODEL before running."
        )

    response = client().chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.0,
    )

    return response.choices[0].message.content


# ============================================================
# AGENT + MEMORY
# ============================================================

SOLVER_SYSTEM = """You are a coding agent in a longitudinal learning experiment.

You solve exactly one Python task at a time.

A persistent skills memory is provided. It contains lessons inferred from previous tasks.
Use it as potentially useful advice, but the current task specification is authoritative.

Return ONLY the complete contents of solution.py inside one Python code block.
"""


MEMORY_SYSTEM = """You maintain a compact persistent coding playbook.

Inputs:
- previous skills.md,
- task specification,
- submitted solution,
- proxy evaluator verdict: PASS or FAIL.

You NEVER see hidden-test results.

Update the playbook using only evidence available from those inputs.

Rules:
1. At most 12 bullet points.
2. Prefer reusable operational lessons.
3. PASS is evidence that the approach may be useful.
4. FAIL is evidence to avoid or revise the approach.
5. Never claim hidden-test knowledge.
6. Preserve useful older lessons unless contradicted.
7. Output markdown bullet points only.
"""


def extract_python(text):
    m = re.search(
        r"```python\s*(.*?)```",
        text,
        flags=re.S,
    )

    if m:
        return m.group(1).strip() + "\n"

    m = re.search(
        r"```\s*(.*?)```",
        text,
        flags=re.S,
    )

    if m:
        return m.group(1).strip() + "\n"

    return text.strip() + "\n"


def solve(task, skills):
    prompt = f"""PERSISTENT SKILLS MEMORY:

{skills}

CURRENT TASK:

{task.prompt}
"""

    text = chat([
        {
            "role": "system",
            "content": SOLVER_SYSTEM,
        },
        {
            "role": "user",
            "content": prompt,
        },
    ])

    return extract_python(text)


def update_memory(
    old_skills,
    task,
    solution,
    verdict,
):
    prompt = f"""OLD SKILLS:

{old_skills}

TASK:

{task.prompt}

SUBMITTED SOLUTION:

```python
{solution}
```

PROXY EVALUATOR VERDICT:

{verdict}

Return updated skills.md.
"""

    return chat([
        {
            "role": "system",
            "content": MEMORY_SYSTEM,
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]).strip()


# ============================================================
# SANDBOX TESTING
# ============================================================

def run_tests(solution, tests, timeout=5):
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        (td / "solution.py").write_text(
            solution,
            encoding="utf-8",
        )

        (td / "test_task.py").write_text(
            tests,
            encoding="utf-8",
        )

        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    "test_task.py",
                ],
                cwd=td,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )

            return proc.returncode == 0

        except subprocess.TimeoutExpired:
            return False


# ============================================================
# TRAJECTORIES
# ============================================================

def run_clean_trajectory(
    tasks,
    name,
):
    out = RESULTS_DIR / name
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    skills = INITIAL_SKILLS
    records = []

    for t, task in enumerate(tasks):
        print(
            f"[{name}] round {t:02d}: {task.task_id}",
            flush=True,
        )

        before = skills

        solution = solve(
            task,
            before,
        )

        proxy = run_tests(
            solution,
            task.visible_tests,
        )

        oracle = run_tests(
            solution,
            task.hidden_tests,
        )

        delivered = proxy

        skills = update_memory(
            before,
            task,
            solution,
            "PASS" if delivered else "FAIL",
        )

        rec = {
            "round": t,
            "task_id": task.task_id,
            "family": task.family,
            "proxy_pass": proxy,
            "delivered_proxy_pass": delivered,
            "oracle_pass": oracle,
            "flipped": False,
            "skills_before": before,
            "skills_after": skills,
            "solution": solution,
        }

        records.append(rec)

        (out / f"{t:02d}_{task.task_id}.json").write_text(
            json.dumps(
                rec,
                indent=2,
            ),
            encoding="utf-8",
        )

    return records


def run_counterfactual_trajectory(
    tasks,
    clean_reference,
    flip_rounds: Tuple[int, ...],
    name,
):
    """
    Strict paired counterfactual.

    Before the first intervention:
      exact clean history is copied.

    At an intervention round:
      exact clean solution, proxy result, and oracle result are reused;
      ONLY the delivered proxy verdict is flipped.

    After the first intervention:
      the branch evolves normally from its diverged persistent memory.

    For later flip rounds in a multi-flip branch:
      the branch's own generated solution is evaluated normally,
      then its delivered proxy verdict is flipped.
    """

    flip_rounds = tuple(
        sorted(
            set(
                flip_rounds
            )
        )
    )

    first_flip = min(
        flip_rounds
    )

    out = RESULTS_DIR / name
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []
    skills = INITIAL_SKILLS

    for t, task in enumerate(tasks):

        # ----------------------------------------------------
        # EXACT CLEAN PREFIX
        # ----------------------------------------------------

        if t < first_flip:
            rec = dict(
                clean_reference[t]
            )

            records.append(rec)
            skills = rec[
                "skills_after"
            ]

            print(
                f"[{name}] round {t:02d}: "
                f"{task.task_id} [REUSE CLEAN PREFIX]",
                flush=True,
            )

            (out / f"{t:02d}_{task.task_id}.json").write_text(
                json.dumps(
                    rec,
                    indent=2,
                ),
                encoding="utf-8",
            )

            continue

        print(
            f"[{name}] round {t:02d}: {task.task_id}",
            flush=True,
        )

        before = skills

        # ----------------------------------------------------
        # FIRST INTERVENTION:
        # reuse the exact clean task solution and test outcomes.
        # ----------------------------------------------------

        if t == first_flip:
            clean_rec = (
                clean_reference[t]
            )

            if before != clean_rec[
                "skills_before"
            ]:
                raise RuntimeError(
                    "Clean prefix mismatch before first intervention."
                )

            solution = (
                clean_rec[
                    "solution"
                ]
            )

            proxy = (
                clean_rec[
                    "proxy_pass"
                ]
            )

            oracle = (
                clean_rec[
                    "oracle_pass"
                ]
            )

        else:
            # After divergence, generate from branch memory.
            solution = solve(
                task,
                before,
            )

            proxy = run_tests(
                solution,
                task.visible_tests,
            )

            oracle = run_tests(
                solution,
                task.hidden_tests,
            )

        delivered = proxy
        flipped = False

        if t in flip_rounds:
            delivered = not proxy
            flipped = True

        skills = update_memory(
            before,
            task,
            solution,
            "PASS" if delivered else "FAIL",
        )

        rec = {
            "round": t,
            "task_id": task.task_id,
            "family": task.family,
            "proxy_pass": proxy,
            "delivered_proxy_pass": delivered,
            "oracle_pass": oracle,
            "flipped": flipped,
            "skills_before": before,
            "skills_after": skills,
            "solution": solution,
        }

        records.append(rec)

        (out / f"{t:02d}_{task.task_id}.json").write_text(
            json.dumps(
                rec,
                indent=2,
            ),
            encoding="utf-8",
        )

    return records


# ============================================================
# METRICS
# ============================================================

def text_distance(a, b):
    return 1.0 - SequenceMatcher(
        None,
        a,
        b,
    ).ratio()


def summarize_pair(
    clean,
    branch,
    flip_rounds,
):
    first_flip = min(
        flip_rounds
    )

    future_indices = list(
        range(
            first_flip + 1,
            len(clean),
        )
    )

    immediate_memory_distance = (
        text_distance(
            clean[first_flip][
                "skills_after"
            ],
            branch[first_flip][
                "skills_after"
            ],
        )
    )

    memory_distances = [
        text_distance(
            clean[t][
                "skills_after"
            ],
            branch[t][
                "skills_after"
            ],
        )
        for t in future_indices
    ]

    solution_changes = [
        clean[t][
            "solution"
        ]
        != branch[t][
            "solution"
        ]
        for t in future_indices
    ]

    clean_oracle = [
        int(
            clean[t][
                "oracle_pass"
            ]
        )
        for t in future_indices
    ]

    branch_oracle = [
        int(
            branch[t][
                "oracle_pass"
            ]
        )
        for t in future_indices
    ]

    clean_rate = (
        sum(clean_oracle)
        / len(clean_oracle)
        if clean_oracle
        else float("nan")
    )

    branch_rate = (
        sum(branch_oracle)
        / len(branch_oracle)
        if branch_oracle
        else float("nan")
    )

    # Local leverage proxy:
    # immediate persistent-memory change + fraction of future solutions changed.
    future_solution_change_rate = (
        sum(solution_changes)
        / len(solution_changes)
        if solution_changes
        else 0.0
    )

    leverage_score = (
        immediate_memory_distance
        + future_solution_change_rate
    )

    return {
        "flip_rounds":
            ",".join(
                str(x)
                for x in flip_rounds
            ),

        "first_flip":
            first_flip,

        "family":
            clean[first_flip][
                "family"
            ],

        "task_id":
            clean[first_flip][
                "task_id"
            ],

        "immediate_memory_distance":
            immediate_memory_distance,

        "mean_future_memory_distance":
            (
                sum(memory_distances)
                / len(memory_distances)
                if memory_distances
                else 0.0
            ),

        "final_memory_distance":
            (
                memory_distances[-1]
                if memory_distances
                else immediate_memory_distance
            ),

        "future_solution_changes":
            sum(
                int(x)
                for x in solution_changes
            ),

        "future_tasks":
            len(
                future_indices
            ),

        "future_solution_change_rate":
            future_solution_change_rate,

        "future_clean_oracle_rate":
            clean_rate,

        "future_branch_oracle_rate":
            branch_rate,

        "future_oracle_harm":
            clean_rate
            - branch_rate,

        "leverage_score":
            leverage_score,
    }


def interaction_metric(
    single_a,
    single_b,
    double_ab,
):
    """
    Positive value => super-additive harm.

    I(a,b) = H(a,b) - H(a) - H(b)
    """

    return (
        double_ab[
            "future_oracle_harm"
        ]
        -
        single_a[
            "future_oracle_harm"
        ]
        -
        single_b[
            "future_oracle_harm"
        ]
    )


# ============================================================
# CSV
# ============================================================

def write_csv(
    rows,
    path,
):
    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=[
            "smoke",
            "full",
        ],
        default="smoke",
    )

    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    if not MODEL:
        raise RuntimeError(
            "Set PILOT_MODEL, e.g. "
            "export PILOT_MODEL='your-model-id'"
        )

    tasks = list(
        TASKS
    )

    if args.max_tasks:
        tasks = tasks[
            :args.max_tasks
        ]

    RESULTS_DIR.mkdir(
        exist_ok=True,
    )

    # --------------------------------------------------------
    # SMOKE
    # --------------------------------------------------------

    if args.mode == "smoke":
        smoke_tasks = tasks[:3]

        clean = run_clean_trajectory(
            smoke_tasks,
            "smoke_clean",
        )

        print()
        print(
            "SMOKE COMPLETE"
        )

        print(
            "hidden-test passes:",
            sum(
                int(
                    x[
                        "oracle_pass"
                    ]
                )
                for x in clean
            ),
            "/",
            len(clean),
        )

        return

    # --------------------------------------------------------
    # FULL
    # --------------------------------------------------------

    clean = run_clean_trajectory(
        tasks,
        "clean",
    )

    # First task of each family.
    candidate_flips = [
        0,
        3,
        6,
        9,
    ]

    candidate_flips = [
        x
        for x in candidate_flips
        if x < len(tasks) - 1
    ]

    single_results = {}
    rows = []

    for flip in candidate_flips:
        branch = (
            run_counterfactual_trajectory(
                tasks,
                clean_reference=clean,
                flip_rounds=(
                    flip,
                ),
                name=f"flip_{flip}",
            )
        )

        metrics = summarize_pair(
            clean,
            branch,
            (
                flip,
            ),
        )

        single_results[
            flip
        ] = metrics

        row = {
            "condition":
                "single",

            **metrics,
        }

        rows.append(
            row
        )

        print()
        print(
            "PAIR RESULT"
        )

        for k, v in metrics.items():
            print(
                f"  {k}: {v}"
            )

    # --------------------------------------------------------
    # DOUBLE-FLIP INTERACTION
    # --------------------------------------------------------

    double_pairs = []

    if len(
        candidate_flips
    ) >= 2:
        for i in range(
            len(
                candidate_flips
            ) - 1
        ):
            double_pairs.append(
                (
                    candidate_flips[i],
                    candidate_flips[i + 1],
                )
            )

    interaction_rows = []

    for a, b in double_pairs:

        branch = (
            run_counterfactual_trajectory(
                tasks,
                clean_reference=clean,
                flip_rounds=(
                    a,
                    b,
                ),
                name=f"flip_{a}_{b}",
            )
        )

        metrics = summarize_pair(
            clean,
            branch,
            (
                a,
                b,
            ),
        )

        interaction = interaction_metric(
            single_results[
                a
            ],
            single_results[
                b
            ],
            metrics,
        )

        row = {
            "condition":
                "double",

            **metrics,

            "interaction":
                interaction,

            "single_harm_a":
                single_results[
                    a
                ][
                    "future_oracle_harm"
                ],

            "single_harm_b":
                single_results[
                    b
                ][
                    "future_oracle_harm"
                ],
        }

        interaction_rows.append(
            row
        )

        print()
        print(
            "DOUBLE-FLIP RESULT"
        )

        print(
            f"  flips: {a}, {b}"
        )

        print(
            "  joint harm:",
            metrics[
                "future_oracle_harm"
            ],
        )

        print(
            "  additive prediction:",
            (
                single_results[
                    a
                ][
                    "future_oracle_harm"
                ]
                +
                single_results[
                    b
                ][
                    "future_oracle_harm"
                ]
            ),
        )

        print(
            "  interaction:",
            interaction,
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    write_csv(
        rows,
        RESULTS_DIR
        / "single_flip_summary.csv",
    )

    if interaction_rows:
        write_csv(
            interaction_rows,
            RESULTS_DIR
            / "double_flip_summary.csv",
        )

    print()
    print(
        "=" * 72
    )

    print(
        "FULL PILOT COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        "Single flips:",
        RESULTS_DIR
        / "single_flip_summary.csv",
    )

    if interaction_rows:
        print(
            "Double flips:",
            RESULTS_DIR
            / "double_flip_summary.csv",
        )


if __name__ == "__main__":
    main()
