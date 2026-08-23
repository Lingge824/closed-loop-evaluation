from dataclasses import replace
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from pilot_v1 import (
    Config,
    success_probs,
    reference_performance,
    make_random_stream,
    sample_task_families,
    estimate_family_success,
    update_curriculum,
    update_capability,
    sqrt_js,
)


EPS = 1e-12
BASE_CFG = Config()


# ============================================================
# GENERIC TRAJECTORY RUNNER
# ============================================================

def run_custom_trajectory(
    cfg,
    stream,
    error_mask=None,
    impulses=None,
):
    """
    Run the same V1 adaptive system.

    error_mask:
        bool array [T, N].
        True means evaluator verdict is flipped.

    impulses:
        list/set of (round, within_batch_index).
        Used for one or several isolated evaluator mistakes.
    """

    difficulty = np.linspace(
        0.25,
        0.80,
        cfg.K,
    )

    capability = np.linspace(
        0.40,
        0.52,
        cfg.K,
    )

    q = np.ones(cfg.K) / cfg.K

    q_hist = [q.copy()]
    c_hist = [capability.copy()]

    perf_hist = [
        reference_performance(
            capability,
            difficulty,
            cfg.tau,
        )
    ]

    accuracy_hist = []

    if impulses is None:
        impulses = set()
    else:
        impulses = set(impulses)

    for t in range(cfg.T):

        # ----------------------------------------------------
        # Sample current tasks
        # ----------------------------------------------------

        task_ids = sample_task_families(
            q,
            stream["task"][t],
        )

        p_family = success_probs(
            capability,
            difficulty,
            cfg.tau,
        )

        p_tasks = p_family[
            task_ids
        ]

        # ----------------------------------------------------
        # True outcomes
        # ----------------------------------------------------

        y_true = (
            stream["outcome"][t]
            < p_tasks
        ).astype(int)

        y_hat = y_true.copy()

        # ----------------------------------------------------
        # Persistent corruption mask
        # ----------------------------------------------------

        if error_mask is not None:

            mask = error_mask[t]

            y_hat[mask] = (
                1 - y_hat[mask]
            )

        # ----------------------------------------------------
        # Isolated impulse errors
        # ----------------------------------------------------

        for round_idx, item_idx in impulses:

            if t == round_idx:

                y_hat[item_idx] = (
                    1 - y_hat[item_idx]
                )

        accuracy_hist.append(
            np.mean(
                y_hat == y_true
            )
        )

        # ----------------------------------------------------
        # Closed-loop update
        # ----------------------------------------------------

        p_hat = (
            estimate_family_success(
                task_ids,
                y_hat,
                cfg.K,
            )
        )

        q_new = (
            update_curriculum(
                q,
                p_hat,
                cfg,
            )
        )

        capability_new = (
            update_capability(
                capability,
                q_new,
                difficulty,
                cfg,
            )
        )

        q = q_new
        capability = capability_new

        q_hist.append(
            q.copy()
        )

        c_hist.append(
            capability.copy()
        )

        perf_hist.append(
            reference_performance(
                capability,
                difficulty,
                cfg.tau,
            )
        )

    return {
        "q": np.asarray(q_hist),
        "c": np.asarray(c_hist),
        "perf": np.asarray(perf_hist),
        "accuracy": np.asarray(accuracy_hist),
    }


# ============================================================
# ERROR MASKS
# ============================================================

def make_iid_error_mask(
    cfg,
    seed,
    error_rate,
):
    """
    EXACT total error count.

    Errors are distributed uniformly over all T*N evaluator
    decisions.
    """

    rng = np.random.default_rng(
        seed
    )

    total_items = (
        cfg.T * cfg.N
    )

    n_errors = int(
        round(
            error_rate
            * total_items
        )
    )

    priority = rng.random(
        total_items
    )

    chosen = np.argsort(
        priority
    )[:n_errors]

    mask = np.zeros(
        total_items,
        dtype=bool,
    )

    mask[
        chosen
    ] = True

    return mask.reshape(
        cfg.T,
        cfg.N,
    )


def make_bursty_error_mask(
    cfg,
    seed,
    error_rate,
    active_fraction,
    n_bursts=4,
):
    """
    Same EXACT number of errors as IID.

    But errors are concentrated into several persistent temporal
    bursts.

    Example:

        overall error rate = 10%
        active_fraction = 25%

    => inside burst windows evaluator is wrong about 40% of time.

    Bursts are spread through the trajectory so the comparison
    isn't merely "all errors happened early".
    """

    rng = np.random.default_rng(
        seed
    )

    total_items = (
        cfg.T * cfg.N
    )

    n_errors = int(
        round(
            error_rate
            * total_items
        )
    )

    active_round_count = max(
        1,
        int(
            round(
                cfg.T
                * active_fraction
            )
        ),
    )

    if (
        active_round_count
        * cfg.N
        < n_errors
    ):
        raise ValueError(
            "Burst windows do not contain enough positions "
            "for requested error count."
        )

    # --------------------------------------------------------
    # Spread multiple burst windows across the full horizon.
    # --------------------------------------------------------

    rounds_per_segment = (
        cfg.T / n_bursts
    )

    burst_len = max(
        1,
        active_round_count
        // n_bursts,
    )

    active_rounds = []

    for b in range(
        n_bursts
    ):

        center = int(
            (
                b + 0.5
            )
            * rounds_per_segment
        )

        start = max(
            0,
            center
            - burst_len // 2,
        )

        end = min(
            cfg.T,
            start + burst_len,
        )

        # Adjust if clipping occurred.
        start = max(
            0,
            end - burst_len,
        )

        active_rounds.extend(
            range(
                start,
                end,
            )
        )

    active_rounds = np.asarray(
        sorted(
            set(
                active_rounds
            )
        ),
        dtype=int,
    )

    # If integer rounding gave too few rounds, add nearby rounds.
    if len(active_rounds) < active_round_count:

        remaining = [
            t
            for t in range(cfg.T)
            if t not in set(
                active_rounds.tolist()
            )
        ]

        extra = remaining[
            :(
                active_round_count
                - len(active_rounds)
            )
        ]

        active_rounds = np.asarray(
            sorted(
                list(
                    active_rounds
                )
                + extra
            ),
            dtype=int,
        )

    candidate_positions = []

    for t in active_rounds:

        for i in range(
            cfg.N
        ):

            candidate_positions.append(
                (
                    t,
                    i,
                )
            )

    candidate_positions = (
        np.asarray(
            candidate_positions,
            dtype=int,
        )
    )

    priority = rng.random(
        len(
            candidate_positions
        )
    )

    selected = candidate_positions[
        np.argsort(
            priority
        )[:n_errors]
    ]

    mask = np.zeros(
        (
            cfg.T,
            cfg.N,
        ),
        dtype=bool,
    )

    mask[
        selected[:, 0],
        selected[:, 1],
    ] = True

    return mask


# ============================================================
# DEVIATION CURVES
# ============================================================

def deviation_curves(
    clean,
    perturbed,
    start_round,
):
    """
    Compare paired clean / perturbed trajectories.
    """

    q_dev = []

    c_dev = []

    perf_dev = []

    signed_perf = []

    for t in range(
        start_round + 1,
        len(
            clean["perf"]
        ),
    ):

        q_dev.append(
            sqrt_js(
                clean["q"][t],
                perturbed["q"][t],
            )
        )

        c_dev.append(
            np.linalg.norm(
                clean["c"][t]
                - perturbed["c"][t]
            )
            / np.sqrt(
                clean["c"].shape[1]
            )
        )

        perf_difference = (
            clean["perf"][t]
            - perturbed["perf"][t]
        )

        signed_perf.append(
            perf_difference
        )

        perf_dev.append(
            abs(
                perf_difference
            )
        )

    return {
        "q": np.asarray(
            q_dev
        ),
        "c": np.asarray(
            c_dev
        ),
        "perf": np.asarray(
            perf_dev
        ),
        "signed_perf":
            np.asarray(
                signed_perf
            ),
    }


def early_late_summary(
    curve,
    early_window=5,
):
    curve = np.asarray(
        curve,
        dtype=float,
    )

    if len(curve) == 0:

        return {
            "early": 0.0,
            "late": 0.0,
            "ratio": 0.0,
        }

    early_n = min(
        early_window,
        len(curve),
    )

    late_n = max(
        5,
        len(curve) // 4,
    )

    late_n = min(
        late_n,
        len(curve),
    )

    early = np.mean(
        curve[:early_n]
    )

    late = np.mean(
        curve[-late_n:]
    )

    ratio = (
        late
        /
        (
            early
            + EPS
        )
    )

    return {
        "early": early,
        "late": late,
        "ratio": ratio,
    }


# ============================================================
# EXPERIMENT 1
#
# LONG-HORIZON SINGLE-ERROR TEST
# ============================================================

def experiment_horizon_scaling():
    print()
    print(
        "============================================"
    )
    print(
        "EXPERIMENT 1: SINGLE-ERROR HORIZON SCALING"
    )
    print(
        "============================================"
    )

    horizons = [
        40,
        80,
        160,
        320,
    ]

    impulse_round = 5

    impulse_indices = [
        0,
        7,
        18,
        31,
    ]

    summaries = {}

    longest_perf_curves = []
    longest_c_curves = []

    for T in horizons:

        cfg = replace(
            BASE_CFG,
            T=T,
        )

        perf_ratios = []
        c_ratios = []

        late_perf_values = []
        late_c_values = []

        snowball_perf = []
        snowball_c = []

        for seed in range(
            30
        ):

            stream = (
                make_random_stream(
                    cfg,
                    20_000 + seed,
                )
            )

            clean = (
                run_custom_trajectory(
                    cfg,
                    stream,
                )
            )

            for idx in (
                impulse_indices
            ):

                perturbed = (
                    run_custom_trajectory(
                        cfg,
                        stream,
                        impulses=[
                            (
                                impulse_round,
                                idx,
                            )
                        ],
                    )
                )

                curves = (
                    deviation_curves(
                        clean,
                        perturbed,
                        impulse_round,
                    )
                )

                perf = (
                    early_late_summary(
                        curves[
                            "perf"
                        ]
                    )
                )

                cap = (
                    early_late_summary(
                        curves[
                            "c"
                        ]
                    )
                )

                perf_ratios.append(
                    perf[
                        "ratio"
                    ]
                )

                c_ratios.append(
                    cap[
                        "ratio"
                    ]
                )

                late_perf_values.append(
                    perf[
                        "late"
                    ]
                )

                late_c_values.append(
                    cap[
                        "late"
                    ]
                )

                # A conservative snowball flag:
                #
                # late deviation > 2x early deviation
                # AND isn't just machine precision.
                snowball_perf.append(
                    (
                        perf[
                            "late"
                        ]
                        >
                        2.0
                        * perf[
                            "early"
                        ]
                    )
                    and
                    (
                        perf[
                            "late"
                        ]
                        > 1e-6
                    )
                )

                snowball_c.append(
                    (
                        cap[
                            "late"
                        ]
                        >
                        2.0
                        * cap[
                            "early"
                        ]
                    )
                    and
                    (
                        cap[
                            "late"
                        ]
                        > 1e-6
                    )
                )

                if T == max(
                    horizons
                ):

                    longest_perf_curves.append(
                        curves[
                            "perf"
                        ]
                    )

                    longest_c_curves.append(
                        curves[
                            "c"
                        ]
                    )

        summaries[T] = {
            "median_perf_ratio":
                np.median(
                    perf_ratios
                ),

            "median_c_ratio":
                np.median(
                    c_ratios
                ),

            "mean_late_perf":
                np.mean(
                    late_perf_values
                ),

            "mean_late_c":
                np.mean(
                    late_c_values
                ),

            "snowball_perf_fraction":
                np.mean(
                    snowball_perf
                ),

            "snowball_c_fraction":
                np.mean(
                    snowball_c
                ),
        }

        print()
        print(
            f"T={T}"
        )

        print(
            "  median late/early performance ratio:",
            summaries[T][
                "median_perf_ratio"
            ],
        )

        print(
            "  median late/early capability ratio:",
            summaries[T][
                "median_c_ratio"
            ],
        )

        print(
            "  mean late performance deviation:",
            summaries[T][
                "mean_late_perf"
            ],
        )

        print(
            "  mean late capability deviation:",
            summaries[T][
                "mean_late_c"
            ],
        )

        print(
            "  fraction performance snowball:",
            summaries[T][
                "snowball_perf_fraction"
            ],
        )

        print(
            "  fraction capability snowball:",
            summaries[T][
                "snowball_c_fraction"
            ],
        )

    # --------------------------------------------------------
    # Longest-horizon average response
    # --------------------------------------------------------

    Path(
        "results"
    ).mkdir(
        exist_ok=True
    )

    perf_matrix = np.asarray(
        longest_perf_curves
    )

    c_matrix = np.asarray(
        longest_c_curves
    )

    h = np.arange(
        1,
        perf_matrix.shape[1]
        + 1,
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        h,
        np.mean(
            perf_matrix,
            axis=0,
        ),
    )

    plt.xlabel(
        "Rounds after one evaluator error"
    )

    plt.ylabel(
        "Mean absolute true-performance deviation"
    )

    plt.title(
        "Single-Error Response over 320-Round Horizon"
    )

    plt.tight_layout()

    plt.savefig(
        "results/v3_single_error_perf_320.png",
        dpi=180,
    )

    plt.close()

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        h,
        np.mean(
            c_matrix,
            axis=0,
        ),
    )

    plt.xlabel(
        "Rounds after one evaluator error"
    )

    plt.ylabel(
        "Mean capability-state deviation"
    )

    plt.title(
        "Single-Error Capability Response over 320 Rounds"
    )

    plt.tight_layout()

    plt.savefig(
        "results/v3_single_error_capability_320.png",
        dpi=180,
    )

    plt.close()

    return summaries


# ============================================================
# EXPERIMENT 2
#
# ERROR DENSITY SCALING
# ============================================================

def fit_power_exponent(
    x,
    y,
):
    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    valid = (
        (x > 0)
        &
        (y > 0)
    )

    if np.sum(
        valid
    ) < 2:

        return np.nan

    return np.polyfit(
        np.log(
            x[valid]
        ),
        np.log(
            y[valid]
        ),
        1,
    )[0]


def experiment_error_density():
    print()
    print(
        "============================================"
    )
    print(
        "EXPERIMENT 2: ERROR DENSITY SCALING"
    )
    print(
        "============================================"
    )

    cfg = replace(
        BASE_CFG,
        T=160,
    )

    error_rates = [
        0.025,
        0.05,
        0.10,
        0.20,
    ]

    results = {}

    for rate in error_rates:

        late_signed = []
        late_abs = []
        full_abs = []
        accuracies = []

        for seed in range(
            30
        ):

            stream = (
                make_random_stream(
                    cfg,
                    30_000 + seed,
                )
            )

            clean = (
                run_custom_trajectory(
                    cfg,
                    stream,
                )
            )

            mask = (
                make_iid_error_mask(
                    cfg,
                    seed=40_000 + seed,
                    error_rate=rate,
                )
            )

            corrupted = (
                run_custom_trajectory(
                    cfg,
                    stream,
                    error_mask=mask,
                )
            )

            diff = (
                clean["perf"]
                - corrupted["perf"]
            )

            late_n = (
                cfg.T // 4
            )

            late_signed.append(
                np.mean(
                    diff[
                        -late_n:
                    ]
                )
            )

            late_abs.append(
                np.mean(
                    np.abs(
                        diff[
                            -late_n:
                        ]
                    )
                )
            )

            full_abs.append(
                np.mean(
                    np.abs(
                        diff
                    )
                )
            )

            accuracies.append(
                corrupted[
                    "accuracy"
                ].mean()
            )

        results[
            rate
        ] = {
            "accuracy":
                np.mean(
                    accuracies
                ),

            "late_signed":
                np.mean(
                    late_signed
                ),

            "late_abs":
                np.mean(
                    late_abs
                ),

            "full_abs":
                np.mean(
                    full_abs
                ),
        }

        print()
        print(
            f"error_rate={rate:.3f}"
        )

        print(
            "  actual accuracy:",
            results[
                rate
            ][
                "accuracy"
            ],
        )

        print(
            "  mean late signed regret:",
            results[
                rate
            ][
                "late_signed"
            ],
        )

        print(
            "  mean late absolute deviation:",
            results[
                rate
            ][
                "late_abs"
            ],
        )

        print(
            "  mean whole-trajectory abs deviation:",
            results[
                rate
            ][
                "full_abs"
            ],
        )

        print(
            "  late abs deviation / error rate:",
            (
                results[
                    rate
                ][
                    "late_abs"
                ]
                / rate
            ),
        )

    late_abs_values = [
        results[
            r
        ][
            "late_abs"
        ]
        for r in error_rates
    ]

    full_abs_values = [
        results[
            r
        ][
            "full_abs"
        ]
        for r in error_rates
    ]

    exponent_late = (
        fit_power_exponent(
            error_rates,
            late_abs_values,
        )
    )

    exponent_full = (
        fit_power_exponent(
            error_rates,
            full_abs_values,
        )
    )

    print()
    print(
        "Power-law scaling:"
    )

    print(
        "  late deviation exponent:",
        exponent_late,
    )

    print(
        "  full trajectory exponent:",
        exponent_full,
    )

    ratio_20_vs_10 = (
        results[
            0.20
        ][
            "late_abs"
        ]
        /
        (
            2.0
            * results[
                0.10
            ][
                "late_abs"
            ]
            + EPS
        )
    )

    print()
    print(
        "H(20%) / [2 * H(10%)] =",
        ratio_20_vs_10,
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plt.figure(
        figsize=(7, 5)
    )

    plt.plot(
        error_rates,
        late_abs_values,
        marker="o",
    )

    plt.xlabel(
        "Evaluator error rate"
    )

    plt.ylabel(
        "Late absolute true-performance deviation"
    )

    plt.title(
        "Does Evaluator Harm Scale Linearly with Error Rate?"
    )

    plt.tight_layout()

    plt.savefig(
        "results/v3_error_density_scaling.png",
        dpi=180,
    )

    plt.close()

    return {
        "results": results,
        "exponent_late":
            exponent_late,
        "exponent_full":
            exponent_full,
        "ratio_20_vs_10":
            ratio_20_vs_10,
    }


# ============================================================
# EXPERIMENT 3
#
# PAIRWISE ERROR INTERACTION
# ============================================================

def late_signed_damage(
    clean,
    corrupted,
    start,
):
    """
    Mean oracle regret after a common evaluation window.

    Positive:
        corrupted agent is worse.

    Negative:
        corruption happened to help.
    """

    start = min(
        start,
        len(
            clean["perf"]
        ) - 1,
    )

    return np.mean(
        clean["perf"][
            start:
        ]
        -
        corrupted["perf"][
            start:
        ]
    )


def experiment_pairwise_interaction():
    print()
    print(
        "============================================"
    )
    print(
        "EXPERIMENT 3: PAIRWISE ERROR INTERACTION"
    )
    print(
        "============================================"
    )

    cfg = replace(
        BASE_CFG,
        T=160,
    )

    error_pairs = [
        (
            (5, 0),
            (20, 7),
        ),
        (
            (5, 7),
            (40, 18),
        ),
        (
            (12, 18),
            (40, 31),
        ),
        (
            (20, 0),
            (60, 7),
        ),
        (
            (5, 31),
            (80, 18),
        ),
        (
            (40, 7),
            (80, 31),
        ),
    ]

    interactions = []
    normalized_interactions = []

    h_single_sum = []
    h_pair = []

    for seed in range(
        30
    ):

        stream = (
            make_random_stream(
                cfg,
                50_000 + seed,
            )
        )

        clean = (
            run_custom_trajectory(
                cfg,
                stream,
            )
        )

        # Cache all single-error trajectories.
        unique_impulses = set()

        for e1, e2 in error_pairs:

            unique_impulses.add(
                e1
            )

            unique_impulses.add(
                e2
            )

        single_runs = {}

        for impulse in (
            unique_impulses
        ):

            single_runs[
                impulse
            ] = (
                run_custom_trajectory(
                    cfg,
                    stream,
                    impulses=[
                        impulse
                    ],
                )
            )

        for e1, e2 in (
            error_pairs
        ):

            both = (
                run_custom_trajectory(
                    cfg,
                    stream,
                    impulses=[
                        e1,
                        e2,
                    ],
                )
            )

            # Evaluate all three effects over the same late window:
            # several rounds after the second error.
            common_start = (
                max(
                    e1[0],
                    e2[0],
                )
                + 5
            )

            H1 = (
                late_signed_damage(
                    clean,
                    single_runs[
                        e1
                    ],
                    common_start,
                )
            )

            H2 = (
                late_signed_damage(
                    clean,
                    single_runs[
                        e2
                    ],
                    common_start,
                )
            )

            H12 = (
                late_signed_damage(
                    clean,
                    both,
                    common_start,
                )
            )

            interaction = (
                H12
                - H1
                - H2
            )

            normalized = (
                interaction
                /
                (
                    abs(H1)
                    + abs(H2)
                    + EPS
                )
            )

            interactions.append(
                interaction
            )

            normalized_interactions.append(
                normalized
            )

            h_single_sum.append(
                H1 + H2
            )

            h_pair.append(
                H12
            )

    interactions = (
        np.asarray(
            interactions
        )
    )

    normalized_interactions = (
        np.asarray(
            normalized_interactions
        )
    )

    h_single_sum = np.asarray(
        h_single_sum
    )

    h_pair = np.asarray(
        h_pair
    )

    print()
    print(
        "Number of error pairs:",
        len(
            interactions
        ),
    )

    print(
        "Mean interaction H12-H1-H2:",
        np.mean(
            interactions
        ),
    )

    print(
        "Median normalized interaction:",
        np.median(
            normalized_interactions
        ),
    )

    print(
        "Mean normalized interaction:",
        np.mean(
            normalized_interactions
        ),
    )

    print(
        "Fraction positive interaction:",
        np.mean(
            interactions > 0
        ),
    )

    print(
        "Fraction normalized interaction > 0.25:",
        np.mean(
            normalized_interactions
            > 0.25
        ),
    )

    print(
        "90th percentile normalized interaction:",
        np.quantile(
            normalized_interactions,
            0.90,
        ),
    )

    # --------------------------------------------------------
    # Plot pair effect vs additive prediction
    # --------------------------------------------------------

    plt.figure(
        figsize=(7, 5)
    )

    plt.scatter(
        h_single_sum,
        h_pair,
        s=18,
        alpha=0.45,
    )

    lo = min(
        np.min(
            h_single_sum
        ),
        np.min(
            h_pair
        ),
    )

    hi = max(
        np.max(
            h_single_sum
        ),
        np.max(
            h_pair
        ),
    )

    plt.plot(
        [
            lo,
            hi,
        ],
        [
            lo,
            hi,
        ],
    )

    plt.xlabel(
        "Additive prediction H(e1) + H(e2)"
    )

    plt.ylabel(
        "Observed joint damage H(e1, e2)"
    )

    plt.title(
        "Do Evaluator Errors Interact Nonlinearly?"
    )

    plt.tight_layout()

    plt.savefig(
        "results/v3_pairwise_interaction.png",
        dpi=180,
    )

    plt.close()

    return {
        "mean_interaction":
            np.mean(
                interactions
            ),

        "median_normalized":
            np.median(
                normalized_interactions
            ),

        "positive_fraction":
            np.mean(
                interactions > 0
            ),

        "strong_positive_fraction":
            np.mean(
                normalized_interactions
                > 0.25
            ),
    }


# ============================================================
# EXPERIMENT 4
#
# SAME ACCURACY, DIFFERENT TEMPORAL STRUCTURE
# ============================================================

def experiment_temporal_persistence():
    print()
    print(
        "============================================"
    )
    print(
        "EXPERIMENT 4: TEMPORAL ERROR PERSISTENCE"
    )
    print(
        "============================================"
    )

    cfg = replace(
        BASE_CFG,
        T=160,
    )

    error_rate = 0.10

    modes = [
        "iid",
        "burst50",
        "burst25",
    ]

    summary = {
        mode: {
            "late_signed": [],
            "late_abs": [],
            "full_abs": [],
            "accuracy": [],
            "diff_curves": [],
        }
        for mode in modes
    }

    for seed in range(
        30
    ):

        stream = (
            make_random_stream(
                cfg,
                60_000 + seed,
            )
        )

        clean = (
            run_custom_trajectory(
                cfg,
                stream,
            )
        )

        masks = {
            "iid":
                make_iid_error_mask(
                    cfg,
                    seed=70_000 + seed,
                    error_rate=error_rate,
                ),

            "burst50":
                make_bursty_error_mask(
                    cfg,
                    seed=70_000 + seed,
                    error_rate=error_rate,
                    active_fraction=0.50,
                ),

            "burst25":
                make_bursty_error_mask(
                    cfg,
                    seed=70_000 + seed,
                    error_rate=error_rate,
                    active_fraction=0.25,
                ),
        }

        for mode in modes:

            corrupted = (
                run_custom_trajectory(
                    cfg,
                    stream,
                    error_mask=masks[
                        mode
                    ],
                )
            )

            diff = (
                clean["perf"]
                - corrupted["perf"]
            )

            late_n = (
                cfg.T // 4
            )

            summary[
                mode
            ][
                "late_signed"
            ].append(
                np.mean(
                    diff[
                        -late_n:
                    ]
                )
            )

            summary[
                mode
            ][
                "late_abs"
            ].append(
                np.mean(
                    np.abs(
                        diff[
                            -late_n:
                        ]
                    )
                )
            )

            summary[
                mode
            ][
                "full_abs"
            ].append(
                np.mean(
                    np.abs(
                        diff
                    )
                )
            )

            summary[
                mode
            ][
                "accuracy"
            ].append(
                corrupted[
                    "accuracy"
                ].mean()
            )

            summary[
                mode
            ][
                "diff_curves"
            ].append(
                np.abs(
                    diff
                )
            )

    print()

    aggregate = {}

    for mode in modes:

        aggregate[
            mode
        ] = {
            "accuracy":
                np.mean(
                    summary[
                        mode
                    ][
                        "accuracy"
                    ]
                ),

            "late_signed":
                np.mean(
                    summary[
                        mode
                    ][
                        "late_signed"
                    ]
                ),

            "late_abs":
                np.mean(
                    summary[
                        mode
                    ][
                        "late_abs"
                    ]
                ),

            "full_abs":
                np.mean(
                    summary[
                        mode
                    ][
                        "full_abs"
                    ]
                ),
        }

        print(
            mode
        )

        print(
            "  accuracy:",
            aggregate[
                mode
            ][
                "accuracy"
            ],
        )

        print(
            "  mean late signed regret:",
            aggregate[
                mode
            ][
                "late_signed"
            ],
        )

        print(
            "  mean late absolute deviation:",
            aggregate[
                mode
            ][
                "late_abs"
            ],
        )

        print(
            "  mean full absolute deviation:",
            aggregate[
                mode
            ][
                "full_abs"
            ],
        )

        print()

    burst50_ratio = (
        aggregate[
            "burst50"
        ][
            "late_abs"
        ]
        /
        (
            aggregate[
                "iid"
            ][
                "late_abs"
            ]
            + EPS
        )
    )

    burst25_ratio = (
        aggregate[
            "burst25"
        ][
            "late_abs"
        ]
        /
        (
            aggregate[
                "iid"
            ][
                "late_abs"
            ]
            + EPS
        )
    )

    print(
        "burst50 / iid late deviation ratio:",
        burst50_ratio,
    )

    print(
        "burst25 / iid late deviation ratio:",
        burst25_ratio,
    )

    # --------------------------------------------------------
    # Plot average deviation trajectory
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    rounds = np.arange(
        cfg.T + 1
    )

    for mode in modes:

        matrix = np.asarray(
            summary[
                mode
            ][
                "diff_curves"
            ]
        )

        plt.plot(
            rounds,
            np.mean(
                matrix,
                axis=0,
            ),
            label=mode,
        )

    plt.xlabel(
        "Adaptation round"
    )

    plt.ylabel(
        "Mean absolute true-performance deviation"
    )

    plt.title(
        "Same Accuracy, Different Temporal Error Structure"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "results/v3_temporal_persistence.png",
        dpi=180,
    )

    plt.close()

    return {
        "aggregate":
            aggregate,
        "burst50_ratio":
            burst50_ratio,
        "burst25_ratio":
            burst25_ratio,
    }


# ============================================================
# FINAL INTERPRETATION
# ============================================================

def main():
    Path(
        "results"
    ).mkdir(
        exist_ok=True
    )

    print(
        "Closed-Loop Evaluation Pilot V3"
    )

    print()
    print(
        "Question:"
    )

    print(
        "Are evaluator errors mostly additive local leverage, "
        "or can repeated feedback create delayed nonlinear "
        "snowball effects?"
    )

    horizon = (
        experiment_horizon_scaling()
    )

    density = (
        experiment_error_density()
    )

    pairwise = (
        experiment_pairwise_interaction()
    )

    temporal = (
        experiment_temporal_persistence()
    )

    print()
    print(
        "============================================"
    )
    print(
        "V3 SUMMARY"
    )
    print(
        "============================================"
    )

    h320 = horizon[
        320
    ]

    print()
    print(
        "LONG-HORIZON SINGLE ERROR"
    )

    print(
        "T=320 median perf late/early ratio:",
        h320[
            "median_perf_ratio"
        ],
    )

    print(
        "T=320 performance snowball fraction:",
        h320[
            "snowball_perf_fraction"
        ],
    )

    print(
        "T=320 capability snowball fraction:",
        h320[
            "snowball_c_fraction"
        ],
    )

    print()
    print(
        "ERROR DENSITY"
    )

    print(
        "late-deviation scaling exponent:",
        density[
            "exponent_late"
        ],
    )

    print(
        "H(20%) / [2 H(10%)]:",
        density[
            "ratio_20_vs_10"
        ],
    )

    print()
    print(
        "PAIRWISE INTERACTION"
    )

    print(
        "median normalized interaction:",
        pairwise[
            "median_normalized"
        ],
    )

    print(
        "fraction positive:",
        pairwise[
            "positive_fraction"
        ],
    )

    print(
        "fraction strong positive:",
        pairwise[
            "strong_positive_fraction"
        ],
    )

    print()
    print(
        "TEMPORAL PERSISTENCE"
    )

    print(
        "burst50 / iid:",
        temporal[
            "burst50_ratio"
        ],
    )

    print(
        "burst25 / iid:",
        temporal[
            "burst25_ratio"
        ],
    )

    # --------------------------------------------------------
    # Pre-specified qualitative evidence counters.
    # --------------------------------------------------------

    evidence = 0

    # 1. Single error becomes >2x larger later.
    if (
        h320[
            "median_perf_ratio"
        ]
        > 2.0
        or
        h320[
            "snowball_perf_fraction"
        ]
        > 0.25
    ):
        evidence += 1

    # 2. Damage rises superlinearly with error density.
    if (
        density[
            "exponent_late"
        ]
        > 1.20
        or
        density[
            "ratio_20_vs_10"
        ]
        > 1.20
    ):
        evidence += 1

    # 3. Errors have substantial positive interaction.
    if (
        pairwise[
            "median_normalized"
        ]
        > 0.10
        and
        pairwise[
            "positive_fraction"
        ]
        > 0.60
    ):
        evidence += 1

    # 4. Same accuracy but clustered errors much worse.
    if (
        temporal[
            "burst25_ratio"
        ]
        > 1.50
        or
        temporal[
            "burst50_ratio"
        ]
        > 1.50
    ):
        evidence += 1

    print()
    print(
        "Snowball evidence count:",
        evidence,
        "/ 4",
    )

    print()

    if evidence >= 3:

        print(
            "CLEAR SNOWBALL REGIME"
        )

        print()

        print(
            "Repeated evaluator errors interact nonlinearly "
            "and/or amplify over long horizons."
        )

        print(
            "The paper should retain BOTH evaluation leverage "
            "and closed-loop stability."
        )

        print()

        print(
            "Interpretation:"
        )

        print(
            "local leverage describes the immediate causal impact "
            "of an error; system stability describes whether "
            "multiple such perturbations compound."
        )

    elif evidence == 2:

        print(
            "MIXED LEVERAGE + AMPLIFICATION"
        )

        print()

        print(
            "Immediate update leverage explains much of the harm, "
            "but there is meaningful nonlinear compounding in "
            "some regimes."
        )

        print(
            "Keep stability as a secondary system-level mechanism."
        )

    else:

        print(
            "MOSTLY ADDITIVE LEVERAGE"
        )

        print()

        print(
            "In this controlled system, evaluator errors mostly "
            "behave like persistent local update leverage rather "
            "than a runaway snowball process."
        )

        print(
            "The main paper framing should emphasize evaluation "
            "leverage; stability can remain a secondary question."
        )


if __name__ == "__main__":
    main()