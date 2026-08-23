from dataclasses import replace

import numpy as np

from pilot_v1 import (
    Config,
    run_impulse_set,
)

from pilot_v2 import (
    ALL_REGIMES,
    TRAIN_REGIMES,
    HELDOUT_REGIMES,
    TRAIN_SEEDS,
    TEST_SEEDS,
    IMPULSE_ROUNDS,
    IMPULSE_INDICES,
    LinearPredictor,
    safe_spearman,
    r2_score,
)


# ============================================================
# CONFIG
# ============================================================

BASE_CFG = Config()

EPS = 1e-10


# ============================================================
# DATASET
# ============================================================

def build_rows(
    regimes,
    seeds,
    name,
):
    """
    Build counterfactual impulse dataset.

    No simulator dynamics are changed from V1/V2.

    We only expose a richer set of EARLY measurements so we can
    test whether the V2 result is trivial persistence or genuinely
    benefits from closed-loop dynamics.
    """

    rows = []

    print()
    print(
        f"===== BUILDING {name.upper()} ====="
    )

    for alpha, beta in regimes:

        print(
            f"alpha={alpha}, beta={beta}"
        )

        cfg = replace(
            BASE_CFG,
            alpha=alpha,
            beta=beta,
        )

        records = run_impulse_set(
            cfg,
            seeds=seeds,
            impulse_rounds=IMPULSE_ROUNDS,
            indices=IMPULSE_INDICES,
        )

        for r in records:

            q = np.asarray(
                r["q_curve"],
                dtype=float,
            )

            c = np.asarray(
                r["c_curve"],
                dtype=float,
            )

            perf = np.asarray(
                r["perf_curve"],
                dtype=float,
            )

            # ------------------------------------------------
            # EARLY measurements
            # ------------------------------------------------

            # One-step perturbation magnitude.
            q1 = q[0]
            c1 = c[0]
            p1 = perf[0]

            # First three rounds.
            q3_mean = np.mean(
                q[:3]
            )

            c3_mean = np.mean(
                c[:3]
            )

            p3_mean = np.mean(
                perf[:3]
            )

            # Last observed value after short probe.
            q3_last = q[
                min(2, len(q) - 1)
            ]

            c3_last = c[
                min(2, len(c) - 1)
            ]

            p3_last = perf[
                min(2, len(perf) - 1)
            ]

            # Early performance growth / decay.
            if len(perf) >= 3:

                x = np.arange(
                    1,
                    4,
                    dtype=float,
                )

                y = np.log(
                    np.maximum(
                        perf[:3],
                        EPS,
                    )
                )

                lambda_perf = (
                    np.polyfit(
                        x,
                        y,
                        1,
                    )[0]
                )

            else:

                lambda_perf = 0.0

            # ------------------------------------------------
            # LONG-HORIZON target
            # ------------------------------------------------

            # Skip first 5 rounds so predictor and target are
            # temporally separated.
            if len(perf) > 5:

                late_perf = np.mean(
                    perf[5:]
                )

            else:

                late_perf = (
                    perf[-1]
                )

            rows.append(
                {
                    "alpha": alpha,
                    "beta": beta,

                    "remaining_fraction":
                        (
                            BASE_CFG.T
                            - r["t0"]
                        )
                        / BASE_CFG.T,

                    # One-step baselines.
                    "q1": q1,
                    "c1": c1,
                    "p1": p1,

                    # Three-step impact.
                    "q3_mean":
                        q3_mean,

                    "c3_mean":
                        c3_mean,

                    "p3_mean":
                        p3_mean,

                    "q3_last":
                        q3_last,

                    "c3_last":
                        c3_last,

                    "p3_last":
                        p3_last,

                    # Growth / decay.
                    "lambda_q":
                        r["lambda_q"],

                    "lambda_c":
                        r["lambda_c"],

                    "lambda_perf":
                        lambda_perf,

                    # Target.
                    "late_perf":
                        late_perf,
                }
            )

    print(
        f"{name}: {len(rows)} interventions"
    )

    return rows


# ============================================================
# FEATURES
# ============================================================

def log_feature(x):
    return np.log10(
        max(
            float(x),
            EPS,
        )
    )


def feature_vector(
    row,
    feature_set,
):
    """
    Falsification baselines.

    system:
        Only nominal system parameters.

    early_perf:
        Only the true-performance deviation visible during
        the first few rounds.

    one_step:
        Only one-step state displacement.

    impact:
        Three-round state displacement.

    growth:
        Only growth/decay rates.

    impact_growth:
        State impact + dynamic growth information.

    impact_perf:
        Impact + early performance baseline.

    all_early:
        Everything observable during short perturbation probe.

    full:
        all early observations + nominal alpha/beta.
    """

    if feature_set == "system":

        return np.array(
            [
                row["alpha"],
                row["beta"],
                row[
                    "remaining_fraction"
                ],
            ],
            dtype=float,
        )

    if feature_set == "early_perf":

        return np.array(
            [
                log_feature(
                    row["p1"]
                ),

                log_feature(
                    row["p3_mean"]
                ),

                log_feature(
                    row["p3_last"]
                ),

                row[
                    "lambda_perf"
                ],
            ],
            dtype=float,
        )

    if feature_set == "one_step":

        return np.array(
            [
                log_feature(
                    row["q1"]
                ),

                log_feature(
                    row["c1"]
                ),

                log_feature(
                    row["p1"]
                ),
            ],
            dtype=float,
        )

    if feature_set == "impact":

        return np.array(
            [
                log_feature(
                    row["q3_mean"]
                ),

                log_feature(
                    row["c3_mean"]
                ),
            ],
            dtype=float,
        )

    if feature_set == "growth":

        return np.array(
            [
                row["lambda_q"],
                row["lambda_c"],
            ],
            dtype=float,
        )

    if feature_set == "impact_growth":

        return np.array(
            [
                log_feature(
                    row["q3_mean"]
                ),

                log_feature(
                    row["c3_mean"]
                ),

                row["lambda_q"],
                row["lambda_c"],
            ],
            dtype=float,
        )

    if feature_set == "impact_perf":

        return np.array(
            [
                log_feature(
                    row["q3_mean"]
                ),

                log_feature(
                    row["c3_mean"]
                ),

                log_feature(
                    row["p3_mean"]
                ),
            ],
            dtype=float,
        )

    if feature_set == "all_early":

        return np.array(
            [
                log_feature(
                    row["q1"]
                ),

                log_feature(
                    row["c1"]
                ),

                log_feature(
                    row["p1"]
                ),

                log_feature(
                    row["q3_mean"]
                ),

                log_feature(
                    row["c3_mean"]
                ),

                log_feature(
                    row["p3_mean"]
                ),

                row["lambda_q"],
                row["lambda_c"],
                row[
                    "lambda_perf"
                ],
            ],
            dtype=float,
        )

    if feature_set == "full":

        return np.array(
            [
                row["alpha"],
                row["beta"],
                row[
                    "remaining_fraction"
                ],

                log_feature(
                    row["q1"]
                ),

                log_feature(
                    row["c1"]
                ),

                log_feature(
                    row["p1"]
                ),

                log_feature(
                    row["q3_mean"]
                ),

                log_feature(
                    row["c3_mean"]
                ),

                log_feature(
                    row["p3_mean"]
                ),

                row["lambda_q"],
                row["lambda_c"],
                row[
                    "lambda_perf"
                ],
            ],
            dtype=float,
        )

    raise ValueError(
        feature_set
    )


FEATURE_SETS = [
    "system",
    "early_perf",
    "one_step",
    "impact",
    "growth",
    "impact_growth",
    "impact_perf",
    "all_early",
    "full",
]


# ============================================================
# X / Y
# ============================================================

def make_xy(
    rows,
    feature_set,
):
    X = np.asarray(
        [
            feature_vector(
                row,
                feature_set,
            )
            for row in rows
        ],
        dtype=float,
    )

    y_raw = np.asarray(
        [
            row[
                "late_perf"
            ]
            for row in rows
        ],
        dtype=float,
    )

    y_log = np.log10(
        y_raw + EPS
    )

    return (
        X,
        y_log,
        y_raw,
    )


# ============================================================
# METRICS
# ============================================================

def evaluate(
    model,
    X,
    y_log,
    y_raw,
):
    pred_log = (
        model.predict(
            X
        )
    )

    pred_raw = np.maximum(
        10 ** pred_log
        - EPS,
        0.0,
    )

    rho = safe_spearman(
        pred_raw,
        y_raw,
    )

    r2_log = r2_score(
        y_log,
        pred_log,
    )

    r2_raw = r2_score(
        y_raw,
        pred_raw,
    )

    mae = np.mean(
        np.abs(
            pred_raw
            - y_raw
        )
    )

    normalized_mae = (
        mae
        / np.mean(
            y_raw
        )
    )

    return {
        "rho": rho,
        "r2_log":
            r2_log,
        "r2_raw":
            r2_raw,
        "nmae":
            normalized_mae,
    }


# ============================================================
# TRAIN / TEST
# ============================================================

def train_and_test(
    train_rows,
    test_rows,
    feature_set,
):
    (
        X_train,
        y_train_log,
        _
    ) = make_xy(
        train_rows,
        feature_set,
    )

    (
        X_test,
        y_test_log,
        y_test_raw,
    ) = make_xy(
        test_rows,
        feature_set,
    )

    model = LinearPredictor()

    model.fit(
        X_train,
        y_train_log,
    )

    return evaluate(
        model,
        X_test,
        y_test_log,
        y_test_raw,
    )


# ============================================================
# SHUFFLE CONTROL
# ============================================================

def shuffle_control(
    train_rows,
    test_rows,
    seed=12345,
):
    """
    Train the strongest feature set against randomly permuted
    long-horizon outcomes.

    If this still predicts well, we have leakage / implementation
    problems.
    """

    feature_set = (
        "all_early"
    )

    (
        X_train,
        y_train_log,
        _
    ) = make_xy(
        train_rows,
        feature_set,
    )

    (
        X_test,
        y_test_log,
        y_test_raw,
    ) = make_xy(
        test_rows,
        feature_set,
    )

    rng = (
        np.random.default_rng(
            seed
        )
    )

    shuffled = (
        y_train_log.copy()
    )

    rng.shuffle(
        shuffled
    )

    model = LinearPredictor()

    model.fit(
        X_train,
        shuffled,
    )

    return evaluate(
        model,
        X_test,
        y_test_log,
        y_test_raw,
    )


# ============================================================
# PRIMARY HELD-OUT TEST
# ============================================================

def primary_test():
    print()
    print(
        "============================================"
    )

    print(
        "V2.5 PRIMARY FALSIFICATION TEST"
    )

    print(
        "============================================"
    )

    train_rows = build_rows(
        TRAIN_REGIMES,
        TRAIN_SEEDS,
        "train",
    )

    test_rows = build_rows(
        HELDOUT_REGIMES,
        TEST_SEEDS,
        "heldout",
    )

    results = {}

    print()
    print(
        "===== FEATURE ABLATION ====="
    )

    print(
        "feature              rho     R2_log   R2_raw   nMAE"
    )

    for feature_set in (
        FEATURE_SETS
    ):

        metrics = (
            train_and_test(
                train_rows,
                test_rows,
                feature_set,
            )
        )

        results[
            feature_set
        ] = metrics

        print(
            f"{feature_set:18s} "
            f"{metrics['rho']:>6.3f} "
            f"{metrics['r2_log']:>8.3f} "
            f"{metrics['r2_raw']:>8.3f} "
            f"{metrics['nmae']:>7.3f}"
        )

    shuffled = (
        shuffle_control(
            train_rows,
            test_rows,
        )
    )

    print()
    print(
        "===== SHUFFLE CONTROL ====="
    )

    print(
        f"rho={shuffled['rho']:.3f} "
        f"R2_log={shuffled['r2_log']:.3f} "
        f"R2_raw={shuffled['r2_raw']:.3f}"
    )

    return (
        results,
        shuffled,
    )


# ============================================================
# LEAVE-ONE-REGIME-OUT
# ============================================================

def leave_one_regime_out():
    print()
    print(
        "============================================"
    )

    print(
        "V2.5 LEAVE-ONE-REGIME-OUT"
    )

    print(
        "============================================"
    )

    datasets = {}

    for j, regime in enumerate(
        ALL_REGIMES
    ):

        start_seed = (
            1000
            + 20 * j
        )

        rows = build_rows(
            [regime],
            range(
                start_seed,
                start_seed + 10,
            ),
            f"regime {regime}",
        )

        datasets[
            regime
        ] = rows

    looro_results = {
        feature_set: []
        for feature_set
        in FEATURE_SETS
    }

    for heldout in (
        ALL_REGIMES
    ):

        train_rows = []

        for regime in (
            ALL_REGIMES
        ):

            if regime != heldout:

                train_rows.extend(
                    datasets[
                        regime
                    ]
                )

        test_rows = (
            datasets[
                heldout
            ]
        )

        for feature_set in (
            FEATURE_SETS
        ):

            metrics = (
                train_and_test(
                    train_rows,
                    test_rows,
                    feature_set,
                )
            )

            looro_results[
                feature_set
            ].append(
                metrics["rho"]
            )

    print()
    print(
        "===== LOORO MEDIAN RHO ====="
    )

    print(
        "feature              median_rho   min_rho   >0.5"
    )

    for feature_set in (
        FEATURE_SETS
    ):

        values = np.asarray(
            looro_results[
                feature_set
            ]
        )

        print(
            f"{feature_set:18s} "
            f"{np.nanmedian(values):>10.3f} "
            f"{np.nanmin(values):>9.3f} "
            f"{np.mean(values > 0.5):>6.2f}"
        )

    return looro_results


# ============================================================
# INTERPRETATION
# ============================================================

def interpret(
    primary,
    shuffled,
    looro,
):
    early_perf = (
        primary[
            "early_perf"
        ]
    )

    one_step = (
        primary[
            "one_step"
        ]
    )

    impact = (
        primary[
            "impact"
        ]
    )

    growth = (
        primary[
            "growth"
        ]
    )

    impact_growth = (
        primary[
            "impact_growth"
        ]
    )

    all_early = (
        primary[
            "all_early"
        ]
    )

    median_impact = (
        np.nanmedian(
            looro[
                "impact"
            ]
        )
    )

    median_impact_growth = (
        np.nanmedian(
            looro[
                "impact_growth"
            ]
        )
    )

    median_early_perf = (
        np.nanmedian(
            looro[
                "early_perf"
            ]
        )
    )

    print()
    print(
        "============================================"
    )

    print(
        "V2.5 INTERPRETATION"
    )

    print(
        "============================================"
    )

    print()

    print(
        "1. EARLY PERFORMANCE BASELINE"
    )

    print(
        "early_perf rho =",
        early_perf[
            "rho"
        ],
    )

    print(
        "impact_growth rho =",
        impact_growth[
            "rho"
        ],
    )

    print()

    print(
        "2. ONE-STEP PERSISTENCE"
    )

    print(
        "one_step rho =",
        one_step[
            "rho"
        ],
    )

    print(
        "three-step impact rho =",
        impact[
            "rho"
        ],
    )

    print()

    print(
        "3. DOES GROWTH ADD INFORMATION?"
    )

    print(
        "impact-only rho =",
        impact[
            "rho"
        ],
    )

    print(
        "growth-only rho =",
        growth[
            "rho"
        ],
    )

    print(
        "impact+growth rho =",
        impact_growth[
            "rho"
        ],
    )

    print()

    print(
        "4. SHUFFLE CONTROL"
    )

    print(
        "shuffle rho =",
        shuffled[
            "rho"
        ],
    )

    print()

    print(
        "5. LOORO"
    )

    print(
        "median early-performance rho =",
        median_early_perf,
    )

    print(
        "median impact-only rho =",
        median_impact,
    )

    print(
        "median impact+growth rho =",
        median_impact_growth,
    )

    print()

    # --------------------------------------------------------
    # Decision logic
    # --------------------------------------------------------

    leakage_ok = (
        abs(
            shuffled[
                "rho"
            ]
        )
        < 0.15
    )

    state_beats_perf = (
        impact_growth[
            "rho"
        ]
        >
        early_perf[
            "rho"
        ]
        + 0.05
    )

    growth_adds = (
        impact_growth[
            "rho"
        ]
        >
        impact[
            "rho"
        ]
        + 0.03
    )

    not_one_step_trivial = (
        impact_growth[
            "rho"
        ]
        >
        one_step[
            "rho"
        ]
        + 0.05
    )

    looro_robust = (
        median_impact_growth
        > 0.5
    )

    print()

    if (
        leakage_ok
        and
        looro_robust
        and
        state_beats_perf
        and
        (
            growth_adds
            or
            not_one_step_trivial
        )
    ):

        print(
            "STRONG FALSIFICATION PASS"
        )

        print()

        print(
            "The V2 result is not explained by shuffled leakage, "
            "early performance alone, or trivial one-step persistence."
        )

        print(
            "Dynamic closed-loop state response adds measurable "
            "predictive information."
        )

        print()

        print(
            "Proceed to V3: unseen error topology."
        )

    elif (
        leakage_ok
        and
        looro_robust
    ):

        print(
            "PARTIAL FALSIFICATION PASS"
        )

        print()

        print(
            "The predictive phenomenon is real and robust, "
            "but the current 'stability dynamics' framing may be too strong."
        )

        print(
            "Inspect whether impact alone explains nearly everything."
        )

        print(
            "If so, the better concept may be closed-loop update leverage "
            "rather than instability/growth."
        )

    else:

        print(
            "FALSIFICATION CONCERN"
        )

        print()

        print(
            "A trivial baseline or implementation artifact explains "
            "too much of the V2 result."
        )

        print(
            "Do not proceed to real agents until this is understood."
        )


# ============================================================
# MAIN
# ============================================================

def main():
    print(
        "Closed-Loop Stability Pilot V2.5"
    )

    print()

    print(
        "No simulator parameters are changed."
    )

    print(
        "Goal: attempt to explain away the strong V2 result "
        "with simpler baselines."
    )

    (
        primary,
        shuffled,
    ) = primary_test()

    looro = (
        leave_one_regime_out()
    )

    interpret(
        primary,
        shuffled,
        looro,
    )


if __name__ == "__main__":
    main()