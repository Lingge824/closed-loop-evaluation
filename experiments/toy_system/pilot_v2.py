from dataclasses import replace
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from pilot_v1 import (
    Config,
    run_impulse_set,
)


# ============================================================
# CONFIG
# ============================================================

BASE_CFG = Config()

# The same 12 regimes used in v1.
ALL_REGIMES = [
    (0.5, 0.02),
    (1.0, 0.02),
    (2.0, 0.02),
    (4.0, 0.02),

    (0.5, 0.05),
    (1.0, 0.05),
    (2.0, 0.05),
    (4.0, 0.05),

    (0.5, 0.10),
    (1.0, 0.10),
    (2.0, 0.10),
    (4.0, 0.10),
]


# ------------------------------------------------------------
# IMPORTANT:
#
# Training only sees low/medium feedback regimes.
#
# It NEVER sees:
#   alpha = 4.0
#   OR
#   beta = 0.10
#
# during fitting.
# ------------------------------------------------------------

TRAIN_REGIMES = [
    (0.5, 0.02),
    (1.0, 0.02),
    (2.0, 0.02),

    (0.5, 0.05),
    (1.0, 0.05),
    (2.0, 0.05),
]


HELDOUT_REGIMES = [
    (4.0, 0.02),
    (4.0, 0.05),

    (0.5, 0.10),
    (1.0, 0.10),
    (2.0, 0.10),

    (4.0, 0.10),
]


PRIMARY_HELDOUT = (4.0, 0.10)


# Separate random seeds.
#
# Training and held-out experiments therefore differ in BOTH:
#   1. feedback regime
#   2. random realization
#
# This is stricter than reusing identical seeds.

TRAIN_SEEDS = range(20)
TEST_SEEDS = range(100, 120)


IMPULSE_ROUNDS = [
    5,
    8,
    12,
    16,
    20,
]


IMPULSE_INDICES = [
    0,
    7,
    18,
    31,
]


EPS = 1e-10


# ============================================================
# DATASET CREATION
# ============================================================

def build_dataset(
    regimes,
    seeds,
    name,
):
    """
    Generate single-error counterfactual experiments.

    Each data point corresponds to ONE evaluator verdict flip.

    Static evaluator corruption is therefore identical:
        exactly one mistake.

    What differs is how the adaptive system responds to it.
    """

    rows = []

    print()
    print(
        f"===== BUILDING {name.upper()} DATA ====="
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

        for record in records:

            q_curve = np.asarray(
                record["q_curve"]
            )

            c_curve = np.asarray(
                record["c_curve"]
            )

            perf_curve = np.asarray(
                record["perf_curve"]
            )

            # ------------------------------------------------
            # EARLY information.
            #
            # Only first 3-5 rounds after the perturbation.
            # ------------------------------------------------

            early_q_impact = np.mean(
                q_curve[:3]
            )

            early_c_impact = np.mean(
                c_curve[:3]
            )

            lambda_q = record[
                "lambda_q"
            ]

            lambda_c = record[
                "lambda_c"
            ]

            # ------------------------------------------------
            # LONG-HORIZON outcome.
            #
            # Skip the first five rounds used for early probing.
            #
            # We use MEAN rather than SUM so interventions at
            # different t0 values have comparable scale.
            # ------------------------------------------------

            if len(perf_curve) > 5:

                late_perf_mean = np.mean(
                    perf_curve[5:]
                )

                late_q_mean = np.mean(
                    q_curve[5:]
                )

                late_c_mean = np.mean(
                    c_curve[5:]
                )

            else:

                late_perf_mean = (
                    perf_curve[-1]
                )

                late_q_mean = (
                    q_curve[-1]
                )

                late_c_mean = (
                    c_curve[-1]
                )

            rows.append(
                {
                    "alpha": alpha,
                    "beta": beta,

                    "seed":
                        record["seed"],

                    "t0":
                        record["t0"],

                    "idx":
                        record["idx"],

                    # Known system information.
                    "remaining_fraction":
                        (
                            BASE_CFG.T
                            - record["t0"]
                        )
                        / BASE_CFG.T,

                    # Closed-loop response.
                    "early_q_impact":
                        early_q_impact,

                    "early_c_impact":
                        early_c_impact,

                    "lambda_q":
                        lambda_q,

                    "lambda_c":
                        lambda_c,

                    # Long-horizon targets.
                    "late_perf":
                        late_perf_mean,

                    "late_q":
                        late_q_mean,

                    "late_c":
                        late_c_mean,
                }
            )

    print(
        f"{name}: {len(rows)} interventions"
    )

    return rows


# ============================================================
# FEATURES
# ============================================================

def feature_vector(
    row,
    feature_set,
):
    """
    Compare several predictors.

    STATIC:
        Every experiment has exactly one evaluator error,
        so static evaluator accuracy contains ZERO information.
        Constant predictor represents that baseline.

    SYSTEM:
        knows feedback parameters but does not observe how
        the system actually reacts.

    CLOSED_LOOP:
        observes only the first few rounds after perturbation.

    FULL:
        system parameters + early closed-loop response.
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

    elif feature_set == "closed_loop":

        return np.array(
            [
                np.log10(
                    row[
                        "early_q_impact"
                    ]
                    + EPS
                ),

                np.log10(
                    row[
                        "early_c_impact"
                    ]
                    + EPS
                ),

                row["lambda_q"],
                row["lambda_c"],
            ],
            dtype=float,
        )

    elif feature_set == "full":

        return np.array(
            [
                row["alpha"],
                row["beta"],
                row[
                    "remaining_fraction"
                ],

                np.log10(
                    row[
                        "early_q_impact"
                    ]
                    + EPS
                ),

                np.log10(
                    row[
                        "early_c_impact"
                    ]
                    + EPS
                ),

                row["lambda_q"],
                row["lambda_c"],
            ],
            dtype=float,
        )

    else:
        raise ValueError(
            feature_set
        )


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
        ]
    )

    # Long-run deviation is positive and strongly skewed.
    #
    # Predict log deviation.
    y_raw = np.asarray(
        [
            row["late_perf"]
            for row in rows
        ]
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
# VERY SIMPLE LINEAR MODEL
# ============================================================

class LinearPredictor:

    def __init__(self):
        self.mean = None
        self.std = None
        self.coef = None

    def fit(
        self,
        X,
        y,
    ):
        """
        Standardized ordinary least squares.

        Deliberately simple:
        NO neural network
        NO random forest
        NO tuning
        NO hyperparameter search.
        """

        self.mean = X.mean(
            axis=0
        )

        self.std = X.std(
            axis=0
        )

        self.std[
            self.std < 1e-12
        ] = 1.0

        Xs = (
            X - self.mean
        ) / self.std

        # Add intercept.
        design = np.column_stack(
            [
                np.ones(
                    len(Xs)
                ),
                Xs,
            ]
        )

        self.coef = (
            np.linalg.lstsq(
                design,
                y,
                rcond=None,
            )[0]
        )

    def predict(
        self,
        X,
    ):
        Xs = (
            X - self.mean
        ) / self.std

        design = np.column_stack(
            [
                np.ones(
                    len(Xs)
                ),
                Xs,
            ]
        )

        return (
            design
            @ self.coef
        )


# ============================================================
# METRICS
# ============================================================

def safe_spearman(
    x,
    y,
):
    x = np.asarray(x)
    y = np.asarray(y)

    valid = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

    x = x[valid]
    y = y[valid]

    if (
        len(x) < 3
        or
        np.std(x) < 1e-15
        or
        np.std(y) < 1e-15
    ):
        return np.nan

    return spearmanr(
        x,
        y,
    ).statistic


def r2_score(
    y_true,
    y_pred,
):
    denominator = np.sum(
        (
            y_true
            - y_true.mean()
        ) ** 2
    )

    if denominator < 1e-15:
        return np.nan

    numerator = np.sum(
        (
            y_true
            - y_pred
        ) ** 2
    )

    return (
        1.0
        - numerator
        / denominator
    )


def evaluate_predictions(
    y_log_true,
    y_log_pred,
    y_raw_true,
):
    pred_raw = (
        10 ** y_log_pred
        - EPS
    )

    pred_raw = np.maximum(
        pred_raw,
        0.0,
    )

    rho = safe_spearman(
        pred_raw,
        y_raw_true,
    )

    r2_log = r2_score(
        y_log_true,
        y_log_pred,
    )

    r2_raw = r2_score(
        y_raw_true,
        pred_raw,
    )

    mae = np.mean(
        np.abs(
            pred_raw
            - y_raw_true
        )
    )

    mean_target = np.mean(
        y_raw_true
    )

    normalized_mae = (
        mae
        / mean_target
        if mean_target > 0
        else np.nan
    )

    return {
        "rho": rho,
        "r2_log": r2_log,
        "r2_raw": r2_raw,
        "mae": mae,
        "normalized_mae":
            normalized_mae,
        "pred_raw":
            pred_raw,
    }


# ============================================================
# CONSTANT / STATIC-ACCURACY BASELINE
# ============================================================

def constant_prediction(
    y_train_log,
    n_test,
):
    """
    Since every impulse contains exactly ONE evaluator error,
    static accuracy cannot distinguish interventions.

    The strongest possible "accuracy only" model therefore predicts
    the same expected long-term effect for every intervention.
    """

    value = np.mean(
        y_train_log
    )

    return np.full(
        n_test,
        value,
    )


# ============================================================
# TRAIN + HELD-OUT TEST
# ============================================================

def run_heldout_prediction(
    train_rows,
    test_rows,
):
    results = {}

    # --------------------------------------------------------
    # Constant baseline
    # --------------------------------------------------------

    _, y_train_log, _ = (
        make_xy(
            train_rows,
            "system",
        )
    )

    _, y_test_log, y_test_raw = (
        make_xy(
            test_rows,
            "system",
        )
    )

    baseline_pred = (
        constant_prediction(
            y_train_log,
            len(test_rows),
        )
    )

    results["constant"] = (
        evaluate_predictions(
            y_test_log,
            baseline_pred,
            y_test_raw,
        )
    )

    # --------------------------------------------------------
    # Learned predictors
    # --------------------------------------------------------

    for feature_set in [
        "system",
        "closed_loop",
        "full",
    ]:

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
            y_test_raw
        ) = make_xy(
            test_rows,
            feature_set,
        )

        model = LinearPredictor()

        model.fit(
            X_train,
            y_train_log,
        )

        prediction = model.predict(
            X_test
        )

        metrics = (
            evaluate_predictions(
                y_test_log,
                prediction,
                y_test_raw,
            )
        )

        metrics["model"] = model

        results[
            feature_set
        ] = metrics

    return results


# ============================================================
# REGIME-LEVEL EVALUATION
# ============================================================

def regime_level_table(
    test_rows,
    predictions,
):
    regimes = sorted(
        set(
            (
                row["alpha"],
                row["beta"],
            )
            for row in test_rows
        )
    )

    rows_out = []

    for regime in regimes:

        indices = [
            i
            for i, row
            in enumerate(test_rows)
            if (
                row["alpha"],
                row["beta"],
            ) == regime
        ]

        true_values = np.asarray(
            [
                test_rows[i][
                    "late_perf"
                ]
                for i in indices
            ]
        )

        pred_values = predictions[
            indices
        ]

        rows_out.append(
            {
                "alpha":
                    regime[0],

                "beta":
                    regime[1],

                "true_mean":
                    true_values.mean(),

                "pred_mean":
                    pred_values.mean(),

                "ratio":
                    (
                        pred_values.mean()
                        /
                        true_values.mean()
                    ),
            }
        )

    return rows_out


# ============================================================
# PRIMARY TEST
# ============================================================

def experiment_primary_heldout():
    print()
    print(
        "============================================"
    )

    print(
        "PRIMARY HELD-OUT PREDICTION TEST"
    )

    print(
        "============================================"
    )

    print()
    print(
        "TRAIN regimes:"
    )

    for r in TRAIN_REGIMES:
        print(
            " ",
            r,
        )

    print()
    print(
        "HELD-OUT regimes:"
    )

    for r in HELDOUT_REGIMES:
        print(
            " ",
            r,
        )

    train_rows = build_dataset(
        TRAIN_REGIMES,
        TRAIN_SEEDS,
        "train",
    )

    test_rows = build_dataset(
        HELDOUT_REGIMES,
        TEST_SEEDS,
        "heldout",
    )

    results = (
        run_heldout_prediction(
            train_rows,
            test_rows,
        )
    )

    print()
    print(
        "===== HELD-OUT RESULTS ====="
    )

    print()

    for name in [
        "constant",
        "system",
        "closed_loop",
        "full",
    ]:

        r = results[name]

        print(
            f"{name:12s} "
            f"rho={r['rho']:.3f} "
            f"R2_log={r['r2_log']:.3f} "
            f"R2_raw={r['r2_raw']:.3f} "
            f"nMAE={r['normalized_mae']:.3f}"
        )

    # --------------------------------------------------------
    # Regime-level prediction using FULL model.
    # --------------------------------------------------------

    full_pred = (
        results["full"][
            "pred_raw"
        ]
    )

    table = regime_level_table(
        test_rows,
        full_pred,
    )

    print()
    print(
        "===== HELD-OUT REGIME MEANS ====="
    )

    print(
        "alpha  beta   true_mean    pred_mean    pred/true"
    )

    for row in table:

        print(
            f"{row['alpha']:<5.1f} "
            f"{row['beta']:<6.2f} "
            f"{row['true_mean']:<12.6g} "
            f"{row['pred_mean']:<12.6g} "
            f"{row['ratio']:.3f}"
        )

    regime_true = np.asarray(
        [
            r["true_mean"]
            for r in table
        ]
    )

    regime_pred = np.asarray(
        [
            r["pred_mean"]
            for r in table
        ]
    )

    regime_rho = safe_spearman(
        regime_pred,
        regime_true,
    )

    print()
    print(
        "Regime-level Spearman rho =",
        regime_rho,
    )

    # --------------------------------------------------------
    # PRIMARY unseen high-feedback regime
    # --------------------------------------------------------

    primary_indices = [
        i
        for i, row
        in enumerate(test_rows)
        if (
            row["alpha"],
            row["beta"],
        ) == PRIMARY_HELDOUT
    ]

    primary_true = np.asarray(
        [
            test_rows[i][
                "late_perf"
            ]
            for i in primary_indices
        ]
    )

    primary_pred = full_pred[
        primary_indices
    ]

    primary_rho = safe_spearman(
        primary_pred,
        primary_true,
    )

    primary_r2 = r2_score(
        np.log10(
            primary_true + EPS
        ),
        np.log10(
            primary_pred + EPS
        ),
    )

    print()
    print(
        "===== PRIMARY UNSEEN REGIME ====="
    )

    print(
        "alpha=4.0, beta=0.10"
    )

    print(
        "Number of interventions:",
        len(primary_true),
    )

    print(
        "Spearman rho:",
        primary_rho,
    )

    print(
        "Log-space R2:",
        primary_r2,
    )

    print(
        "True mean late deviation:",
        primary_true.mean(),
    )

    print(
        "Predicted mean late deviation:",
        primary_pred.mean(),
    )

    # --------------------------------------------------------
    # Plot individual predictions
    # --------------------------------------------------------

    Path(
        "results"
    ).mkdir(
        exist_ok=True
    )

    y_true = np.asarray(
        [
            row["late_perf"]
            for row in test_rows
        ]
    )

    plt.figure(
        figsize=(7, 5)
    )

    plt.scatter(
        y_true,
        full_pred,
        s=15,
        alpha=0.4,
    )

    lo = min(
        y_true.min(),
        full_pred.min(),
    )

    hi = max(
        y_true.max(),
        full_pred.max(),
    )

    plt.plot(
        [lo, hi],
        [lo, hi],
    )

    plt.xlabel(
        "True long-horizon deviation"
    )

    plt.ylabel(
        "Predicted long-horizon deviation"
    )

    plt.title(
        "Held-Out Closed-Loop Prediction"
    )

    plt.tight_layout()

    plt.savefig(
        "results/v2_heldout_prediction.png",
        dpi=180,
    )

    plt.close()

    # --------------------------------------------------------
    # Model comparison
    # --------------------------------------------------------

    names = [
        "constant",
        "system",
        "closed_loop",
        "full",
    ]

    rho_values = [
        results[name]["rho"]
        for name in names
    ]

    plt.figure(
        figsize=(7, 5)
    )

    plt.bar(
        names,
        rho_values,
    )

    plt.ylabel(
        "Held-out Spearman rho"
    )

    plt.title(
        "What Predicts Long-Term Harm?"
    )

    plt.tight_layout()

    plt.savefig(
        "results/v2_model_comparison.png",
        dpi=180,
    )

    plt.close()

    # --------------------------------------------------------
    # Regime means
    # --------------------------------------------------------

    x = np.arange(
        len(table)
    )

    width = 0.35

    plt.figure(
        figsize=(9, 5)
    )

    plt.bar(
        x - width / 2,
        regime_true,
        width,
        label="true",
    )

    plt.bar(
        x + width / 2,
        regime_pred,
        width,
        label="predicted",
    )

    plt.xticks(
        x,
        [
            f"a={r['alpha']},b={r['beta']}"
            for r in table
        ],
        rotation=35,
        ha="right",
    )

    plt.ylabel(
        "Mean long-horizon deviation"
    )

    plt.title(
        "Prediction of Entire Unseen Feedback Regimes"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "results/v2_regime_prediction.png",
        dpi=180,
    )

    plt.close()

    return (
        train_rows,
        test_rows,
        results,
        regime_rho,
        primary_rho,
        primary_r2,
    )


# ============================================================
# LEAVE-ONE-REGIME-OUT TEST
# ============================================================

def experiment_leave_one_regime_out():
    """
    Stronger robustness check.

    Each regime is held out once.

    Train on the other 11 regimes.
    Test ONLY on the unseen regime.

    We care about whether closed-loop features consistently
    predict individual intervention consequences.
    """

    print()
    print(
        "============================================"
    )

    print(
        "LEAVE-ONE-REGIME-OUT"
    )

    print(
        "============================================"
    )

    # Use fewer seeds here to keep runtime reasonable.
    dataset_by_regime = {}

    for regime_index, regime in enumerate(
        ALL_REGIMES
    ):

        alpha, beta = regime

        # Different seed block for each regime.
        seed_start = (
            500
            + regime_index * 20
        )

        rows = build_dataset(
            [regime],
            range(
                seed_start,
                seed_start + 10,
            ),
            f"regime {regime}",
        )

        dataset_by_regime[
            regime
        ] = rows

    summary = []

    for heldout in ALL_REGIMES:

        train_rows = []

        for regime in ALL_REGIMES:

            if regime != heldout:

                train_rows.extend(
                    dataset_by_regime[
                        regime
                    ]
                )

        test_rows = (
            dataset_by_regime[
                heldout
            ]
        )

        results = (
            run_heldout_prediction(
                train_rows,
                test_rows,
            )
        )

        summary.append(
            {
                "regime":
                    heldout,

                "system_rho":
                    results[
                        "system"
                    ]["rho"],

                "closed_rho":
                    results[
                        "closed_loop"
                    ]["rho"],

                "full_rho":
                    results[
                        "full"
                    ]["rho"],

                "full_r2":
                    results[
                        "full"
                    ]["r2_log"],
            }
        )

    print()
    print(
        "===== LEAVE-ONE-REGIME-OUT RESULTS ====="
    )

    print(
        "regime             system_rho  closed_rho  full_rho   full_R2"
    )

    for row in summary:

        alpha, beta = (
            row["regime"]
        )

        print(
            f"({alpha:>3.1f}, {beta:>4.2f})"
            f"        "
            f"{row['system_rho']:>7.3f}"
            f"      "
            f"{row['closed_rho']:>7.3f}"
            f"      "
            f"{row['full_rho']:>7.3f}"
            f"     "
            f"{row['full_r2']:>7.3f}"
        )

    full_rhos = np.asarray(
        [
            r["full_rho"]
            for r in summary
        ]
    )

    closed_rhos = np.asarray(
        [
            r["closed_rho"]
            for r in summary
        ]
    )

    system_rhos = np.asarray(
        [
            r["system_rho"]
            for r in summary
        ]
    )

    print()

    print(
        "Median held-out rho:"
    )

    print(
        "  system-only:",
        np.nanmedian(
            system_rhos
        ),
    )

    print(
        "  closed-loop-only:",
        np.nanmedian(
            closed_rhos
        ),
    )

    print(
        "  full:",
        np.nanmedian(
            full_rhos
        ),
    )

    print()

    print(
        "Fraction of regimes with full rho > 0.5:",
        np.mean(
            full_rhos > 0.5
        ),
    )

    return summary


# ============================================================
# MAIN
# ============================================================

def main():
    print(
        "Closed-Loop Stability Pilot V2"
    )

    print()

    print(
        "Goal:"
    )

    print(
        "Predict long-horizon consequences in feedback regimes "
        "that were NEVER observed during model fitting."
    )

    (
        train_rows,
        test_rows,
        results,
        regime_rho,
        primary_rho,
        primary_r2,
    ) = experiment_primary_heldout()

    summary = (
        experiment_leave_one_regime_out()
    )

    print()
    print(
        "============================================"
    )

    print(
        "FINAL V2 INTERPRETATION"
    )

    print(
        "============================================"
    )

    print()

    print(
        "Primary held-out test:"
    )

    print(
        "  closed-loop-only rho =",
        results[
            "closed_loop"
        ]["rho"],
    )

    print(
        "  full rho =",
        results[
            "full"
        ]["rho"],
    )

    print(
        "  full log-R2 =",
        results[
            "full"
        ]["r2_log"],
    )

    print(
        "  regime-level rho =",
        regime_rho,
    )

    print()

    print(
        "Hardest unseen regime "
        "(alpha=4, beta=0.10):"
    )

    print(
        "  rho =",
        primary_rho,
    )

    print(
        "  log-R2 =",
        primary_r2,
    )

    print()

    full_rhos = np.asarray(
        [
            r["full_rho"]
            for r in summary
        ]
    )

    median_looro = np.nanmedian(
        full_rhos
    )

    print(
        "Leave-one-regime-out median rho =",
        median_looro,
    )

    print()

    # --------------------------------------------------------
    # Pre-specified decision rule.
    # --------------------------------------------------------

    if (
        results[
            "closed_loop"
        ]["rho"] >= 0.5
        and
        regime_rho >= 0.7
        and
        median_looro >= 0.5
    ):

        print(
            "STRONG PASS"
        )

        print()

        print(
            "Short-horizon closed-loop response generalizes "
            "to unseen feedback regimes."
        )

        print(
            "This is substantially stronger evidence than the "
            "within-regime correlations from V1."
        )

        print()

        print(
            "Next experiment:"
        )

        print(
            "held-out ERROR TOPOLOGY prediction "
            "(IID / cluster -> temporal/delayed corruption)."
        )

    elif (
        results[
            "full"
        ]["rho"] >= 0.4
        or
        median_looro >= 0.4
    ):

        print(
            "PARTIAL PASS"
        )

        print()

        print(
            "There is meaningful out-of-regime predictive signal, "
            "but it is not yet robust enough for the main thesis."
        )

        print(
            "Inspect which regimes fail before changing the model."
        )

    else:

        print(
            "FAIL"
        )

        print()

        print(
            "The V1 correlation does not generalize strongly "
            "outside the regimes used to fit the predictor."
        )

        print(
            "We should reconsider whether early closed-loop "
            "sensitivity is genuinely predictive."
        )


if __name__ == "__main__":
    main()