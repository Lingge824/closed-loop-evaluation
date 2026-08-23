import numpy as np

from pilot_v2 import (
    TRAIN_REGIMES,
    HELDOUT_REGIMES,
    TRAIN_SEEDS,
    TEST_SEEDS,
    LinearPredictor,
    safe_spearman,
    r2_score,
)

from pilot_v25 import (
    build_rows,
    log_feature,
)


N_PERMUTATIONS = 500


# ============================================================
# FEATURES
# ============================================================

def make_features(rows, feature_set):
    """
    base:
        Strong privileged baseline:
        early TRUE performance response + remaining horizon.

    dynamics:
        Closed-loop internal state response only.

    impact:
        Magnitude only.

    growth:
        Growth / decay only.

    extended:
        Early performance + closed-loop dynamics.
    """

    X = []

    for row in rows:

        if feature_set == "base":

            features = [
                log_feature(
                    row["p1"]
                ),
                log_feature(
                    row["p3_mean"]
                ),
                log_feature(
                    row["p3_last"]
                ),
                row["lambda_perf"],
                row["remaining_fraction"],
            ]

        elif feature_set == "impact":

            features = [
                log_feature(
                    row["q3_mean"]
                ),
                log_feature(
                    row["c3_mean"]
                ),
            ]

        elif feature_set == "growth":

            features = [
                row["lambda_q"],
                row["lambda_c"],
            ]

        elif feature_set == "dynamics":

            features = [
                log_feature(
                    row["q3_mean"]
                ),
                log_feature(
                    row["c3_mean"]
                ),
                row["lambda_q"],
                row["lambda_c"],
            ]

        elif feature_set == "extended":

            features = [
                # Early observable performance.
                log_feature(
                    row["p1"]
                ),
                log_feature(
                    row["p3_mean"]
                ),
                log_feature(
                    row["p3_last"]
                ),
                row["lambda_perf"],

                # Closed-loop internal state.
                log_feature(
                    row["q3_mean"]
                ),
                log_feature(
                    row["c3_mean"]
                ),
                row["lambda_q"],
                row["lambda_c"],

                # Known amount of horizon remaining.
                row["remaining_fraction"],
            ]

        else:
            raise ValueError(
                feature_set
            )

        X.append(
            features
        )

    return np.asarray(
        X,
        dtype=float,
    )


def make_target(rows):
    """
    Long-horizon mean true-performance deviation.
    """

    y_raw = np.asarray(
        [
            row["late_perf"]
            for row in rows
        ],
        dtype=float,
    )

    y_log = np.log10(
        y_raw + 1e-10
    )

    return y_log, y_raw


# ============================================================
# METRICS
# ============================================================

def raw_predictions(pred_log):
    return np.maximum(
        10 ** pred_log
        - 1e-10,
        0.0,
    )


def evaluate_absolute(
    y_log,
    y_raw,
    pred_log,
):
    pred_raw = raw_predictions(
        pred_log
    )

    return {
        "rho":
            safe_spearman(
                pred_raw,
                y_raw,
            ),

        "r2_log":
            r2_score(
                y_log,
                pred_log,
            ),

        "r2_raw":
            r2_score(
                y_raw,
                pred_raw,
            ),

        "nmae":
            np.mean(
                np.abs(
                    pred_raw
                    - y_raw
                )
            )
            / np.mean(y_raw),
    }


def regime_groups(rows):
    groups = {}

    for i, row in enumerate(rows):

        regime = (
            row["alpha"],
            row["beta"],
        )

        groups.setdefault(
            regime,
            []
        ).append(i)

    return groups


def within_regime_rhos(
    rows,
    prediction,
    target,
):
    """
    Crucial:

    Remove the easy cross-regime ordering effect.

    Ask whether the predictor can rank which evaluator mistakes
    are most dangerous WITHIN the same held-out feedback regime.
    """

    rhos = []

    for regime, idx in regime_groups(
        rows
    ).items():

        idx = np.asarray(
            idx,
            dtype=int,
        )

        rho = safe_spearman(
            prediction[idx],
            target[idx],
        )

        if np.isfinite(rho):
            rhos.append(rho)

    return np.asarray(
        rhos
    )


def print_metrics(
    name,
    metrics,
    within_rhos,
):
    print(
        f"{name:18s} "
        f"rho={metrics['rho']:>6.3f} "
        f"R2_log={metrics['r2_log']:>7.3f} "
        f"R2_raw={metrics['r2_raw']:>7.3f} "
        f"within_med={np.median(within_rhos):>6.3f}"
    )


# ============================================================
# REGULAR MODEL COMPARISON
# ============================================================

def fit_predict(
    train_rows,
    test_rows,
    feature_set,
):
    X_train = make_features(
        train_rows,
        feature_set,
    )

    X_test = make_features(
        test_rows,
        feature_set,
    )

    y_train_log, _ = (
        make_target(
            train_rows
        )
    )

    y_test_log, y_test_raw = (
        make_target(
            test_rows
        )
    )

    model = LinearPredictor()

    model.fit(
        X_train,
        y_train_log,
    )

    pred_log = model.predict(
        X_test
    )

    metrics = evaluate_absolute(
        y_test_log,
        y_test_raw,
        pred_log,
    )

    pred_raw = raw_predictions(
        pred_log
    )

    within = within_regime_rhos(
        test_rows,
        pred_raw,
        y_test_raw,
    )

    return {
        "model": model,
        "pred_log": pred_log,
        "pred_raw": pred_raw,
        "metrics": metrics,
        "within_rhos": within,
    }


# ============================================================
# CORE FALSIFICATION:
#
# CAN DYNAMICS PREDICT WHAT EARLY PERFORMANCE MISSES?
# ============================================================

def residual_test(
    train_rows,
    test_rows,
):
    print()
    print(
        "============================================"
    )
    print(
        "RESIDUAL AMPLIFICATION TEST"
    )
    print(
        "============================================"
    )

    # --------------------------------------------------------
    # Step 1:
    # Fit the strongest early-performance baseline.
    # --------------------------------------------------------

    X_base_train = make_features(
        train_rows,
        "base",
    )

    X_base_test = make_features(
        test_rows,
        "base",
    )

    y_train_log, _ = (
        make_target(
            train_rows
        )
    )

    y_test_log, _ = (
        make_target(
            test_rows
        )
    )

    base_model = LinearPredictor()

    base_model.fit(
        X_base_train,
        y_train_log,
    )

    base_train_pred = (
        base_model.predict(
            X_base_train
        )
    )

    base_test_pred = (
        base_model.predict(
            X_base_test
        )
    )

    # --------------------------------------------------------
    # What early performance FAILS to explain.
    # --------------------------------------------------------

    residual_train = (
        y_train_log
        - base_train_pred
    )

    residual_test_true = (
        y_test_log
        - base_test_pred
    )

    print()
    print(
        "Std of held-out residual target:",
        np.std(
            residual_test_true
        ),
    )

    # --------------------------------------------------------
    # Step 2:
    # Can closed-loop state dynamics predict these residuals?
    # --------------------------------------------------------

    X_dyn_train = make_features(
        train_rows,
        "dynamics",
    )

    X_dyn_test = make_features(
        test_rows,
        "dynamics",
    )

    residual_model = (
        LinearPredictor()
    )

    residual_model.fit(
        X_dyn_train,
        residual_train,
    )

    residual_pred = (
        residual_model.predict(
            X_dyn_test
        )
    )

    residual_rho = (
        safe_spearman(
            residual_pred,
            residual_test_true,
        )
    )

    residual_r2 = r2_score(
        residual_test_true,
        residual_pred,
    )

    within_residual = (
        within_regime_rhos(
            test_rows,
            residual_pred,
            residual_test_true,
        )
    )

    print()
    print(
        "Dynamics -> residual future amplification"
    )

    print(
        "Pooled residual rho:",
        residual_rho,
    )

    print(
        "Residual R2:",
        residual_r2,
    )

    print(
        "Median within-regime residual rho:",
        np.median(
            within_residual
        ),
    )

    print(
        "Minimum within-regime residual rho:",
        np.min(
            within_residual
        ),
    )

    # --------------------------------------------------------
    # Step 3:
    # Add residual prediction back to baseline.
    # --------------------------------------------------------

    combined_test_pred = (
        base_test_pred
        + residual_pred
    )

    _, y_test_raw = (
        make_target(
            test_rows
        )
    )

    base_metrics = (
        evaluate_absolute(
            y_test_log,
            y_test_raw,
            base_test_pred,
        )
    )

    combined_metrics = (
        evaluate_absolute(
            y_test_log,
            y_test_raw,
            combined_test_pred,
        )
    )

    base_raw = raw_predictions(
        base_test_pred
    )

    combined_raw = raw_predictions(
        combined_test_pred
    )

    base_within = (
        within_regime_rhos(
            test_rows,
            base_raw,
            y_test_raw,
        )
    )

    combined_within = (
        within_regime_rhos(
            test_rows,
            combined_raw,
            y_test_raw,
        )
    )

    print()
    print(
        "Absolute prediction after residual correction:"
    )

    print_metrics(
        "early-perf base",
        base_metrics,
        base_within,
    )

    print_metrics(
        "base + dynamics",
        combined_metrics,
        combined_within,
    )

    print()
    print(
        "Delta R2_log:",
        (
            combined_metrics[
                "r2_log"
            ]
            -
            base_metrics[
                "r2_log"
            ]
        ),
    )

    print(
        "Delta pooled rho:",
        (
            combined_metrics[
                "rho"
            ]
            -
            base_metrics[
                "rho"
            ]
        ),
    )

    print(
        "Delta within-regime median rho:",
        (
            np.median(
                combined_within
            )
            -
            np.median(
                base_within
            )
        ),
    )

    return {
        "residual_train":
            residual_train,

        "residual_test":
            residual_test_true,

        "residual_pred":
            residual_pred,

        "residual_rho":
            residual_rho,

        "residual_r2":
            residual_r2,

        "within_residual":
            within_residual,

        "base_metrics":
            base_metrics,

        "combined_metrics":
            combined_metrics,

        "base_within":
            base_within,

        "combined_within":
            combined_within,

        "X_dyn_train":
            X_dyn_train,

        "X_dyn_test":
            X_dyn_test,
    }


# ============================================================
# CORRECT PERMUTATION TEST
# ============================================================

def shuffle_within_regime(
    values,
    rows,
    rng,
):
    """
    Preserve:
        - regime-level target distribution
        - regime mean risk
        - alpha/beta effects

    Destroy:
        - which individual perturbation produced which outcome

    This is the right null for our individual leverage question.
    """

    shuffled = values.copy()

    for _, idx in regime_groups(
        rows
    ).items():

        idx = np.asarray(
            idx,
            dtype=int,
        )

        local = (
            shuffled[idx].copy()
        )

        rng.shuffle(
            local
        )

        shuffled[idx] = local

    return shuffled


def permutation_test(
    train_rows,
    test_rows,
    residual_result,
):
    print()
    print(
        "============================================"
    )
    print(
        "WITHIN-REGIME PERMUTATION NULL"
    )
    print(
        "============================================"
    )

    residual_train = (
        residual_result[
            "residual_train"
        ]
    )

    residual_test_true = (
        residual_result[
            "residual_test"
        ]
    )

    X_train = (
        residual_result[
            "X_dyn_train"
        ]
    )

    X_test = (
        residual_result[
            "X_dyn_test"
        ]
    )

    actual_within = np.median(
        residual_result[
            "within_residual"
        ]
    )

    actual_pooled = (
        residual_result[
            "residual_rho"
        ]
    )

    null_within = []
    null_pooled = []

    rng = np.random.default_rng(
        20260820
    )

    for _ in range(
        N_PERMUTATIONS
    ):

        shuffled_residual = (
            shuffle_within_regime(
                residual_train,
                train_rows,
                rng,
            )
        )

        model = LinearPredictor()

        model.fit(
            X_train,
            shuffled_residual,
        )

        pred = model.predict(
            X_test
        )

        pooled_rho = (
            safe_spearman(
                pred,
                residual_test_true,
            )
        )

        within = (
            within_regime_rhos(
                test_rows,
                pred,
                residual_test_true,
            )
        )

        null_pooled.append(
            pooled_rho
        )

        null_within.append(
            np.median(
                within
            )
        )

    null_pooled = np.asarray(
        null_pooled
    )

    null_within = np.asarray(
        null_within
    )

    p_within = (
        1
        + np.sum(
            null_within
            >= actual_within
        )
    ) / (
        N_PERMUTATIONS
        + 1
    )

    p_pooled = (
        1
        + np.sum(
            null_pooled
            >= actual_pooled
        )
    ) / (
        N_PERMUTATIONS
        + 1
    )

    print()
    print(
        "Actual pooled residual rho:",
        actual_pooled,
    )

    print(
        "Permutation null pooled mean:",
        np.mean(
            null_pooled
        ),
    )

    print(
        "Permutation null pooled 95%:",
        np.quantile(
            null_pooled,
            [
                0.025,
                0.975,
            ],
        ),
    )

    print(
        "Permutation p-value pooled:",
        p_pooled,
    )

    print()

    print(
        "Actual median within-regime residual rho:",
        actual_within,
    )

    print(
        "Permutation null within mean:",
        np.mean(
            null_within
        ),
    )

    print(
        "Permutation null within 95%:",
        np.quantile(
            null_within,
            [
                0.025,
                0.975,
            ],
        ),
    )

    print(
        "Permutation p-value within:",
        p_within,
    )

    return {
        "p_within":
            p_within,

        "p_pooled":
            p_pooled,

        "null_within":
            null_within,

        "null_pooled":
            null_pooled,
    }


# ============================================================
# DIRECT FEATURE COMPARISON
# ============================================================

def direct_models(
    train_rows,
    test_rows,
):
    print()
    print(
        "============================================"
    )
    print(
        "DIRECT HELD-OUT MODELS"
    )
    print(
        "============================================"
    )

    results = {}

    print()
    print(
        "model               rho    R2_log  R2_raw  within_med"
    )

    for feature_set in [
        "base",
        "impact",
        "growth",
        "dynamics",
        "extended",
    ]:

        result = fit_predict(
            train_rows,
            test_rows,
            feature_set,
        )

        results[
            feature_set
        ] = result

        print_metrics(
            feature_set,
            result[
                "metrics"
            ],
            result[
                "within_rhos"
            ],
        )

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    print(
        "Closed-Loop Stability Pilot V2.6"
    )

    print()
    print(
        "Question:"
    )

    print(
        "After accounting for early TRUE performance deviation, "
        "does closed-loop state dynamics still predict "
        "additional long-horizon amplification?"
    )

    train_rows = build_rows(
        TRAIN_REGIMES,
        TRAIN_SEEDS,
        "v26 train",
    )

    test_rows = build_rows(
        HELDOUT_REGIMES,
        TEST_SEEDS,
        "v26 heldout",
    )

    direct = direct_models(
        train_rows,
        test_rows,
    )

    residual = residual_test(
        train_rows,
        test_rows,
    )

    permutation = (
        permutation_test(
            train_rows,
            test_rows,
            residual,
        )
    )

    print()
    print(
        "============================================"
    )
    print(
        "V2.6 DECISION"
    )
    print(
        "============================================"
    )

    residual_within = np.median(
        residual[
            "within_residual"
        ]
    )

    delta_r2 = (
        residual[
            "combined_metrics"
        ]["r2_log"]
        -
        residual[
            "base_metrics"
        ]["r2_log"]
    )

    delta_within = (
        np.median(
            residual[
                "combined_within"
            ]
        )
        -
        np.median(
            residual[
                "base_within"
            ]
        )
    )

    print()
    print(
        "Residual within-regime rho:",
        residual_within,
    )

    print(
        "Delta R2_log beyond early performance:",
        delta_r2,
    )

    print(
        "Delta within-regime rho:",
        delta_within,
    )

    print(
        "Permutation p-value:",
        permutation[
            "p_within"
        ],
    )

    print()

    if (
        residual_within >= 0.30
        and
        permutation[
            "p_within"
        ] <= 0.01
        and
        delta_r2 >= 0.01
    ):

        print(
            "STABILITY SIGNAL SURVIVES"
        )

        print()

        print(
            "Closed-loop state response predicts future amplification "
            "even after conditioning on early true-performance damage."
        )

        print(
            "The stability framing remains defensible."
        )

        print(
            "Proceed to V3: unseen corruption processes."
        )

    elif (
        permutation[
            "p_within"
        ] <= 0.05
        and
        residual_within > 0.10
    ):

        print(
            "LEVERAGE DOMINATES, DYNAMICS ADD MODEST SIGNAL"
        )

        print()

        print(
            "The phenomenon is real, but most predictive power comes "
            "from early perturbation magnitude rather than amplification."
        )

        print(
            "Reframe around closed-loop evaluation leverage "
            "rather than strong instability language."
        )

    else:

        print(
            "MOSTLY PERSISTENCE"
        )

        print()

        print(
            "Once early damage is known, dynamics add little "
            "reliable predictive information."
        )

        print(
            "Do not build the paper around stability amplification."
        )


if __name__ == "__main__":
    main()