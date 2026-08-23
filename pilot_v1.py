from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


# ============================================================
# CONFIG
# ============================================================

@dataclass(frozen=True)
class Config:
    K: int = 8
    N: int = 40
    T: int = 40

    tau: float = 0.15

    # Selection sensitivity:
    # how strongly curriculum reacts to estimated learning value.
    alpha: float = 2.0

    # Learner plasticity / adaptation rate.
    beta: float = 0.05

    # How quickly curriculum moves toward the new target distribution.
    curriculum_rate: float = 0.50

    error_rate: float = 0.10
    exploration: float = 0.05


# ============================================================
# BASIC MATH
# ============================================================

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def success_probs(capability, difficulty, tau):
    return sigmoid((capability - difficulty) / tau)


def learning_value_from_prob(p):
    """
    True learning value.

    Maximum at p = 0.5.
    Low when a task is trivial or nearly impossible.
    """
    return 4.0 * p * (1.0 - p)


def reference_performance(capability, difficulty, tau):
    """
    Frozen reference distribution:
    uniform over task families.

    It NEVER enters the self-improvement loop.
    """
    return success_probs(
        capability,
        difficulty,
        tau,
    ).mean()


def js_divergence(p, q):
    eps = 1e-12

    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)

    p /= p.sum()
    q /= q.sum()

    m = 0.5 * (p + q)

    return 0.5 * (
        np.sum(p * np.log(p / m))
        +
        np.sum(q * np.log(q / m))
    )


def sqrt_js(p, q):
    """
    sqrt(JS) behaves like a distance and is easier to interpret
    as perturbation magnitude.
    """
    return np.sqrt(max(js_divergence(p, q), 0.0))


# ============================================================
# COMMON RANDOM NUMBERS
# ============================================================

def make_random_stream(cfg, seed):
    """
    Clean and perturbed trajectories receive exactly the same
    underlying randomness.

    Only evaluator interventions differ.
    """
    rng = np.random.default_rng(seed)

    return {
        "task": rng.random((cfg.T, cfg.N)),
        "outcome": rng.random((cfg.T, cfg.N)),
        "flip": rng.random((cfg.T, cfg.N)),
    }


def sample_task_families(q, uniform_draws):
    cdf = np.cumsum(q)

    ids = np.searchsorted(
        cdf,
        uniform_draws,
        side="right",
    )

    return np.clip(
        ids,
        0,
        len(q) - 1,
    )


# ============================================================
# MATCHED ERROR SELECTION
# ============================================================

def choose_balanced_flip_indices(
    y_true,
    priority,
    m,
):
    """
    Try to make approximately half of the errors:

        success -> failure

    and half:

        failure -> success

    so IID and clustered evaluators have similar confusion structure,
    not merely identical total accuracy.
    """

    selected = []

    n_positive = m // 2
    n_negative = m - n_positive

    for label, n_wanted in [
        (1, n_positive),
        (0, n_negative),
    ]:
        candidates = np.where(
            y_true == label
        )[0]

        if len(candidates) == 0:
            continue

        order = candidates[
            np.argsort(priority[candidates])
        ]

        selected.extend(
            order[:n_wanted].tolist()
        )

    # If one label type did not have enough examples,
    # fill remaining slots regardless of label.
    if len(selected) < m:

        selected_set = set(selected)

        remaining = np.array(
            [
                i
                for i in range(len(y_true))
                if i not in selected_set
            ],
            dtype=int,
        )

        order = remaining[
            np.argsort(priority[remaining])
        ]

        need = m - len(selected)

        selected.extend(
            order[:need].tolist()
        )

    return np.asarray(
        selected[:m],
        dtype=int,
    )


def corrupt_iid(
    y_true,
    flip_randomness,
    cfg,
):
    """
    Exactly 10% errors, approximately balanced by direction.
    """

    y_hat = y_true.copy()

    m = int(
        round(cfg.N * cfg.error_rate)
    )

    idx = choose_balanced_flip_indices(
        y_true,
        flip_randomness,
        m,
    )

    y_hat[idx] = 1 - y_hat[idx]

    return y_hat


def corrupt_clustered(
    y_true,
    task_ids,
    flip_randomness,
    cfg,
    target_families,
):
    """
    Same number of errors and similar error directions as IID,
    but errors are preferentially concentrated in fixed task families.

    IMPORTANT:
    target families do NOT adapt based on current agent state.
    """

    y_hat = y_true.copy()

    m = int(
        round(cfg.N * cfg.error_rate)
    )

    is_target = np.isin(
        task_ids,
        target_families,
    )

    # Smaller priority gets selected first.
    priority = flip_randomness.copy()

    # Strong fixed preference for target families.
    priority += (
        (~is_target).astype(float)
        * 10.0
    )

    idx = choose_balanced_flip_indices(
        y_true,
        priority,
        m,
    )

    y_hat[idx] = 1 - y_hat[idx]

    return y_hat


# ============================================================
# ESTIMATION + CLOSED-LOOP UPDATE
# ============================================================

def estimate_family_success(
    task_ids,
    y_hat,
    K,
):
    counts = np.bincount(
        task_ids,
        minlength=K,
    )

    successes = np.bincount(
        task_ids,
        weights=y_hat,
        minlength=K,
    )

    # Beta(1,1) smoothing.
    return (
        successes + 1.0
    ) / (
        counts + 2.0
    )


def update_curriculum(
    q,
    p_hat,
    cfg,
):
    """
    ALIGNED curriculum.

    Estimated learning value:
        4 p_hat (1 - p_hat)

    The curriculum now tries to select tasks around the
    estimated capability frontier.

    This matches the learner dynamics below.
    """

    estimated_learning_value = (
        learning_value_from_prob(
            p_hat
        )
    )

    logits = (
        cfg.alpha
        * estimated_learning_value
    )

    logits -= logits.max()

    target_q = np.exp(logits)
    target_q /= target_q.sum()

    # Keep explicit exploration.
    target_q = (
        (1.0 - cfg.exploration)
        * target_q
        +
        cfg.exploration
        * np.ones(cfg.K)
        / cfg.K
    )

    # Smooth curriculum update.
    q_new = (
        (1.0 - cfg.curriculum_rate)
        * q
        +
        cfg.curriculum_rate
        * target_q
    )

    q_new /= q_new.sum()

    return q_new


def update_capability(
    capability,
    q_new,
    difficulty,
    cfg,
):
    """
    True learner dynamics.

    Learning is strongest around the true capability frontier.
    """

    p_true = success_probs(
        capability,
        difficulty,
        cfg.tau,
    )

    true_learning_value = (
        learning_value_from_prob(
            p_true
        )
    )

    delta = (
        cfg.beta
        * q_new
        * true_learning_value
        * (1.0 - capability)
    )

    return np.clip(
        capability + delta,
        0.0,
        1.0,
    )


# ============================================================
# TRAJECTORY
# ============================================================

def run_trajectory(
    cfg,
    stream,
    evaluator="clean",
    impulse=None,
):
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

    # Fixed error-topology regions.
    #
    # With our initial capability/difficulty:
    #
    # families 2 and 3 are close to p=0.5
    # and therefore high learning-value families.
    #
    # families 6 and 7 are much lower-learning-value.
    high_leverage_families = np.array(
        [2, 3],
        dtype=int,
    )

    low_leverage_families = np.array(
        [6, 7],
        dtype=int,
    )

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

    for t in range(cfg.T):

        # ----------------------------------------------------
        # Sample current tasks
        # ----------------------------------------------------

        task_ids = sample_task_families(
            q,
            stream["task"][t],
        )

        # ----------------------------------------------------
        # True agent behavior
        # ----------------------------------------------------

        p_family = success_probs(
            capability,
            difficulty,
            cfg.tau,
        )

        p_tasks = p_family[task_ids]

        y_true = (
            stream["outcome"][t]
            < p_tasks
        ).astype(int)

        # ----------------------------------------------------
        # Evaluator
        # ----------------------------------------------------

        if evaluator == "clean":

            y_hat = y_true.copy()

        elif evaluator == "iid":

            y_hat = corrupt_iid(
                y_true,
                stream["flip"][t],
                cfg,
            )

        elif evaluator == "cluster_high":

            y_hat = corrupt_clustered(
                y_true,
                task_ids,
                stream["flip"][t],
                cfg,
                high_leverage_families,
            )

        elif evaluator == "cluster_low":

            y_hat = corrupt_clustered(
                y_true,
                task_ids,
                stream["flip"][t],
                cfg,
                low_leverage_families,
            )

        elif evaluator == "impulse":

            y_hat = y_true.copy()

            impulse_round, impulse_idx = impulse

            if t == impulse_round:

                y_hat[impulse_idx] = (
                    1 - y_hat[impulse_idx]
                )

        else:
            raise ValueError(
                f"Unknown evaluator: {evaluator}"
            )

        accuracy_hist.append(
            np.mean(
                y_hat == y_true
            )
        )

        # ----------------------------------------------------
        # Feedback loop
        # ----------------------------------------------------

        p_hat = estimate_family_success(
            task_ids,
            y_hat,
            cfg.K,
        )

        q_new = update_curriculum(
            q,
            p_hat,
            cfg,
        )

        capability_new = update_capability(
            capability,
            q_new,
            difficulty,
            cfg,
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
        "accuracy": np.asarray(
            accuracy_hist
        ),
    }


# ============================================================
# EXPERIMENT 1
#
# SAME ACCURACY, DIFFERENT FUTURES
# ============================================================

def experiment_matched_accuracy(
    cfg,
    num_seeds=50,
):
    modes = [
        "clean",
        "iid",
        "cluster_low",
        "cluster_high",
    ]

    perf = {
        mode: []
        for mode in modes
    }

    accuracy = {
        mode: []
        for mode in modes
    }

    final_q = {
        mode: []
        for mode in modes
    }

    final_c = {
        mode: []
        for mode in modes
    }

    for seed in range(num_seeds):

        stream = make_random_stream(
            cfg,
            seed,
        )

        for mode in modes:

            result = run_trajectory(
                cfg,
                stream,
                evaluator=mode,
            )

            perf[mode].append(
                result["perf"]
            )

            accuracy[mode].append(
                result["accuracy"].mean()
            )

            final_q[mode].append(
                result["q"][-1]
            )

            final_c[mode].append(
                result["c"][-1]
            )

    for mode in modes:

        perf[mode] = np.asarray(
            perf[mode]
        )

        final_q[mode] = np.asarray(
            final_q[mode]
        )

        final_c[mode] = np.asarray(
            final_c[mode]
        )

    print()
    print(
        "===== MATCHED ACCURACY V1 ====="
    )

    for mode in modes:

        print(
            f"{mode:14s} "
            f"accuracy="
            f"{np.mean(accuracy[mode]):.3f} "
            f"final_true_perf="
            f"{perf[mode][:, -1].mean():.5f} "
            f"+/- "
            f"{perf[mode][:, -1].std():.5f}"
        )

    clean_final_perf = (
        perf["clean"][:, -1]
    )

    print()
    print(
        "Deviation relative to clean:"
    )

    for mode in [
        "iid",
        "cluster_low",
        "cluster_high",
    ]:

        signed = (
            clean_final_perf
            - perf[mode][:, -1]
        )

        print()
        print(mode)

        print(
            "  mean signed harm:",
            signed.mean(),
        )

        print(
            "  mean |performance deviation|:",
            np.abs(signed).mean(),
        )

        q_dev = []

        c_dev = []

        for i in range(num_seeds):

            q_dev.append(
                sqrt_js(
                    final_q["clean"][i],
                    final_q[mode][i],
                )
            )

            c_dev.append(
                np.linalg.norm(
                    final_c["clean"][i]
                    - final_c[mode][i]
                )
                / np.sqrt(cfg.K)
            )

        print(
            "  mean final curriculum deviation:",
            np.mean(q_dev),
        )

        print(
            "  mean final capability deviation:",
            np.mean(c_dev),
        )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    rounds = np.arange(
        cfg.T + 1
    )

    plt.figure(
        figsize=(8, 5)
    )

    for mode in modes:

        mean = perf[mode].mean(
            axis=0
        )

        std = perf[mode].std(
            axis=0
        )

        plt.plot(
            rounds,
            mean,
            label=mode,
        )

        plt.fill_between(
            rounds,
            mean - std,
            mean + std,
            alpha=0.12,
        )

    plt.xlabel(
        "Adaptation round"
    )

    plt.ylabel(
        "Frozen-oracle true performance"
    )

    plt.title(
        "Matched Accuracy, Different Error Topologies"
    )

    plt.legend()
    plt.tight_layout()

    Path("results").mkdir(
        exist_ok=True
    )

    plt.savefig(
        "results/v1_matched_accuracy.png",
        dpi=180,
    )

    plt.close()


# ============================================================
# IMPULSE METRICS
# ============================================================

def log_growth_rate(
    deviations,
    horizon=5,
):
    """
    Estimate early perturbation growth / decay:

        slope(log d_h)

    Positive:
        perturbation tends to grow.

    Negative:
        perturbation tends to decay.
    """

    n = min(
        horizon,
        len(deviations),
    )

    if n < 2:
        return np.nan

    x = np.arange(
        1,
        n + 1,
        dtype=float,
    )

    y = np.log(
        np.maximum(
            deviations[:n],
            1e-12,
        )
    )

    return np.polyfit(
        x,
        y,
        1,
    )[0]


def impulse_metrics(
    clean,
    perturbed,
    t0,
    cfg,
):
    max_horizon = (
        cfg.T - t0
    )

    q_dev = []

    c_dev = []

    perf_dev = []

    for h in range(
        1,
        max_horizon + 1,
    ):

        q_dev.append(
            sqrt_js(
                clean["q"][t0 + h],
                perturbed["q"][t0 + h],
            )
        )

        c_dev.append(
            np.linalg.norm(
                clean["c"][t0 + h]
                - perturbed["c"][t0 + h]
            )
            / np.sqrt(cfg.K)
        )

        perf_dev.append(
            abs(
                clean["perf"][t0 + h]
                - perturbed["perf"][t0 + h]
            )
        )

    q_dev = np.asarray(q_dev)
    c_dev = np.asarray(c_dev)
    perf_dev = np.asarray(perf_dev)

    # --------------------------------------------------------
    # Predictor 1:
    # finite-horizon sensitivity / impact
    # --------------------------------------------------------

    early_impact = np.sum(
        q_dev[:3]
    )

    # --------------------------------------------------------
    # Predictor 2:
    # actual early growth / decay
    # --------------------------------------------------------

    lambda_q = log_growth_rate(
        q_dev,
        horizon=5,
    )

    lambda_c = log_growth_rate(
        c_dev,
        horizon=5,
    )

    # --------------------------------------------------------
    # Long-horizon outcomes
    # --------------------------------------------------------

    late_start = min(
        5,
        len(perf_dev),
    )

    late_perf_area = np.sum(
        perf_dev[late_start:]
    )

    late_q_area = np.sum(
        q_dev[late_start:]
    )

    late_c_area = np.sum(
        c_dev[late_start:]
    )

    final_signed_harm = (
        clean["perf"][-1]
        - perturbed["perf"][-1]
    )

    final_abs_harm = abs(
        final_signed_harm
    )

    return {
        "early_impact": early_impact,
        "lambda_q": lambda_q,
        "lambda_c": lambda_c,

        "late_perf_area": late_perf_area,
        "late_q_area": late_q_area,
        "late_c_area": late_c_area,

        "final_signed_harm": final_signed_harm,
        "final_abs_harm": final_abs_harm,

        "q_curve": q_dev,
        "c_curve": c_dev,
        "perf_curve": perf_dev,
    }


def safe_spearman(x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    valid = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

    x = x[valid]
    y = y[valid]

    if len(x) < 3:
        return np.nan, np.nan

    if np.std(x) < 1e-15:
        return np.nan, np.nan

    if np.std(y) < 1e-15:
        return np.nan, np.nan

    result = spearmanr(
        x,
        y,
    )

    return (
        result.statistic,
        result.pvalue,
    )


def run_impulse_set(
    cfg,
    seeds,
    impulse_rounds,
    indices,
):
    records = []

    for seed in seeds:

        stream = make_random_stream(
            cfg,
            10_000 + seed,
        )

        clean = run_trajectory(
            cfg,
            stream,
            evaluator="clean",
        )

        for t0 in impulse_rounds:

            for idx in indices:

                perturbed = run_trajectory(
                    cfg,
                    stream,
                    evaluator="impulse",
                    impulse=(
                        t0,
                        idx,
                    ),
                )

                record = impulse_metrics(
                    clean,
                    perturbed,
                    t0,
                    cfg,
                )

                record["seed"] = seed
                record["t0"] = t0
                record["idx"] = idx

                records.append(
                    record
                )

    return records


# ============================================================
# EXPERIMENT 2
#
# IMPULSE RESPONSE
# ============================================================

def experiment_impulse(cfg):
    records = run_impulse_set(
        cfg,
        seeds=range(40),
        impulse_rounds=[
            5,
            8,
            12,
            16,
            20,
        ],
        indices=[
            0,
            7,
            18,
            31,
        ],
    )

    early_impact = np.asarray(
        [
            r["early_impact"]
            for r in records
        ]
    )

    lambda_q = np.asarray(
        [
            r["lambda_q"]
            for r in records
        ]
    )

    lambda_c = np.asarray(
        [
            r["lambda_c"]
            for r in records
        ]
    )

    late_perf = np.asarray(
        [
            r["late_perf_area"]
            for r in records
        ]
    )

    late_q = np.asarray(
        [
            r["late_q_area"]
            for r in records
        ]
    )

    late_c = np.asarray(
        [
            r["late_c_area"]
            for r in records
        ]
    )

    final_harm = np.asarray(
        [
            r["final_signed_harm"]
            for r in records
        ]
    )

    # --------------------------------------------------------
    # Correlations
    # --------------------------------------------------------

    rho_impact_perf, p_impact_perf = (
        safe_spearman(
            early_impact,
            late_perf,
        )
    )

    rho_lambda_q_perf, p_lambda_q_perf = (
        safe_spearman(
            lambda_q,
            late_perf,
        )
    )

    rho_lambda_q_q, p_lambda_q_q = (
        safe_spearman(
            lambda_q,
            late_q,
        )
    )

    rho_lambda_c_perf, p_lambda_c_perf = (
        safe_spearman(
            lambda_c,
            late_perf,
        )
    )

    print()
    print(
        "===== IMPULSE RESPONSE V1 ====="
    )

    print(
        "Number of interventions:",
        len(records),
    )

    print()

    print(
        "Finite-horizon sensitivity:"
    )

    print(
        f"rho(early impact, late performance deviation) "
        f"= {rho_impact_perf:.3f}, "
        f"p={p_impact_perf:.3e}"
    )

    print()

    print(
        "Growth / decay predictors:"
    )

    print(
        f"rho(lambda_q, late performance deviation) "
        f"= {rho_lambda_q_perf:.3f}, "
        f"p={p_lambda_q_perf:.3e}"
    )

    print(
        f"rho(lambda_q, late curriculum deviation) "
        f"= {rho_lambda_q_q:.3f}, "
        f"p={p_lambda_q_q:.3e}"
    )

    print(
        f"rho(lambda_c, late performance deviation) "
        f"= {rho_lambda_c_perf:.3f}, "
        f"p={p_lambda_c_perf:.3e}"
    )

    print()

    print(
        "Mean lambda_q:",
        np.nanmean(lambda_q),
    )

    print(
        "Mean lambda_c:",
        np.nanmean(lambda_c),
    )

    print(
        "Fraction harmful:",
        np.mean(
            final_harm > 0
        ),
    )

    print(
        "Fraction beneficial:",
        np.mean(
            final_harm < 0
        ),
    )

    # --------------------------------------------------------
    # Scatter:
    # early finite-horizon sensitivity
    # --------------------------------------------------------

    plt.figure(
        figsize=(7, 5)
    )

    plt.scatter(
        early_impact,
        late_perf,
        s=15,
        alpha=0.4,
    )

    plt.xlabel(
        "Early finite-horizon sensitivity"
    )

    plt.ylabel(
        "Late cumulative true-performance deviation"
    )

    plt.title(
        f"Early Sensitivity vs Long-Horizon Deviation "
        f"(rho={rho_impact_perf:.2f})"
    )

    plt.tight_layout()

    plt.savefig(
        "results/v1_impact_vs_late_perf.png",
        dpi=180,
    )

    plt.close()

    # --------------------------------------------------------
    # Scatter:
    # capability growth rate
    # --------------------------------------------------------

    plt.figure(
        figsize=(7, 5)
    )

    plt.scatter(
        lambda_c,
        late_perf,
        s=15,
        alpha=0.4,
    )

    plt.xlabel(
        "Early capability-deviation growth rate"
    )

    plt.ylabel(
        "Late cumulative true-performance deviation"
    )

    plt.title(
        f"Early Growth vs Long-Horizon Deviation "
        f"(rho={rho_lambda_c_perf:.2f})"
    )

    plt.tight_layout()

    plt.savefig(
        "results/v1_growth_vs_late_perf.png",
        dpi=180,
    )

    plt.close()

    # --------------------------------------------------------
    # Mean impulse-response curves
    # --------------------------------------------------------

    common_horizon = 10

    q_curves = np.asarray(
        [
            r["q_curve"][:common_horizon]
            for r in records
            if len(r["q_curve"]) >= common_horizon
        ]
    )

    c_curves = np.asarray(
        [
            r["c_curve"][:common_horizon]
            for r in records
            if len(r["c_curve"]) >= common_horizon
        ]
    )

    h = np.arange(
        1,
        common_horizon + 1,
    )

    plt.figure(
        figsize=(7, 5)
    )

    plt.plot(
        h,
        q_curves.mean(axis=0),
    )

    plt.xlabel(
        "Rounds after one evaluator error"
    )

    plt.ylabel(
        "Mean curriculum deviation"
    )

    plt.title(
        "Mean Curriculum Impulse Response"
    )

    plt.tight_layout()

    plt.savefig(
        "results/v1_q_impulse_response.png",
        dpi=180,
    )

    plt.close()

    plt.figure(
        figsize=(7, 5)
    )

    plt.plot(
        h,
        c_curves.mean(axis=0),
    )

    plt.xlabel(
        "Rounds after one evaluator error"
    )

    plt.ylabel(
        "Mean capability-state deviation"
    )

    plt.title(
        "Mean Capability Impulse Response"
    )

    plt.tight_layout()

    plt.savefig(
        "results/v1_c_impulse_response.png",
        dpi=180,
    )

    plt.close()

    return {
        "rho_impact_perf":
            rho_impact_perf,

        "rho_lambda_q_perf":
            rho_lambda_q_perf,

        "rho_lambda_q_q":
            rho_lambda_q_q,

        "rho_lambda_c_perf":
            rho_lambda_c_perf,
    }


# ============================================================
# EXPERIMENT 3
#
# PRE-SPECIFIED STABILITY GRID
#
# THIS IS NOT PARAMETER TUNING.
#
# We run the whole grid and inspect whether sensitivity changes
# systematically with selection strength and learner plasticity.
# ============================================================

def experiment_stability_grid(
    base_cfg,
):
    alphas = [
        0.5,
        1.0,
        2.0,
        4.0,
    ]

    betas = [
        0.02,
        0.05,
        0.10,
    ]

    mean_impact = np.zeros(
        (
            len(betas),
            len(alphas),
        )
    )

    mean_lambda_c = np.zeros_like(
        mean_impact
    )

    mean_late_perf = np.zeros_like(
        mean_impact
    )

    within_setting_rho = np.zeros_like(
        mean_impact
    )

    print()
    print(
        "===== STABILITY GRID ====="
    )

    for bi, beta in enumerate(betas):

        for ai, alpha in enumerate(alphas):

            cfg = replace(
                base_cfg,
                alpha=alpha,
                beta=beta,
            )

            records = run_impulse_set(
                cfg,
                seeds=range(10),
                impulse_rounds=[
                    5,
                    12,
                    20,
                ],
                indices=[
                    0,
                    13,
                    27,
                ],
            )

            impact = np.asarray(
                [
                    r["early_impact"]
                    for r in records
                ]
            )

            lambda_c = np.asarray(
                [
                    r["lambda_c"]
                    for r in records
                ]
            )

            late_perf = np.asarray(
                [
                    r["late_perf_area"]
                    for r in records
                ]
            )

            rho, _ = safe_spearman(
                impact,
                late_perf,
            )

            mean_impact[
                bi,
                ai,
            ] = np.nanmean(
                impact
            )

            mean_lambda_c[
                bi,
                ai,
            ] = np.nanmean(
                lambda_c
            )

            mean_late_perf[
                bi,
                ai,
            ] = np.nanmean(
                late_perf
            )

            within_setting_rho[
                bi,
                ai,
            ] = rho

            print(
                f"alpha={alpha:<4} "
                f"beta={beta:<5} "
                f"mean_impact="
                f"{mean_impact[bi, ai]:.6f} "
                f"mean_lambda_c="
                f"{mean_lambda_c[bi, ai]:.3f} "
                f"late_perf="
                f"{mean_late_perf[bi, ai]:.6f} "
                f"rho="
                f"{within_setting_rho[bi, ai]:.3f}"
            )

    # --------------------------------------------------------
    # Across-setting relationship
    # --------------------------------------------------------

    rho_grid, p_grid = safe_spearman(
        mean_impact.ravel(),
        mean_late_perf.ravel(),
    )

    rho_lambda_grid, p_lambda_grid = (
        safe_spearman(
            mean_lambda_c.ravel(),
            mean_late_perf.ravel(),
        )
    )

    print()
    print(
        "Across parameter settings:"
    )

    print(
        f"rho(mean early sensitivity, mean late deviation) "
        f"= {rho_grid:.3f}, "
        f"p={p_grid:.3e}"
    )

    print(
        f"rho(mean lambda_c, mean late deviation) "
        f"= {rho_lambda_grid:.3f}, "
        f"p={p_lambda_grid:.3e}"
    )

    # --------------------------------------------------------
    # Heatmap: late performance deviation
    # --------------------------------------------------------

    plt.figure(
        figsize=(7, 5)
    )

    plt.imshow(
        mean_late_perf,
        aspect="auto",
        origin="lower",
    )

    plt.xticks(
        np.arange(len(alphas)),
        alphas,
    )

    plt.yticks(
        np.arange(len(betas)),
        betas,
    )

    plt.xlabel(
        "Selection strength alpha"
    )

    plt.ylabel(
        "Learner adaptation rate beta"
    )

    plt.title(
        "Long-Horizon Deviation Across Feedback Settings"
    )

    plt.colorbar(
        label="Mean late true-performance deviation"
    )

    plt.tight_layout()

    plt.savefig(
        "results/v1_stability_grid_late_perf.png",
        dpi=180,
    )

    plt.close()

    # --------------------------------------------------------
    # Heatmap: early sensitivity
    # --------------------------------------------------------

    plt.figure(
        figsize=(7, 5)
    )

    plt.imshow(
        mean_impact,
        aspect="auto",
        origin="lower",
    )

    plt.xticks(
        np.arange(len(alphas)),
        alphas,
    )

    plt.yticks(
        np.arange(len(betas)),
        betas,
    )

    plt.xlabel(
        "Selection strength alpha"
    )

    plt.ylabel(
        "Learner adaptation rate beta"
    )

    plt.title(
        "Early Perturbation Sensitivity"
    )

    plt.colorbar(
        label="Mean early sensitivity"
    )

    plt.tight_layout()

    plt.savefig(
        "results/v1_stability_grid_early_impact.png",
        dpi=180,
    )

    plt.close()

    return {
        "rho_grid": rho_grid,
        "rho_lambda_grid":
            rho_lambda_grid,
        "within_setting_rho":
            within_setting_rho,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    Path(
        "results"
    ).mkdir(
        exist_ok=True
    )

    cfg = Config()

    print(cfg)

    print()
    print(
        "V1 changes:"
    )

    print(
        "1. Curriculum objective now matches learner objective."
    )

    print(
        "2. Structured errors have fixed topology."
    )

    print(
        "3. Error count and approximate FP/FN balance are matched."
    )

    print(
        "4. We distinguish initial impact from perturbation growth."
    )

    experiment_matched_accuracy(
        cfg,
        num_seeds=50,
    )

    impulse = experiment_impulse(
        cfg
    )

    grid = experiment_stability_grid(
        cfg
    )

    print()
    print(
        "===== V1 INTERPRETATION ====="
    )

    print()

    print(
        "Primary question 1:"
    )

    print(
        "Do matched-accuracy evaluators produce materially "
        "different long-run trajectories?"
    )

    print()

    print(
        "Primary question 2:"
    )

    print(
        "Does early finite-horizon sensitivity predict "
        "late true-performance deviation?"
    )

    print(
        "rho =",
        impulse[
            "rho_impact_perf"
        ],
    )

    print()

    print(
        "Primary question 3:"
    )

    print(
        "Does this relationship persist across "
        "pre-specified alpha/beta settings?"
    )

    print(
        "Across-grid rho =",
        grid[
            "rho_grid"
        ],
    )

    print()

    if (
        impulse["rho_impact_perf"] >= 0.5
        and
        grid["rho_grid"] >= 0.5
    ):

        print(
            "PROMISING:"
        )

        print(
            "The aligned system shows substantial evidence "
            "that short-horizon closed-loop sensitivity "
            "predicts long-horizon deviation."
        )

        print(
            "Next step should be a HELD-OUT prediction test, "
            "not more parameter tuning."
        )

    elif (
        impulse["rho_impact_perf"] >= 0.3
        or
        grid["rho_grid"] >= 0.3
    ):

        print(
            "MIXED:"
        )

        print(
            "There is some signal, but it is not yet strong "
            "enough to support the core thesis."
        )

        print(
            "Inspect impulse-response shape and individual "
            "parameter settings before continuing."
        )

    else:

        print(
            "WEAK:"
        )

        print(
            "Even after fixing the known v0 mismatch, "
            "early closed-loop sensitivity has weak predictive value."
        )

        print(
            "At that point we should seriously reconsider "
            "the predictive-stability thesis."
        )


if __name__ == "__main__":
    main()