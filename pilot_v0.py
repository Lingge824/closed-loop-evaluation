from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


# ============================================================
# CONFIG
# ============================================================

@dataclass
class Config:
    K: int = 8
    N: int = 40              # tasks per round; 4 flips = exactly 10%
    T: int = 40              # number of adaptation rounds
    tau: float = 0.15        # smoothness of success probability
    alpha: float = 2.0       # curriculum / selection strength
    beta: float = 0.05       # learner adaptation rate
    error_rate: float = 0.10
    exploration: float = 0.05


# ============================================================
# BASIC UTILITIES
# ============================================================

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def success_probs(capability, difficulty, tau):
    return sigmoid((capability - difficulty) / tau)


def reference_performance(capability, difficulty, tau):
    """
    Frozen reference distribution = uniform over all task families.

    This distribution NEVER enters the self-improvement loop.
    It is only used to measure true final capability.
    """
    return success_probs(
        capability,
        difficulty,
        tau,
    ).mean()


def js_divergence(p, q):
    """
    Jensen-Shannon divergence between two probability distributions.
    """
    eps = 1e-12

    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)

    p /= p.sum()
    q /= q.sum()

    m = 0.5 * (p + q)

    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))

    return 0.5 * (kl_pm + kl_qm)


# ============================================================
# COMMON RANDOM NUMBERS
# ============================================================

def make_random_stream(cfg, seed):
    """
    Pre-generate all randomness.

    Clean and corrupted trajectories use exactly the same random
    numbers. The ONLY intended difference is evaluator corruption.

    This gives us a paired counterfactual comparison.
    """
    rng = np.random.default_rng(seed)

    return {
        "task": rng.random((cfg.T, cfg.N)),
        "outcome": rng.random((cfg.T, cfg.N)),
        "flip": rng.random((cfg.T, cfg.N)),
    }


def sample_task_families(q, uniform_draws):
    """
    Sample categorical task-family IDs using pre-generated uniforms.
    """
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
# EVALUATOR CORRUPTION
# ============================================================

def corrupt_iid(
    y_true,
    flip_randomness,
    cfg,
):
    """
    Flip exactly error_rate * N labels.

    For N=40 and error_rate=0.10:
        exactly 4 labels are flipped every round.
    """
    y_hat = y_true.copy()

    m = int(cfg.N * cfg.error_rate)

    idx = np.argsort(
        flip_randomness
    )[:m]

    y_hat[idx] = 1 - y_hat[idx]

    return y_hat


def corrupt_structured(
    y_true,
    task_ids,
    p_family,
    flip_randomness,
    cfg,
):
    """
    ORIGINAL structured corruption from v0.

    IMPORTANT:
    We intentionally keep this unchanged for this diagnostic run.

    It concentrates errors near the current capability frontier
    and prefers success -> failure errors.

    We already suspect this corruption accidentally helps the
    current curriculum. That is exactly why we are NOT using the
    matched-accuracy result as evidence yet.
    """
    y_hat = y_true.copy()

    m = int(cfg.N * cfg.error_rate)

    # Families with true success probability closest to 0.5.
    frontier_families = np.argsort(
        np.abs(p_family - 0.5)
    )[:3]

    is_frontier = np.isin(
        task_ids,
        frontier_families,
    )

    priority = flip_randomness.copy()

    # Prefer frontier examples.
    priority += (~is_frontier) * 10.0

    # Prefer true success -> reported failure.
    priority += (y_true == 0) * 3.0

    idx = np.argsort(priority)[:m]

    y_hat[idx] = 1 - y_hat[idx]

    return y_hat


# ============================================================
# CLOSED-LOOP UPDATE
# ============================================================

def estimate_family_success(
    task_ids,
    y_hat,
    K,
):
    """
    Estimate evaluator-reported success rate for each task family.

    Beta(1,1) smoothing prevents tiny/empty families from creating
    extreme estimates.
    """
    counts = np.bincount(
        task_ids,
        minlength=K,
    )

    successes = np.bincount(
        task_ids,
        weights=y_hat,
        minlength=K,
    )

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
    ORIGINAL v0 weakness-targeting curriculum.

    Lower estimated success => greater future task probability.

    We already discovered this does not perfectly align with the
    learner's true learning-value function.

    KEEPING IT UNCHANGED FOR THIS DIAGNOSTIC.
    """
    logits = (
        np.log(q + 1e-12)
        + cfg.alpha * (1.0 - p_hat)
    )

    logits -= logits.max()

    q_new = np.exp(logits)
    q_new /= q_new.sum()

    # Small uniform exploration prevents irreversible collapse.
    q_new = (
        (1.0 - cfg.exploration) * q_new
        + cfg.exploration
        * np.ones(cfg.K)
        / cfg.K
    )

    return q_new


def update_capability(
    capability,
    q_new,
    difficulty,
    cfg,
):
    """
    True learner dynamics.

    Learning is strongest near p=0.5:
        learning_value = 4 p (1-p)

    Very easy tasks and impossibly difficult tasks both provide
    less learning.
    """
    p = success_probs(
        capability,
        difficulty,
        cfg.tau,
    )

    learning_value = (
        4.0
        * p
        * (1.0 - p)
    )

    delta = (
        cfg.beta
        * q_new
        * learning_value
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
    """
    Run one complete adaptive trajectory.

    evaluator:
        "clean"
        "iid"
        "structured"
        "impulse"

    impulse:
        (round_index, task_index)

    For impulse mode, exactly ONE verdict is flipped over the
    entire 40-round trajectory.
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

    for t in range(cfg.T):

        # ----------------------------------------------------
        # Sample tasks
        # ----------------------------------------------------

        task_ids = sample_task_families(
            q,
            stream["task"][t],
        )

        # ----------------------------------------------------
        # True underlying success probabilities
        # ----------------------------------------------------

        p_family = success_probs(
            capability,
            difficulty,
            cfg.tau,
        )

        p_tasks = p_family[task_ids]

        # ----------------------------------------------------
        # Generate true outcomes
        # ----------------------------------------------------

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

        elif evaluator == "structured":

            y_hat = corrupt_structured(
                y_true,
                task_ids,
                p_family,
                stream["flip"][t],
                cfg,
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
        # Closed-loop curriculum update
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

        # ----------------------------------------------------
        # Learner update
        # ----------------------------------------------------

        capability_new = update_capability(
            capability,
            q_new,
            difficulty,
            cfg,
        )

        # ----------------------------------------------------
        # Advance state
        # ----------------------------------------------------

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
# EXPERIMENT 1
#
# SAME ACCURACY, DIFFERENT FUTURES
#
# NOTE:
# We treat this as a sanity diagnostic only in v0, because we
# already discovered evaluator noise currently improves learning.
# ============================================================

def experiment_matched_accuracy(
    cfg,
    num_seeds=50,
):
    modes = [
        "clean",
        "iid",
        "structured",
    ]

    perf = {
        mode: []
        for mode in modes
    }

    accuracies = {
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

            accuracies[mode].append(
                result["accuracy"].mean()
            )

    for mode in modes:

        perf[mode] = np.asarray(
            perf[mode]
        )

    print()
    print(
        "===== MATCHED ACCURACY ====="
    )

    for mode in modes:

        mean_acc = np.mean(
            accuracies[mode]
        )

        final_perf = (
            perf[mode][:, -1]
        )

        print(
            f"{mode:12s} "
            f"accuracy={mean_acc:.3f} "
            f"final_true_perf="
            f"{final_perf.mean():.4f} "
            f"+/- {final_perf.std():.4f}"
        )

    clean_final = (
        perf["clean"][:, -1]
    )

    iid_harm = (
        clean_final
        - perf["iid"][:, -1]
    )

    structured_harm = (
        clean_final
        - perf["structured"][:, -1]
    )

    print()

    print(
        "IID mean signed harm:",
        iid_harm.mean(),
    )

    print(
        "Structured mean signed harm:",
        structured_harm.mean(),
    )

    print(
        "IID mean absolute deviation:",
        np.mean(
            np.abs(iid_harm)
        ),
    )

    print(
        "Structured mean absolute deviation:",
        np.mean(
            np.abs(structured_harm)
        ),
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

        mean = (
            perf[mode]
            .mean(axis=0)
        )

        std = (
            perf[mode]
            .std(axis=0)
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
            alpha=0.15,
        )

    plt.xlabel(
        "Adaptation round"
    )

    plt.ylabel(
        "Frozen-oracle true performance"
    )

    plt.title(
        "Matched Static Accuracy"
    )

    plt.legend()
    plt.tight_layout()

    Path(
        "results"
    ).mkdir(
        exist_ok=True
    )

    plt.savefig(
        "results/matched_accuracy.png",
        dpi=180,
    )

    plt.close()


# ============================================================
# IMPULSE METRICS
# ============================================================

def compute_short_gain(
    clean,
    perturbed,
    t0,
    horizons=(1, 2, 3),
):
    """
    Short-term deviation produced by one evaluator mistake.

    For now we only use curriculum JS divergence.

    Later we can test richer gain definitions.
    """
    gain = 0.0

    for h in horizons:

        gain += js_divergence(
            clean["q"][t0 + h],
            perturbed["q"][t0 + h],
        )

    return gain


def compute_final_metrics(
    clean,
    perturbed,
    cfg,
):
    """
    IMPORTANT:

    We now separate:

    1. signed performance harm
    2. absolute performance deviation
    3. final capability-state deviation
    4. final curriculum deviation

    Stability is primarily about deviation magnitude,
    not whether the deviation happens to be beneficial.
    """

    signed_harm = (
        clean["perf"][-1]
        - perturbed["perf"][-1]
    )

    abs_harm = abs(
        signed_harm
    )

    capability_dev = (
        np.linalg.norm(
            clean["c"][-1]
            - perturbed["c"][-1]
        )
        / np.sqrt(cfg.K)
    )

    curriculum_dev = (
        js_divergence(
            clean["q"][-1],
            perturbed["q"][-1],
        )
    )

    return {
        "signed_harm": signed_harm,
        "abs_harm": abs_harm,
        "capability_dev": capability_dev,
        "curriculum_dev": curriculum_dev,
    }


# ============================================================
# EXPERIMENT 2
#
# SINGLE-ERROR IMPULSE
#
# KEY QUESTION:
#
# Does short-run sensitivity predict long-run DEVIATION?
# ============================================================

def experiment_impulse(cfg):
    gains = []

    signed_harms = []
    abs_harms = []

    capability_devs = []
    curriculum_devs = []

    impulse_rounds = [
        5,
        8,
        12,
        16,
        20,
    ]

    # 40 seeds
    # x 5 impulse rounds
    # x 4 verdict positions
    #
    # = 800 counterfactual interventions

    for seed in range(40):

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

            for idx in [
                0,
                7,
                18,
                31,
            ]:

                perturbed = run_trajectory(
                    cfg,
                    stream,
                    evaluator="impulse",
                    impulse=(
                        t0,
                        idx,
                    ),
                )

                # ------------------------------------------------
                # Short-horizon response
                # ------------------------------------------------

                gain = compute_short_gain(
                    clean,
                    perturbed,
                    t0,
                )

                # ------------------------------------------------
                # Long-horizon outcomes
                # ------------------------------------------------

                metrics = (
                    compute_final_metrics(
                        clean,
                        perturbed,
                        cfg,
                    )
                )

                gains.append(
                    gain
                )

                signed_harms.append(
                    metrics[
                        "signed_harm"
                    ]
                )

                abs_harms.append(
                    metrics[
                        "abs_harm"
                    ]
                )

                capability_devs.append(
                    metrics[
                        "capability_dev"
                    ]
                )

                curriculum_devs.append(
                    metrics[
                        "curriculum_dev"
                    ]
                )

    # --------------------------------------------------------
    # Convert to arrays
    # --------------------------------------------------------

    gains = np.asarray(
        gains
    )

    signed_harms = np.asarray(
        signed_harms
    )

    abs_harms = np.asarray(
        abs_harms
    )

    capability_devs = np.asarray(
        capability_devs
    )

    curriculum_devs = np.asarray(
        curriculum_devs
    )

    # --------------------------------------------------------
    # Correlations
    # --------------------------------------------------------

    rho_signed, p_signed = (
        spearmanr(
            gains,
            signed_harms,
        )
    )

    rho_abs, p_abs = (
        spearmanr(
            gains,
            abs_harms,
        )
    )

    rho_cap, p_cap = (
        spearmanr(
            gains,
            capability_devs,
        )
    )

    rho_q, p_q = (
        spearmanr(
            gains,
            curriculum_devs,
        )
    )

    print()
    print(
        "===== IMPULSE EXPERIMENT ====="
    )

    print(
        "Number of interventions:",
        len(gains),
    )

    print()

    print(
        f"rho(gain, signed harm) = "
        f"{rho_signed:.3f}, "
        f"p={p_signed:.3e}"
    )

    print(
        f"rho(gain, |harm|) = "
        f"{rho_abs:.3f}, "
        f"p={p_abs:.3e}"
    )

    print(
        f"rho(gain, final capability deviation) = "
        f"{rho_cap:.3f}, "
        f"p={p_cap:.3e}"
    )

    print(
        f"rho(gain, final curriculum deviation) = "
        f"{rho_q:.3f}, "
        f"p={p_q:.3e}"
    )

    print()

    print(
        "Mean signed long-term harm:",
        signed_harms.mean(),
    )

    print(
        "Mean |long-term harm|:",
        abs_harms.mean(),
    )

    print(
        "Mean final capability deviation:",
        capability_devs.mean(),
    )

    print(
        "Mean final curriculum deviation:",
        curriculum_devs.mean(),
    )

    print()

    print(
        "Fraction of interventions harmful:",
        np.mean(
            signed_harms > 0
        ),
    )

    print(
        "Fraction of interventions beneficial:",
        np.mean(
            signed_harms < 0
        ),
    )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    Path(
        "results"
    ).mkdir(
        exist_ok=True
    )

    # 1. signed harm

    plt.figure(
        figsize=(7, 5)
    )

    plt.scatter(
        gains,
        signed_harms,
        s=16,
        alpha=0.45,
    )

    plt.xlabel(
        "Short-horizon feedback gain"
    )

    plt.ylabel(
        "Signed long-term performance harm"
    )

    plt.title(
        f"Gain vs Signed Harm "
        f"(rho={rho_signed:.2f})"
    )

    plt.tight_layout()

    plt.savefig(
        "results/gain_vs_signed_harm.png",
        dpi=180,
    )

    plt.close()

    # 2. absolute harm

    plt.figure(
        figsize=(7, 5)
    )

    plt.scatter(
        gains,
        abs_harms,
        s=16,
        alpha=0.45,
    )

    plt.xlabel(
        "Short-horizon feedback gain"
    )

    plt.ylabel(
        "|Long-term performance deviation|"
    )

    plt.title(
        f"Gain vs |Performance Deviation| "
        f"(rho={rho_abs:.2f})"
    )

    plt.tight_layout()

    plt.savefig(
        "results/gain_vs_abs_harm.png",
        dpi=180,
    )

    plt.close()

    # 3. capability deviation

    plt.figure(
        figsize=(7, 5)
    )

    plt.scatter(
        gains,
        capability_devs,
        s=16,
        alpha=0.45,
    )

    plt.xlabel(
        "Short-horizon feedback gain"
    )

    plt.ylabel(
        "Final capability-state deviation"
    )

    plt.title(
        f"Gain vs Capability Deviation "
        f"(rho={rho_cap:.2f})"
    )

    plt.tight_layout()

    plt.savefig(
        "results/gain_vs_capability_dev.png",
        dpi=180,
    )

    plt.close()

    # 4. curriculum deviation

    plt.figure(
        figsize=(7, 5)
    )

    plt.scatter(
        gains,
        curriculum_devs,
        s=16,
        alpha=0.45,
    )

    plt.xlabel(
        "Short-horizon feedback gain"
    )

    plt.ylabel(
        "Final curriculum JS divergence"
    )

    plt.title(
        f"Gain vs Curriculum Deviation "
        f"(rho={rho_q:.2f})"
    )

    plt.tight_layout()

    plt.savefig(
        "results/gain_vs_curriculum_dev.png",
        dpi=180,
    )

    plt.close()

    return {
        "rho_signed": rho_signed,
        "rho_abs": rho_abs,
        "rho_cap": rho_cap,
        "rho_q": rho_q,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    cfg = Config()

    print(cfg)

    experiment_matched_accuracy(
        cfg
    )

    results = experiment_impulse(
        cfg
    )

    print()
    print(
        "===== DIAGNOSTIC INTERPRETATION ====="
    )

    print()

    print(
        "Do NOT use rho(gain, signed harm) "
        "as the main stability test."
    )

    print(
        "The important quantities are:"
    )

    print(
        "  1. rho(gain, |harm|)"
    )

    print(
        "  2. rho(gain, capability deviation)"
    )

    print(
        "  3. rho(gain, curriculum deviation)"
    )

    print()

    strongest = max(
        results["rho_abs"],
        results["rho_cap"],
        results["rho_q"],
    )

    if strongest >= 0.5:

        print(
            "GREEN:"
        )

        print(
            "Short-horizon sensitivity has "
            "substantial predictive signal "
            "for long-horizon deviation."
        )

        print(
            "Next step: fix the curriculum-objective "
            "mismatch and test held-out alpha/beta."
        )

    elif strongest >= 0.25:

        print(
            "YELLOW:"
        )

        print(
            "There is some predictive signal, "
            "but it is not yet convincing."
        )

        print(
            "Next step: fix the simulator dynamics "
            "before drawing conclusions."
        )

    else:

        print(
            "RED FOR THIS V0 SYSTEM:"
        )

        print(
            "The current short-horizon gain "
            "does not predict long-horizon deviation."
        )

        print(
            "This does NOT yet falsify the research idea, "
            "because the v0 curriculum and learner objectives "
            "are known to be misaligned."
        )


if __name__ == "__main__":
    main()