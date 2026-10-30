"""Binary risk metrics, confidence intervals, and retrospective stability checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


def binary_metrics(target: np.ndarray, probability: np.ndarray,
                   threshold: float = 0.5) -> dict:
    """Evaluate probabilities and a fixed classification threshold.

    Args:
        target: Binary outcomes with both classes.
        probability: Aligned predicted probabilities.
        threshold: Frozen positive-class cut-off; not fitted on test outcomes.

    Returns:
        AUC, KS, Gini, AP, calibration losses, and confusion-matrix metrics.
    """
    prediction = probability >= threshold
    tn, fp, fn, tp = confusion_matrix(target, prediction, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        target, prediction, average="binary", zero_division=0
    )
    false_positive, true_positive, _ = roc_curve(target, probability)
    auc = roc_auc_score(target, probability)
    return {
        "n": len(target), "positive_rate": float(np.mean(target)),
        "auc": float(auc), "ks": float(np.max(true_positive - false_positive)),
        "gini": float(2 * auc - 1),
        "average_precision": float(average_precision_score(target, probability)),
        "brier": float(brier_score_loss(target, probability)),
        "log_loss": float(log_loss(target, probability, labels=[0, 1])),
        "threshold": float(threshold), "precision": float(precision),
        "recall": float(recall), "f1": float(f1),
        "accuracy": float((tp + tn) / len(target)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def ks_threshold(target: np.ndarray, probability: np.ndarray) -> float:
    """Choose a diagnostic threshold on validation or policy data only.

    Args:
        target: Non-test binary outcomes.
        probability: Aligned non-test probabilities.

    Returns:
        Finite threshold maximising TPR minus FPR.
    """
    fpr, tpr, thresholds = roc_curve(target, probability)
    return float(thresholds[np.argmax(tpr - fpr)])


def wilson_interval(events: int, count: int) -> tuple[float, float]:
    """Compute a two-sided nominal 95% Wilson interval.

    Args:
        events: Number of positive labels.
        count: Number of observations in the group.

    Returns:
        Lower and upper binomial proportion bounds; NaNs for an empty group.
    """
    if count == 0:
        return float("nan"), float("nan")
    rate, z = events / count, 1.959963984540054
    scale = 1 + z * z / count
    centre = rate + z * z / (2 * count)
    half = z * np.sqrt(rate * (1 - rate) / count + z * z / (4 * count * count))
    return float((centre - half) / scale), float((centre + half) / scale)


def paired_bootstrap(
    target: np.ndarray, predictions: dict, repeats: int = 200, seed: int = 42
) -> pd.DataFrame:
    """Estimate paired AUC uncertainty using stratified resampling.

    Args:
        target: Fixed retrospective holdout labels.
        predictions: Model-name to holdout-probability mapping.
        repeats: Number of bootstrap draws.
        seed: Reproducible RNG seed.

    Returns:
        Percentile confidence intervals for each AUC and differences to first model.
    """
    rng = np.random.default_rng(seed)
    indices = [np.flatnonzero(target == value) for value in (0, 1)]
    names = list(predictions)
    estimates = np.zeros((repeats, len(names)))
    for iteration in range(repeats):
        sample = np.concatenate([
            rng.choice(group, len(group), replace=True) for group in indices
        ])
        for model_index, name in enumerate(names):
            estimates[iteration, model_index] = roc_auc_score(
                target[sample], predictions[name][sample]
            )
    rows = []
    for model_index, name in enumerate(names):
        lower, upper = np.quantile(estimates[:, model_index], [0.025, 0.975])
        difference = estimates[:, model_index] - estimates[:, 0]
        delta_lower, delta_upper = np.quantile(difference, [0.025, 0.975])
        tree_difference = estimates[:, model_index] - estimates[:, min(1,
            len(names) - 1)]
        tree_lower, tree_upper = np.quantile(tree_difference, [0.025, 0.975])
        rows.append({
            "model": name, "auc": roc_auc_score(target, predictions[name]),
            "tree_reference": names[min(1, len(names) - 1)],
            "delta_vs_tree_lower": tree_lower, "delta_vs_tree_upper": tree_upper,
            "auc_ci_lower": lower, "auc_ci_upper": upper,
            "delta_reference": names[0], "delta_ci_lower": delta_lower,
            "delta_ci_upper": delta_upper, "bootstrap_repeats": repeats,
        })
    return pd.DataFrame(rows)


def calibration_table(target: np.ndarray, probability: np.ndarray) -> pd.DataFrame:
    """Compare mean prediction to observed outcome rates in equal-count bins.

    Args:
        target: Evaluation labels.
        probability: Evaluation probabilities.

    Returns:
        Decile counts, mean probabilities, and observed TARGET rates.
    """
    table = pd.DataFrame({"target": target, "probability": probability})
    table["bin"] = pd.qcut(table["probability"], 10, duplicates="drop")
    return table.groupby("bin", observed=True).agg(
        n=("target", "size"), predicted=("probability", "mean"),
        observed=("target", "mean"), positives=("target", "sum"),
    ).reset_index().assign(bin=lambda frame: frame["bin"].astype(str))


def population_stability(reference: np.ndarray, current: np.ndarray) -> dict:
    """Calculate PSI with frozen reference deciles and additive smoothing.

    Args:
        reference: Non-test score distribution defining cut points.
        current: Test or later monitoring score distribution.

    Returns:
        PSI and bin proportions. Random-split PSI is not temporal validation.
    """
    interior = np.unique(np.quantile(reference, np.linspace(0.1, 0.9, 9)))
    edges = np.r_[-np.inf, interior, np.inf]
    ref = np.histogram(reference, edges)[0] + 0.5
    cur = np.histogram(current, edges)[0] + 0.5
    ref, cur = ref / ref.sum(), cur / cur.sum()
    return {
        "psi": float(np.sum((cur - ref) * np.log(cur / ref))),
        "reference_share": ref.tolist(), "current_share": cur.tolist(),
        "interpretation": "Random holdout comparison; no time stability claim",
    }


def calibrated_probability(calibrator: object, margin: np.ndarray) -> np.ndarray:
    """Apply the fitted logistic calibration map to raw tree margins.

    Args:
        calibrator: Fitted one-dimensional sklearn LogisticRegression.
        margin: Raw LightGBM log-odds predictions.

    Returns:
        Calibrated probabilities consistent with scaled SHAP additivity.
    """
    return expit(calibrator.intercept_[0] + calibrator.coef_[0, 0] * margin)
