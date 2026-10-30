"""Single-axis charts for the project reports, using default matplotlib colours."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve


def save_current(path: Path) -> None:
    """Save the current single chart and release its memory.

    Args:
        path: Destination PNG file.

    Returns:
        None; the chart is exported at a readable resolution.
    """
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def evaluation_plots(target: np.ndarray, predictions: dict, root: Path) -> None:
    """Render planned model comparisons using final test predictions.

    Args:
        target: Final retrospective holdout outcomes.
        predictions: Model names mapped to aligned probability arrays.
        root: Repository root.

    Returns:
        None; ROC, PR, and calibration charts are saved.
    """
    destination = root / "outputs/phase3"
    plt.figure(figsize=(7.6, 4.6))
    for name, probability in predictions.items():
        fpr, tpr, _ = roc_curve(target, probability)
        plt.plot(fpr, tpr, label=name)
    plt.plot([0, 1], [0, 1], "--", label="Random")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Retrospective holdout: ROC comparison")
    plt.legend(loc="lower right", fontsize=8)
    save_current(destination / "roc_comparison.png")
    plt.figure(figsize=(7.6, 4.6))
    for name, probability in predictions.items():
        precision, recall, _ = precision_recall_curve(target, probability)
        plt.plot(recall, precision, label=name)
    plt.axhline(float(target.mean()), linestyle="--", label="Base event rate")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Retrospective holdout: precision-recall")
    plt.legend(fontsize=8)
    save_current(destination / "precision_recall.png")
    plt.figure(figsize=(7.6, 4.6))
    for filename, label in (("calibration_raw.csv", "Uncalibrated"),
                            ("calibration_final.csv", "Calibrated")):
        table = pd.read_csv(destination / filename)
        plt.plot(table["predicted"], table["observed"], marker="o", label=label)
    plt.plot([0, 0.5], [0, 0.5], "--", label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed TARGET=1 rate")
    plt.title("Calibration: test deciles, no test fitting")
    plt.legend(fontsize=9)
    save_current(destination / "calibration.png")


def result_plots(root: Path) -> None:
    """Generate audit charts from saved aggregate tables.

    Args:
        root: Repository root containing completed aggregate result tables.

    Returns:
        None; one chart per file is saved for deterministic report embedding.
    """
    p1, p3 = root / "outputs/phase1", root / "outputs/phase3"
    quality = pd.read_csv(p1 / "data_quality_profile.csv").nlargest(12, "missing_rate")
    plt.figure(figsize=(8.5, 4.8))
    plt.barh(quality["feature"][::-1], quality["missing_rate"][::-1] * 100)
    plt.xlabel("Missing values (%)")
    plt.title("Original application table: highest missing rates")
    save_current(p1 / "missing_profile.png")
    iv = pd.read_csv(p1 / "numeric_iv_fit.csv").head(12)
    plt.figure(figsize=(8.5, 4.8))
    plt.barh(iv["feature"][::-1], iv["iv"][::-1])
    plt.xlabel("Information value, fitting partition only")
    plt.title("Numeric IV: not independent contributions")
    save_current(p1 / "iv_ranking.png")
    candidates = pd.read_csv(p1 / "candidate_signals_fit.csv").head(6)
    plt.figure(figsize=(8.5, 4.6))
    labels = [f"EXT2={row.EXT_SOURCE_2}; tenure={row.EMPLOYMENT_YEARS}; "
              + f"LTI={row.CREDIT_INCOME_RATIO}" for row in candidates.itertuples()]
    plt.barh(labels[::-1], candidates["target_rate"][::-1] * 100)
    plt.xlabel("Observed TARGET=1 rate (%)")
    plt.title("Cross-business signals: fit partition")
    save_current(p1 / "cross_business_signals.png")
    iterations = pd.read_csv(p3 / "iteration_validation_metrics.csv")
    plt.figure(figsize=(9, 4.5))
    plt.plot(range(len(iterations)), iterations["auc"], marker="o")
    plt.xticks(range(len(iterations)), iterations["name"], rotation=45,
               ha="right", fontsize=8)
    plt.ylabel("Tuning ROC AUC")
    plt.title("Declared iterations: validation-only selection")
    save_current(p3 / "iteration_progress.png")
    metrics = pd.read_csv(p3 / "strategy_test_metrics.csv")
    plt.figure(figsize=(8.3, 4.8))
    x = np.arange(len(metrics))
    for offset, column, label in ((-0.24, "auto_approve_share", "Automatic approval"),
                                  (0, "manual_share", "Manual review"),
                                  (0.24, "auto_reject_share", "Automatic rejection")):
        plt.bar(x + offset, metrics[column] * 100, width=0.24, label=label)
    plt.xticks(x, metrics["strategy"])
    plt.ylabel("Share of held-out applications (%)")
    plt.title("Research simulation: workload and decision mix")
    plt.legend(fontsize=9)
    save_current(p3 / "strategy_mix.png")
