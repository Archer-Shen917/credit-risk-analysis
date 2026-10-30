"""Exact log-odds SHAP explanations, including the linear calibration map."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.special import expit

from credit_risk.evaluation import calibrated_probability
from credit_risk.io_utils import write_json, write_table


def save_force_plot(
    base: float, values: np.ndarray, names: list[str], path: Path,
) -> None:
    """Save a compact force plot with the exact omitted-feature contribution.

    Args:
        base: Calibrated expected log-odds baseline.
        values: Exact calibrated log-odds contributions for one application.
        names: Feature names aligned with contributions.
        path: Output PNG filename.

    Returns:
        None; top six drivers plus the residual contribution are rendered.
    """
    order = np.argsort(np.abs(values))[::-1][:6]
    retained = values[order]
    compact = np.r_[retained, values.sum() - retained.sum()]
    labels = [names[index] for index in order] + ["Other features (sum)"]
    shap.force_plot(
        base, compact, feature_names=labels, matplotlib=True, show=False,
        figsize=(16, 3.3), text_rotation=12, contribution_threshold=0.01,
    )
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close("all")


def explain_final(bundle: dict, test_x: pd.DataFrame, target: np.ndarray,
                  destination: Path, seed: int) -> dict:
    """Explain a random global sample and six score-selected local cases.

    Args:
        bundle: Final model and logistic calibration objects.
        test_x: Final-preprocessed holdout features.
        target: Outcomes used only to annotate cases after selection.
        destination: Stage 3 output folder.
        seed: Seed for the uniformly sampled global explanation subset.

    Returns:
        Additivity errors and case metadata; no causal effects are claimed.
    """
    rng = np.random.default_rng(seed)
    sample_index = rng.choice(len(test_x), min(1500, len(test_x)), replace=False)
    sample = test_x.iloc[sample_index]
    model, calibrator = bundle["final_tree"], bundle["calibrator"]
    explainer = shap.TreeExplainer(model, model_output="raw")
    explanation = explainer(sample)
    values = np.asarray(explanation.values)
    if values.ndim == 3:
        values = values[:, :, 1]
    raw_base = float(np.asarray(explanation.base_values).reshape(-1)[0])
    slope = float(calibrator.coef_[0, 0])
    intercept = float(calibrator.intercept_[0])
    base, scaled = slope * raw_base + intercept, slope * values
    predicted = calibrated_probability(calibrator, model.predict(sample,
        raw_score=True))
    error = float(np.max(np.abs(expit(base + scaled.sum(axis=1)) - predicted)))
    if error > 1e-6:
        raise AssertionError(f"SHAP calibration additivity failed: {error}")
    direction = []
    numeric_display = sample.copy()
    for index, name in enumerate(sample.columns):
        if isinstance(sample[name].dtype, pd.CategoricalDtype):
            numeric_display[name] = sample[name].cat.codes
            association = np.nan
        elif (
            sample[name].nunique(dropna=True) < 2
            or np.unique(scaled[:, index]).size < 2
        ):
            association = np.nan
        else:
            association = sample[name].corr(pd.Series(
                scaled[:, index], index=sample.index
            ), method="spearman")
        direction.append(association)
    importance = pd.DataFrame({
        "feature": sample.columns, "mean_absolute_shap": np.abs(scaled).mean(axis=0),
        "spearman_value_vs_shap": direction,
        "gain_importance": model.booster_.feature_importance(importance_type="gain"),
    }).sort_values("mean_absolute_shap", ascending=False)
    write_table(importance, destination / "shap_global_importance.csv")
    shap.summary_plot(scaled, numeric_display, max_display=10, show=False)
    plt.xlabel("SHAP contribution to calibrated log-odds")
    plt.tight_layout()
    plt.savefig(destination / "shap_summary.png", dpi=160, bbox_inches="tight")
    plt.close("all")
    probabilities = calibrated_probability(calibrator, model.predict(test_x,
        raw_score=True))
    order = np.argsort(probabilities)
    local_rows, contributions = [], []
    for prefix, fractions in (("L", (0.01, 0.05, 0.09)), ("H", (0.91, 0.95, 0.99))):
        for number, quantile in enumerate(fractions, 1):
            index = int(order[min(int(len(order) * quantile), len(order) - 1)])
            case_id = f"{prefix}{number}"
            local = explainer(test_x.iloc[[index]])
            contribution = np.asarray(local.values).reshape(-1) * slope
            local_base = slope * float(np.asarray(local.base_values).reshape(-1)[
                0]) + intercept
            reconstructed = float(expit(local_base + contribution.sum()))
            save_force_plot(local_base, contribution, test_x.columns.tolist(),
                            destination / f"shap_force_{case_id}.png")
            top = np.argsort(np.abs(contribution))[::-1][:3]
            local_rows.append({
                "case": case_id, "score_quantile": quantile,
                "predicted_probability": probabilities[index],
                "target": int(target[index]),
                "base_log_odds": local_base, "sum_contributions": contribution.sum(),
                "reconstructed_probability": reconstructed,
                "top_drivers": "; ".join(test_x.columns[top].tolist()),
                "selection": "probability rank only; outcomes not used for selection",
            })
            contributions.extend([
                {"case": case_id, "feature": name, "calibrated_log_odds_shap": value}
                for name, value in zip(test_x.columns, contribution)
            ])
    write_table(pd.DataFrame(local_rows), destination / "shap_local_cases.csv")
    write_table(pd.DataFrame(contributions),
        destination / "shap_local_contributions.csv")
    summary = {
        "global_sample_n": len(sample), "sampling": "uniform random holdout sample",
        "local_cases": 6, "calibrated_additivity_max_error": error,
        "baseline_log_odds": base, "calibration_slope": slope,
        "units": "calibrated log-odds; not percentage-point or causal effects",
        "tree_dependence": (
            "tree_path_dependent; feature dependence affects attribution"
        ),
    }
    write_json(summary, destination / "shap_summary.json")
    return summary
