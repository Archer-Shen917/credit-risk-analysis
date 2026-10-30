"""Research-only three-tier strategies and group diagnostics, not lending actions."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from credit_risk.config import PolicySpec
from credit_risk.evaluation import wilson_interval
from credit_risk.io_utils import write_json, write_table


def assign_tiers(probability: np.ndarray, policy: PolicySpec) -> np.ndarray:
    """Partition probabilities into nonoverlapping low, review, and high tiers.

    Args:
        probability: Calibrated model probabilities.
        policy: Frozen ordered lower and upper thresholds.

    Returns:
        String labels for every input record.

    Raises:
        ValueError: Thresholds or probabilities are outside valid ranges.
    """
    if not 0 <= policy.lower < policy.upper <= 1:
        raise ValueError("Policy requires 0 <= lower < upper <= 1")
    if not np.isfinite(probability).all() or np.any((probability < 0) | (
        probability > 1)):
        raise ValueError("Probabilities must be finite values in [0, 1]")
    return np.where(probability < policy.lower, "low",
                    np.where(probability >= policy.upper, "high", "review"))


def select_policies(
    probability: np.ndarray, target: np.ndarray, destination: Path,
) -> list[PolicySpec]:
    """Choose two policy variants only on the dedicated policy partition.

    Args:
        probability: Calibrated policy-partition probabilities.
        target: Outcomes from the same non-test partition.
        destination: Folder for the complete threshold search audit.

    Returns:
        Conservative and balanced policies with explicit constraint status.
    """
    rows, policies = [], []
    for name, cap, upper_quantile in (
        ("A_conservative", 0.04, 0.75),
        ("B_balanced", 0.05, 0.85),
    ):
        candidates = []
        upper = float(np.quantile(probability, upper_quantile))
        for lower_quantile in np.arange(0.20, 0.71, 0.05):
            lower = float(np.quantile(probability, lower_quantile))
            if lower >= upper:
                continue
            spec = PolicySpec(name, lower, upper, cap, False)
            tiers = assign_tiers(probability, spec)
            low, review = tiers == "low", tiers == "review"
            n_low = int(low.sum())
            ci_lower, ci_upper = wilson_interval(int(target[low].sum()), n_low)
            record = {
                "policy": name, "lower": lower, "upper": upper,
                "lower_quantile": lower_quantile, "upper_quantile": upper_quantile,
                "auto_approve_share": float(low.mean()),
                "manual_share": float(review.mean()),
                "low_target_rate": float(target[low].mean()),
                "low_ci_lower": ci_lower, "low_ci_upper": ci_upper,
                "risk_cap": cap,
                "feasible": ci_upper <= cap and review.mean() <= 0.40,
            }
            rows.append(record)
            candidates.append(record)
        feasible = [record for record in candidates if record["feasible"]]
        if feasible:
            winner = max(feasible, key=lambda record: record["auto_approve_share"])
        else:
            # Do not pretend that business targets were attainable.
            winner = min(candidates, key=lambda record: record["low_ci_upper"])
        policies.append(PolicySpec(
            name, winner["lower"], winner["upper"], cap,
            bool(winner["feasible"]), winner,
        ))
    write_table(pd.DataFrame(rows), destination / "threshold_grid_policy.csv")
    write_json([asdict(policy) for policy in policies],
        destination / "frozen_policies.json")
    return policies


def surrogate_rule_tiers(features: pd.DataFrame) -> np.ndarray:
    """Apply a researcher-defined comparison rule, not a bank's actual engine.

    Args:
        features: Application features with score, tenure, and loan-income ratio.

    Returns:
        Low, review, and high tier labels; missing inputs default to review.
    """
    score = features["EXT_SOURCE_2"]
    ratio = features["CREDIT_INCOME_RATIO"]
    tenure = features["EMPLOYMENT_YEARS"]
    low = score.ge(0.65) & ratio.le(4) & tenure.ge(2)
    high = score.lt(0.35) | ratio.gt(10)
    return np.where(high, "high", np.where(low, "low", "review"))


def summarise_strategy(
    name: str, tiers: np.ndarray, target: np.ndarray, probability: np.ndarray,
) -> tuple[dict, pd.DataFrame]:
    """Evaluate tier outcomes without assuming an oracle manual review process.

    Args:
        name: Strategy identifier.
        tiers: Fixed low, review, and high group labels.
        target: Test outcomes used solely for final retrospective evaluation.
        probability: Calibrated risk estimates from the final model.

    Returns:
        Strategy-level rates and tier-level counts, observed rates, and intervals.
    """
    rows = []
    for tier in ("low", "review", "high"):
        mask = tiers == tier
        n, positives = int(mask.sum()), int(target[mask].sum())
        lower, upper = wilson_interval(positives, n)
        rows.append({
            "strategy": name, "tier": tier, "n": n, "share": n / len(target),
            "positives": positives,
            "observed_target_rate": positives / n if n else np.nan,
            "mean_predicted_risk": float(probability[mask].mean()) if n else np.nan,
            "ci_lower": lower, "ci_upper": upper,
        })
    low, manual, high = tiers == "low", tiers == "review", tiers == "high"
    weights = low.astype(float) + manual.astype(float) * 0.6
    metrics = {
        "strategy": name, "auto_approve_share": float(low.mean()),
        "manual_share": float(manual.mean()), "auto_reject_share": float(high.mean()),
        "automatic_decision_share": float((~manual).mean()),
        "auto_approve_target_rate": float(target[low].mean()) if low.any() else np.nan,
        "high_target_capture": float(target[high].sum() / target.sum()),
        "projected_approval_share": float(weights.mean()),
        "projected_approved_target_rate": float(np.sum(
            target * weights) / weights.sum()),
        "manual_acceptance_assumption": 0.6,
        "manual_selection_assumption": "random within review tier; no outcome oracle",
    }
    return metrics, pd.DataFrame(rows)


def fairness_diagnostics(
    features: pd.DataFrame, probability: np.ndarray, policy: PolicySpec,
) -> pd.DataFrame:
    """Audit subgroup disparities; this is not a legal fairness certification.

    Args:
        features: Test records; age and gender are audit-only attributes.
        probability: Calibrated test probabilities.
        policy: Frozen policy whose automatic decisions are audited.

    Returns:
        Group counts, AUC, selection rates, TPR, FPR, and uncertainty flags.
    """
    tiers = assign_tiers(probability, policy)
    target = features["TARGET"].to_numpy()
    audit = {
        "gender": features["CODE_GENDER"].fillna("Missing").astype(str),
        "age_band": pd.cut(features["AGE_YEARS"], [0, 30, 45, 60, np.inf],
                           labels=["under30", "30to44", "45to59", "60plus"])
        .astype("object").fillna("Missing"),
    }
    rows = []
    for attribute, values in audit.items():
        for group in sorted(values.unique()):
            mask = values.eq(group).to_numpy()
            y = target[mask]
            high = tiers[mask] == "high"
            low = tiers[mask] == "low"
            n = int(mask.sum())
            positive_n, negative_n = int(y.sum()), int((y == 0).sum())
            lower, upper = wilson_interval(int(low.sum()), n)
            reliable = n >= 500 and min(positive_n, negative_n) >= 30
            rows.append({
                "attribute": attribute, "group": group, "n": n,
                "target_rate": float(y.mean()), "auto_approve_share": float(low.mean()),
                "approval_ci_lower": lower, "approval_ci_upper": upper,
                "high_risk_tpr": float(high[y == 1].mean()) if positive_n else np.nan,
                "high_risk_fpr": float(high[y == 0].mean()) if negative_n else np.nan,
                "auc": roc_auc_score(y, probability[mask]) if reliable else np.nan,
                "sufficient_support": reliable,
            })
    result = pd.DataFrame(rows)
    reference = result["auto_approve_share"].where(
        result["sufficient_support"]
    ).groupby(result["attribute"]).transform("max")
    result["approval_ratio_to_max_group"] = (
        result["auto_approve_share"] / reference
    ).where(result["sufficient_support"])
    return result
