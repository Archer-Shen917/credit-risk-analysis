"""Reproducible WOE/IV and risk interactions for Home Credit Phase 1.

Exploratory summaries only: all quantile boundaries and target-conditioned WOE/IV
are fitted on the full input dataset. Refit on training folds in Phase 2; do not
use this report to make credit decisions or claim out-of-sample accuracy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SMOOTHING = 0.5
N_BINS = 5
MIN_CELL_SIZE = 500
RISK_UPLIFT_MIN = 1.4


def add_derived_features(data: pd.DataFrame) -> pd.DataFrame:
    """Return numerics with interpretable exploratory proxies.

    Args:
        data: Original application-level observations.

    Returns:
        Numeric columns and derived application metrics.
    """
    numeric = data.select_dtypes(include="number").copy()
    numeric = numeric.drop(columns=["SK_ID_CURR", "TARGET"], errors="ignore")
    numeric["DAYS_EMPLOYED"] = numeric["DAYS_EMPLOYED"].mask(
        numeric["DAYS_EMPLOYED"].eq(365243)
    )
    numeric["AGE_YEARS"] = -numeric["DAYS_BIRTH"] / 365.25
    numeric["EMPLOYMENT_YEARS"] = -numeric["DAYS_EMPLOYED"] / 365.25
    income = numeric["AMT_INCOME_TOTAL"].where(
        numeric["AMT_INCOME_TOTAL"].gt(0)
    )
    numeric["CREDIT_INCOME_RATIO"] = numeric["AMT_CREDIT"] / income
    numeric["ANNUITY_INCOME_RATIO"] = numeric["AMT_ANNUITY"] / income
    return numeric.replace([np.inf, -np.inf], np.nan)


def make_numeric_bins(series: pd.Series, n_bins: int = N_BINS) -> pd.Series:
    """Make unsupervised bins; preserve missing values as a separate group.

    Args:
        series: Input numeric feature (including NaN).
        n_bins: Desired number of quantile intervals.

    Returns:
        Labels for data-driven intervals, low-cardinality categories, and NaN.
    """
    cleaned = pd.to_numeric(series, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    valid = cleaned.dropna()
    if valid.empty:
        return pd.Series("Missing", index=series.index, name=series.name)
    unique_count = valid.nunique()
    if unique_count <= n_bins:
        labels = cleaned.map(lambda x: f"Value={x:g}" if pd.notna(x) else "Missing")
        return labels.astype("object")
    try:
        discretized = pd.qcut(cleaned, q=n_bins, duplicates="drop")
        labels = discretized.astype("object").map(
            lambda x: str(x) if pd.notna(x) else "Missing"
        )
        if labels[cleaned.notna()].nunique() < 2:
            # Ties can make quantile edges collapse into a single bin.
            distinct = cleaned.value_counts(dropna=True)
            common = distinct.index[0]
            labels = cleaned.map(
                lambda x: (
                    "Missing"
                    if pd.isna(x)
                    else "Modal value" if x == common else "Other"
                )
            )
        return labels.astype("object")
    except ValueError:
        return pd.Series("Missing", index=series.index, name=series.name)


def calculate_woe_iv(
    feature: pd.Series, target: pd.Series, smoothing: float = SMOOTHING
) -> tuple[pd.DataFrame, dict]:
    """Summarize each supervised risk rate with smoothed WOE and IV.

    WOE = log(% good / % bad), so negative values indicate greater target risk.

    Args:
        feature: Numeric feature to bin independently of TARGET.
        target: Binary positive label, 1 means observed repayment difficulty.
        smoothing: Additive pseudo-count per group and outcome class.

    Returns:
        Full bin table and a feature-level diagnostic summary.

    Raises:
        ValueError: Missing or non-binary target classes.
    """
    if not target.isin([0, 1]).all() or target.nunique() < 2:
        raise ValueError("TARGET必须包含0和1，并且不存在缺失")
    if smoothing <= 0:
        raise ValueError("平滑参数必须大于0")
    labels = make_numeric_bins(feature)
    counts = pd.DataFrame({"bin": labels, "target": target}).groupby(
        "bin", observed=True, sort=True
    )["target"].agg(["size", "sum"])
    counts = counts.rename(columns={"size": "n", "sum": "bad"})
    counts["good"] = counts["n"] - counts["bad"]
    k = len(counts)
    total_good = counts["good"].sum()
    total_bad = counts["bad"].sum()
    counts["dist_good"] = (counts["good"] + smoothing) / (
        total_good + smoothing * k
    )
    counts["dist_bad"] = (counts["bad"] + smoothing) / (
        total_bad + smoothing * k
    )
    counts["woe"] = np.log(counts["dist_good"] / counts["dist_bad"])
    counts["iv_bin"] = (
        counts["dist_good"] - counts["dist_bad"]
    ) * counts["woe"]
    counts["target_rate"] = counts["bad"] / counts["n"]
    counts["sample_share"] = counts["n"] / counts["n"].sum()
    counts["risk_uplift"] = counts["target_rate"] / target.mean()
    counts = counts.reset_index()
    total_iv = float(counts["iv_bin"].sum())
    counts["feature"] = feature.name
    counts["iv_total"] = total_iv
    counts["support_warning"] = counts["n"] < MIN_CELL_SIZE
    missing = feature.isna().sum()
    info = {
        "feature": feature.name,
        "iv": total_iv,
        "missing_count": int(missing),
        "missing_rate": float(missing / len(feature)),
        "number_of_bins": int(k),
        "smallest_group": int(counts["n"].min()),
        "small_bin_count": int(counts["support_warning"].sum()),
        "unique_non_missing": int(feature.nunique(dropna=True)),
        "all_missing": bool(missing == len(feature)),
    }
    return counts, info


def three_levels(series: pd.Series, prefix: str) -> pd.Series:
    """Create low/middle/high bands using 25th/75th percentiles.

    Args:
        series: Observed numeric distribution, with missing values.
        prefix: Human-readable feature name.

    Returns:
        Three or four bands including explicitly missing observations.
    """
    p25, p75 = series.quantile([0.25, 0.75])
    if not np.isfinite(p25) or not np.isfinite(p75) or p25 >= p75:
        raise ValueError(f"{prefix}无法形成稳定的三个分层")
    bins = [-np.inf, p25, p75, np.inf]
    labels = [f"{prefix}:Low", f"{prefix}:Middle", f"{prefix}:High"]
    categorical = pd.cut(series, bins=bins, labels=labels, include_lowest=True)
    return categorical.astype("object").where(categorical.notna(), f"{prefix}:Missing")


def wilson_lower(successes: pd.Series, size: pd.Series) -> pd.Series:
    """Calculate Wilson 95% lower confidence bound for a binomial rate.

    Args:
        successes: Number of positive labels per cell.
        size: Number of observations per cell.

    Returns:
        Wilson lower bounds at the 95% confidence level.
    """
    z = 1.96
    rate = successes / size
    normalizer = 1 + z**2 / size
    centre = rate + z**2 / (2 * size)
    uncertainty = z * np.sqrt(rate * (1 - rate) / size + z**2 / (4 * size**2))
    return ((centre - uncertainty) / normalizer).clip(lower=0)


def cross_risk_table(data: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    """Compute cell-level observations and conservatively flagged candidates.

    Args:
        data: DataFrame containing TARGET and named segmented columns.
        groups: One or more segmentation feature names.

    Returns:
        Descriptive risk-level table with support and CI safeguards.
    """
    grouped = data.groupby(groups, observed=True)["TARGET"].agg(
        n="size", positives="sum", rate="mean"
    )
    result = grouped.reset_index()
    overall = float(data["TARGET"].mean())
    result["share"] = result["n"] / len(data)
    result["uplift"] = result["rate"] / overall
    result["wilson_lower_95"] = wilson_lower(result["positives"], result["n"])
    result["min_n_pass"] = result["n"] >= MIN_CELL_SIZE
    result["candidate_signal"] = (
        result["min_n_pass"]
        & result["uplift"].ge(RISK_UPLIFT_MIN)
        & result["wilson_lower_95"].gt(overall)
    )
    result["comparison"] = "full sample, exploratory only"
    return result.sort_values(["rate", "n"], ascending=[False, False])


def save_bar_chart(table: pd.DataFrame, path: Path) -> None:
    """Save IV ranking graphic (excluding low-support diagnostics from top labels)."""
    top = table.loc[~table["all_missing"]].sort_values("iv").tail(18)
    plt.figure(figsize=(10.2, 7.2))
    plt.barh(top["feature"], top["iv"], color="#276C82")
    plt.xlabel("Information Value (in-sample exploratory)")
    plt.title("Top numeric predictors by IV | all application rows", loc="left")
    plt.tight_layout()
    plt.savefig(path, dpi=175, bbox_inches="tight")
    plt.close()


def save_cross_heatmap(table: pd.DataFrame, path: Path) -> None:
    """Save joint EXT_SOURCE_2 x EXT_SOURCE_3 descriptive risk rates."""
    left, right = "EXT_SOURCE_2", "EXT_SOURCE_3"
    matrix = table.pivot(index=left, columns=right, values="rate")
    colnames = ["Low", "Middle", "High", "Missing"]
    rows = [f"E2:{name}" for name in colnames]
    cols = [f"E3:{name}" for name in colnames]
    matrix = matrix.reindex(index=rows, columns=cols)
    fig, ax = plt.subplots(figsize=(8.1, 5.0))
    image = ax.imshow(matrix.to_numpy(dtype=float) * 100, cmap="Blues", vmin=0)
    ax.set_xticks(range(len(cols)), colnames)
    ax.set_yticks(range(len(rows)), colnames)
    ax.set_xlabel("EXT_SOURCE_3: percentile band")
    ax.set_ylabel("EXT_SOURCE_2: percentile band")
    ax.set_title("Observed TARGET=1 rate (%) | external scores", loc="left")
    for (i, j), value in np.ndenumerate(matrix.to_numpy(dtype=float)):
        if not np.isnan(value):
            ax.text(j, i, f"{value:.1%}", ha="center", va="center", fontsize=10)
    fig.colorbar(image, ax=ax, label="Observed label rate (%)")
    plt.tight_layout()
    plt.savefig(path, dpi=175, bbox_inches="tight")
    plt.close()


def run_extension(data_path: Path, output_dir: Path) -> dict:
    """Run every numeric WOE/IV diagnostic and 2/3-way risk interactions.

    Args:
        data_path: Public Home Credit labeled applicant CSV.
        output_dir: Existing Phase 1 output directory.

    Returns:
        Summary and selected descriptive findings.
    """
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    out_tables = output_dir / "tables"
    out_figures = output_dir / "figures"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figures.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(data_path, low_memory=False)
    if not {"TARGET", "SK_ID_CURR"}.issubset(data.columns):
        raise ValueError("缺少申请唯一标识或目标变量")
    if data["SK_ID_CURR"].duplicated().any():
        raise ValueError("主表申请ID存在重复")
    target = data["TARGET"].astype("int8")
    features = add_derived_features(data)
    all_bins = []
    ranking = []
    for feature in features.columns:
        bins, description = calculate_woe_iv(features[feature], target)
        all_bins.append(bins)
        ranking.append(description)
    ranks = pd.DataFrame(ranking).sort_values("iv", ascending=False)
    ranks.insert(0, "rank", range(1, len(ranks) + 1))
    bins = pd.concat(all_bins, ignore_index=True)
    bins = bins.merge(ranks[["feature", "rank"]], on="feature", how="left")
    ordered_cols = [
        "rank", "feature", "bin", "n", "bad", "good", "target_rate",
        "sample_share", "risk_uplift", "dist_good", "dist_bad", "woe",
        "iv_bin", "iv_total", "support_warning"
    ]
    bins = bins[ordered_cols].sort_values(["rank", "bin"])
    ranks.to_csv(
        out_tables / "iv_numeric_ranking.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bins.to_csv(
        out_tables / "woe_iv_all_numeric_bins.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Standardized names below refer only to anonymous public external signals.
    joined = pd.DataFrame({"TARGET": target})
    joined["EXT_SOURCE_2"] = three_levels(features["EXT_SOURCE_2"], "E2")
    joined["EXT_SOURCE_3"] = three_levels(features["EXT_SOURCE_3"], "E3")
    joined["CREDIT_INCOME_RATIO"] = three_levels(
        features["CREDIT_INCOME_RATIO"], "CIR"
    )
    two = cross_risk_table(joined, ["EXT_SOURCE_2", "EXT_SOURCE_3"])
    three = cross_risk_table(
        joined, ["EXT_SOURCE_2", "EXT_SOURCE_3", "CREDIT_INCOME_RATIO"]
    )
    two.to_csv(out_tables / "cross_ext2_ext3.csv", index=False, encoding="utf-8-sig")
    three.to_csv(
        out_tables / "cross_ext2_ext3_credit_income.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Simpler decision-aid table for user-reviewable candidate combinations.
    candidates = three.loc[three["candidate_signal"]].sort_values(
        ["uplift", "n"], ascending=[False, False]
    )
    candidates.to_csv(
        out_tables / "candidate_high_risk_interactions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    save_bar_chart(ranks, out_figures / "10_numeric_iv_ranking.png")
    save_cross_heatmap(two, out_figures / "11_ext2_ext3_heatmap.png")
    score_names = [
        "EXT_SOURCE_2", "EXT_SOURCE_3", "CREDIT_INCOME_RATIO",
        "AMT_INCOME_TOTAL", "AMT_CREDIT", "AGE_YEARS"
    ]
    selected = ranks[ranks["feature"].isin(score_names)][["feature", "rank", "iv"]]
    summary = {
        "data_origin": "Home Credit public sample, not bank internal data",
        "sample_size": len(data),
        "total_numeric_features_analyzed": len(ranks),
        "n_woe_bins": len(bins),
        "overall_target_rate": float(target.mean()),
        "two_way_groups": len(two),
        "three_way_groups": len(three),
        "three_way_groups_min500": int(three["min_n_pass"].sum()),
        "candidate_high_risk_groups": len(candidates),
        "top_iv": ranks.head(12)[["feature", "iv", "missing_rate", "small_bin_count"]]
        .to_dict("records"),
        "selected_iv": selected.to_dict("records"),
        "candidate_examples": candidates.head(10).to_dict("records"),
        "thresholds": {
            "min_group_size": MIN_CELL_SIZE,
            "min_risk_uplift": RISK_UPLIFT_MIN,
            "wilson_95_lower_above_overall": True,
            "smoothing": SMOOTHING,
            "quantile_bins": N_BINS,
            "cross_p25_p75": True
        },
    }
    with (output_dir / "woe_iv_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    """Handle CLI entry point for Phase 1 risk-segmentation extension."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    summary = run_extension(args.data, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
