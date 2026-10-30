"""Revisit Stage 1: correlation semantics and cross-business exploratory signals."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from credit_risk.evaluation import wilson_interval
from credit_risk.features import build_features
from credit_risk.io_utils import write_json, write_table
from credit_risk.woe import WoeEncoder


def risk_cells(data: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    """Aggregate exploratory label rates with sample-size and uncertainty flags.

    Args:
        data: Frame with TARGET and fixed group labels.
        groups: Group-column names defining mutually exclusive cells.

    Returns:
        Cell counts, risk rates, nominal Wilson intervals, and candidate flags.
    """
    table = data.groupby(groups, observed=True)["TARGET"].agg(
        n="size", positives="sum", target_rate="mean"
    ).reset_index()
    intervals = [wilson_interval(int(row.positives), int(row.n))
                 for row in table.itertuples(index=False)]
    table["ci_lower"] = [item[0] for item in intervals]
    table["ci_upper"] = [item[1] for item in intervals]
    table["risk_lift"] = table["target_rate"] / data["TARGET"].mean()
    table["candidate"] = (
        table["n"].ge(500) & table["risk_lift"].ge(1.4)
        & table["ci_lower"].gt(data["TARGET"].mean())
    )
    return table.sort_values("target_rate", ascending=False)


def core_bands(data: pd.DataFrame, boundaries: dict) -> pd.DataFrame:
    """Apply frozen quartile boundaries to three different business dimensions.

    Args:
        data: Engineered application frame including TARGET.
        boundaries: Feature to fitting-partition [25th, 75th percentile] mapping.

    Returns:
        Frame containing the target and three explicit missing-aware band labels.
    """
    grouped = data[["TARGET"]].copy()
    for feature, limits in boundaries.items():
        bins = pd.cut(
            data[feature], [-np.inf, *limits, np.inf],
            labels=["Low", "Middle", "High"], include_lowest=True,
        )
        grouped[feature] = bins.astype("object").fillna("Missing")
    return grouped


def run_stage1(
    raw: pd.DataFrame, indices: dict, root: Path, description_path: Path
) -> dict:
    """Recompute Stage 1 corrections while preserving original quality findings.

    Args:
        raw: Uploaded application table.
        indices: Disjoint split indices; fit defines all new exploratory bins.
        root: Repository output root.
        description_path: Uploaded field-meaning CSV.

    Returns:
        Summary of correlation diagnostics and training-only signal definitions.
    """
    destination = root / "outputs" / "phase1"
    data = build_features(raw)
    fit = data.loc[indices["fit"]]
    selected = ["EXT_SOURCE_2", "EMPLOYMENT_YEARS", "CREDIT_INCOME_RATIO"]
    boundaries = {name: fit[name].quantile([0.25, 0.75]).tolist()
                  for name in selected}
    grouped = core_bands(fit, boundaries)
    cells = risk_cells(grouped, selected)
    write_table(cells, destination / "cross_business_fit.csv")
    candidates = cells[cells["candidate"]].head(10)
    write_table(candidates, destination / "candidate_signals_fit.csv")
    correlation_fields = selected + ["EXT_SOURCE_3", "TARGET"]
    correlations = []
    for i, first in enumerate(correlation_fields):
        for second in correlation_fields[i + 1:]:
            observed = fit[[first, second]].dropna()
            correlations.append({
                "first": first, "second": second, "pairwise_n": len(observed),
                "pearson": observed[first].corr(observed[second]),
                "spearman": observed[first].corr(observed[second], method="spearman"),
                "scope": "fit partition; pairwise complete observations",
            })
    correlation_table = pd.DataFrame(correlations)
    write_table(correlation_table, destination / "correlation_diagnostics.csv")
    numeric = fit.select_dtypes(include="number").drop(
        columns=["SK_ID_CURR", "TARGET"], errors="ignore"
    )
    numeric["DAYS_EMPLOYED"] = numeric["DAYS_EMPLOYED"].mask(
        numeric["DAYS_EMPLOYED"].eq(365243)
    )
    encoder = WoeEncoder().fit(numeric, fit["TARGET"])
    write_table(encoder.ranking, destination / "numeric_iv_fit.csv")
    write_table(encoder.bin_table, destination / "numeric_woe_bins_fit.csv")
    profile = pd.DataFrame({
        "feature": raw.columns, "dtype": raw.dtypes.astype(str).values,
        "missing_count": raw.isna().sum().values,
        "missing_rate": raw.isna().mean().values,
        "n_unique": raw.nunique().values,
    })
    profile["missing_level"] = pd.cut(
        profile["missing_rate"], [-0.01, 0.10, 0.30, 1],
        labels=["low_le_10pct", "mid_le_30pct", "high_gt_30pct"],
    )
    # Exact 10% belongs to the middle group, consistent with the supplied brief.
    profile.loc[profile["missing_rate"].eq(0.1), "missing_level"] = "mid_le_30pct"
    write_table(profile, destination / "data_quality_profile.csv")
    if description_path.exists():
        try:
            description = pd.read_csv(description_path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            # Supplied metadata contains Windows-1252 punctuation.
            description = pd.read_csv(description_path, encoding="cp1252")
        application = description[description["Table"].str.contains("application_")]
        dictionary = profile.merge(
            application[["Row", "Description"]].drop_duplicates("Row"),
            left_on="feature", right_on="Row", how="left",
        ).drop(columns="Row")
        write_table(dictionary, destination / "field_dictionary.csv")
    summary = {
        "rows": len(raw), "raw_columns": len(raw.columns),
        "positives": int(raw["TARGET"].sum()),
        "target_rate": float(raw["TARGET"].mean()),
        "duplicate_ids": int(raw["SK_ID_CURR"].duplicated().sum()),
        "employment_sentinel": int(raw["DAYS_EMPLOYED"].eq(365243).sum()),
        "max_income": float(raw["AMT_INCOME_TOTAL"].max()),
        "boundaries_fitted_on": "fit", "boundaries": boundaries,
        "fit_rows": len(fit), "numeric_features_profiled": len(numeric.columns),
        "candidate_signals": len(candidates),
        "ext2_missing": float(raw["EXT_SOURCE_2"].isna().mean()),
        "ext3_missing": float(raw["EXT_SOURCE_3"].isna().mean()),
        "full_data_seen_previously": True,
        "missing_historical_tables": [
            "bureau.csv", "bureau_balance.csv", "previous_application.csv",
            "POS_CASH_balance.csv", "credit_card_balance.csv",
            "installments_payments.csv",
        ],
    }
    write_json(summary, destination / "summary.json")
    return summary
