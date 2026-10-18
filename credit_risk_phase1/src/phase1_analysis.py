"""Phase 1 data quality and EDA for public Home Credit application data.

This is an educational analysis of a public consumer-credit dataset, not
an internal HSBC dataset or a securities margin-lending dataset.
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


FIGURE_DPI = 170
MIN_GROUP_SIZE = 500


def classify_field(column: str) -> str:
    """Categorise application columns by their business purpose."""
    if column in {"SK_ID_CURR", "TARGET"}:
        return "标识与标签"
    if column.startswith(("EXT_SOURCE", "AMT_REQ_CREDIT_BUREAU")):
        return "外部信用与查询"
    if column.startswith((
        "APARTMENTS", "BASEMENT", "YEARS_BEGIN", "YEARS_BUILD",
        "COMMONAREA", "ELEVATORS", "ENTRANCES", "FLOORS",
        "LANDAREA", "LIVING", "NONLIVING", "FONDKAPREMONT",
        "HOUSETYPE", "TOTALAREA", "WALLS", "EMERGENCY",
    )):
        return "住宅与不动产"
    if column.startswith(("REGION", "REG_", "LIVE_", "WEEKDAY", "HOUR_")):
        return "地区与申请渠道"
    if column.startswith(("FLAG_", "DAYS_LAST_PHONE", "OBS_", "DEF_")):
        return "联系方式与其他标记"
    if column.startswith(("AMT_", "NAME_CONTRACT")):
        return "申请与财务"
    if column.startswith((
        "DAYS_", "CNT_", "NAME_", "CODE_", "ORGANIZATION",
        "OCCUPATION", "OWN_CAR",
    )):
        return "人口与就业属性"
    return "其他"


def missing_level(rate: float) -> str:
    """Return the missing-rate tier specified by the project brief."""
    if rate < 0.10:
        return "低缺失 <10%"
    if rate <= 0.30:
        return "中缺失 10%-30%"
    return "高缺失 >30%"


def describe_all_fields(data: pd.DataFrame, info: pd.DataFrame) -> pd.DataFrame:
    """Create complete, matched data dictionary with missingness profiling."""
    descriptions = (
        info[info["Table"].eq("application_{train|test}.csv")]
        .drop_duplicates(subset=["Row"])
        .set_index("Row")
    )
    rows = []
    count = len(data)
    for field in data.columns:
        series = data[field]
        count_missing = int(series.isna().sum())
        miss_rate = count_missing / count
        col_info = descriptions.loc[field] if field in descriptions.index else None
        text = str(col_info["Description"]) if col_info is not None else "字段说明缺失"
        special = col_info["Special"] if col_info is not None else None
        rows.append(
            {
                "字段名": field,
                "业务分类": classify_field(field),
                "数据类型": str(series.dtype),
                "缺失数": count_missing,
                "缺失率": round(miss_rate, 6),
                "缺失等级": missing_level(miss_rate),
                "非空唯一值数": int(series.nunique(dropna=True)),
                "是否常量": "是" if series.nunique(dropna=False) <= 1 else "否",
                "英文释义_源文件": text,
                "特殊说明_源文件": "" if pd.isna(special) else str(special),
                "建议_第一阶段": (
                    "主键/标签：仅做唯一性及取值校验"
                    if field in ("SK_ID_CURR", "TARGET")
                    else (
                        "高缺失：保留并评估业务价值，建议先增加缺失标记"
                        if miss_rate > 0.30
                        else (
                            "中缺失：分析缺失机制，第二阶段加入缺失指示"
                            if miss_rate >= 0.10
                            else "低缺失：检查原因，第二阶段按变量类型处理"
                        )
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def grouped_rates(
    data: pd.DataFrame,
    field: str,
    min_n: int = MIN_GROUP_SIZE
) -> pd.DataFrame:
    """Aggregate sample count and observed difficulty-label rate."""
    values = data[field].astype("object").where(data[field].notna(), "Missing")
    grouped = (
        data.groupby(values, observed=True)["TARGET"]
        .agg(["size", "sum", "mean"])
        .rename(columns={"size": "样本量", "sum": "TARGET为1数", "mean": "TARGET为1比例"})
        .reset_index()
        .rename(columns={field: "分组"})
    )
    if "分组" not in grouped.columns:
        grouped = grouped.rename(columns={grouped.columns[0]: "分组"})
    grouped["变量"] = field
    grouped["样本占比"] = grouped["样本量"] / len(data)
    grouped["是否达到最小样本数"] = grouped["样本量"] >= min_n
    return grouped[["变量", "分组", "样本量", "TARGET为1数", "TARGET为1比例", "样本占比", "是否达到最小样本数"]]


def save_plot(name: str, figure_dir: Path) -> None:
    """Persist current matplotlib figure and release it."""
    plt.tight_layout()
    plt.savefig(figure_dir / f"{name}.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()


def run_analysis(data_path: Path, dictionary_path: Path, output_dir: Path) -> dict:
    """Execute all full-population application-level phase-one checks.

    Args:
        data_path: Path to training application CSV.
        dictionary_path: Path to supplied field description CSV.
        output_dir: Destination for tables, figures and summary JSON.

    Returns:
        Summary of observed QC and EDA metrics.

    Raises:
        FileNotFoundError: An expected input is missing.
        ValueError: Required input columns are unavailable.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    table_dir.mkdir(exist_ok=True)
    figure_dir.mkdir(exist_ok=True)
    for path in (data_path, dictionary_path):
        if not path.is_file():
            raise FileNotFoundError(f"输入文件不存在: {path}")

    data = pd.read_csv(data_path, low_memory=False)
    info = pd.read_csv(dictionary_path, encoding="latin-1")
    if not {"SK_ID_CURR", "TARGET"}.issubset(data.columns):
        raise ValueError("输入缺少 SK_ID_CURR 或 TARGET")

    fields = describe_all_fields(data, info)
    fields.to_csv(table_dir / "field_dictionary.csv", index=False, encoding="utf-8-sig")
    distribution = fields.groupby("缺失等级").size().to_dict()
    missing_top = fields.sort_values("缺失率", ascending=False).head(20)
    missing_top.to_csv(
        table_dir / "top_missing_fields.csv",
        index=False,
        encoding="utf-8-sig"
    )

    target_counts = data["TARGET"].value_counts(dropna=False).sort_index()
    targeted = data.loc[data["TARGET"].eq(1)]
    summary = {
        "input_name": data_path.name,
        "dataset": "Home Credit Default Risk - public competition, not HSBC data",
        "sample_count": int(len(data)),
        "field_count": int(len(data.columns)),
        "positive_count": int(data["TARGET"].eq(1).sum()),
        "positive_rate": float(data["TARGET"].eq(1).mean()),
        "negative_count": int(data["TARGET"].eq(0).sum()),
        "target_values": {str(k): int(v) for k, v in target_counts.items()},
        "target_missing": int(data["TARGET"].isna().sum()),
        "duplicate_ids": int(data["SK_ID_CURR"].duplicated().sum()),
        "duplicate_rows": int(data.duplicated().sum()),
        "missing_cell_count": int(data.isna().sum().sum()),
        "missing_cell_rate": float(data.isna().sum().sum() / data.size),
        "missing_level_counts": distribution,
        "constant_fields": fields.loc[fields["是否常量"].eq("是"), "字段名"].tolist(),
        "description_unmatched": fields.loc[
            fields["英文释义_源文件"].eq("字段说明缺失"),
            "字段名"
        ].tolist(),
        "positive_income_invalid": int(data["AMT_INCOME_TOTAL"].le(0).sum()),
        "zero_or_negative_credit": int(data["AMT_CREDIT"].le(0).sum()),
        "employment_sentinel_365243": int(data["DAYS_EMPLOYED"].eq(365243).sum()),
        "gender_xna": int(data["CODE_GENDER"].eq("XNA").sum()),
        "age_nonnegative": int(data["DAYS_BIRTH"].ge(0).sum()),
        "birth_age_min": round(float((-data["DAYS_BIRTH"] / 365.25).min()), 2),
        "birth_age_max": round(float((-data["DAYS_BIRTH"] / 365.25).max()), 2),
        "income_median": float(data["AMT_INCOME_TOTAL"].median()),
        "credit_median": float(data["AMT_CREDIT"].median()),
        "annuity_median": float(data["AMT_ANNUITY"].median()),
        "income_p99": float(data["AMT_INCOME_TOTAL"].quantile(0.99)),
        "income_max": float(data["AMT_INCOME_TOTAL"].max()),
        "positive_income_label_1_count": int(targeted["AMT_INCOME_TOTAL"].gt(0).sum()),
    }
    # Keep original records unchanged; expose explicit flags for data-quality review.
    income_cutoff = data["AMT_INCOME_TOTAL"].quantile(0.99)
    flags = pd.DataFrame(
        {
            "SK_ID_CURR": data["SK_ID_CURR"],
            "employment_365243_flag": data["DAYS_EMPLOYED"].eq(365243).astype("int8"),
            "gender_xna_flag": data["CODE_GENDER"].eq("XNA").astype("int8"),
            "income_above_p99_flag": (
                data["AMT_INCOME_TOTAL"].gt(income_cutoff).astype("int8")
            ),
            "annuity_missing_flag": data["AMT_ANNUITY"].isna().astype("int8"),
        }
    )
    flags.to_csv(table_dir / "data_quality_flags.csv", index=False)
    summary["income_above_p99_count"] = int(flags["income_above_p99_flag"].sum())
    summary["annuity_missing_count"] = int(flags["annuity_missing_flag"].sum())
    pd.DataFrame(
        [
            [
                "DAYS_EMPLOYED=365243",
                "就业天数特殊值/哨兵值",
                summary["employment_sentinel_365243"],
                "标记，第二阶段单独处理；不可视为真实就业年数"
            ],
            ["CODE_GENDER=XNA", "类别未知值", summary["gender_xna"], "标记；敏感属性不直接用于自动审批"],
            [
                "AMT_INCOME_TOTAL>P99",
                "收入分布极端值（非必然错误）",
                summary["income_above_p99_count"],
                "检查来源与业务合理性；对比稳健性后再决定截尾"
            ],
            ["AMT_ANNUITY缺失", "关键申请金额字段缺失", summary["annuity_missing_count"], "检查缺失机制"],
        ],
        columns=["检查规则", "问题类型", "涉及记录数", "建议动作"],
    ).to_csv(table_dir / "issue_registry.csv", index=False, encoding="utf-8-sig")
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)

    # Derived variables are for exploratory summaries only, not treatment decisions.
    derived = data[[
        "TARGET", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
        "DAYS_BIRTH", "DAYS_EMPLOYED", "EXT_SOURCE_2",
        "NAME_CONTRACT_TYPE", "NAME_INCOME_TYPE",
        "NAME_EDUCATION_TYPE", "CODE_GENDER",
    ]].copy()
    derived["AGE_YEARS"] = -derived["DAYS_BIRTH"] / 365.25
    derived["EMPLOYMENT_YEARS"] = -derived["DAYS_EMPLOYED"] / 365.25
    derived.loc[derived["DAYS_EMPLOYED"].eq(365243), "EMPLOYMENT_YEARS"] = np.nan
    derived["CREDIT_INCOME_RATIO"] = (
        derived["AMT_CREDIT"]
        / derived["AMT_INCOME_TOTAL"].where(derived["AMT_INCOME_TOTAL"].gt(0))
    )
    bins = [0, 1, 2, 4, 8, np.inf]
    labels = ["(0,1]", "(1,2]", "(2,4]", "(4,8]", ">8"]
    derived["CREDIT_INCOME_BUCKET"] = pd.cut(
        derived["CREDIT_INCOME_RATIO"],
        bins=bins,
        labels=labels
    )
    derived["AGE_BUCKET"] = pd.cut(
        derived["AGE_YEARS"],
        bins=[0, 25, 35, 45, 55, 65, 120],
        right=False,
        labels=["<25", "25-34", "35-44", "45-54", "55-64", "65+"]
    )
    derived["EXT_SOURCE_2_BUCKET"] = pd.qcut(
        derived["EXT_SOURCE_2"],
        5,
        duplicates="drop"
    )
    tables = []
    for group in [
        "NAME_CONTRACT_TYPE", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
        "CREDIT_INCOME_BUCKET", "AGE_BUCKET", "EXT_SOURCE_2_BUCKET",
    ]:
        tables.append(grouped_rates(derived, group))
    risk_table = pd.concat(tables, ignore_index=True)
    risk_table.to_csv(
        table_dir / "group_default_rates.csv",
        index=False,
        encoding="utf-8-sig"
    )

    metrics = []
    for column in [
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "CREDIT_INCOME_RATIO",
        "AGE_YEARS",
        "EMPLOYMENT_YEARS",
        "EXT_SOURCE_2"
    ]:
        series = derived[column].replace([np.inf, -np.inf], np.nan)
        metrics.append({
            "变量": column,
            "有效数": int(series.count()),
            "缺失数": int(series.isna().sum()),
            "均值": float(series.mean()),
            "中位数": float(series.median()),
            "P01": float(series.quantile(.01)),
            "P25": float(series.quantile(.25)),
            "P75": float(series.quantile(.75)),
            "P99": float(series.quantile(.99)),
            "最大值": float(series.max())
        })
    pd.DataFrame(metrics).to_csv(
        table_dir / "numeric_profiles.csv",
        index=False,
        encoding="utf-8-sig"
    )

    numeric = data.select_dtypes(include=np.number).drop(
        columns=["SK_ID_CURR"],
        errors="ignore"
    ).copy()
    numeric.loc[numeric["DAYS_EMPLOYED"].eq(365243), "DAYS_EMPLOYED"] = np.nan
    numeric["CREDIT_INCOME_RATIO"] = derived["CREDIT_INCOME_RATIO"]
    numeric["AGE_YEARS"] = derived["AGE_YEARS"]
    numeric["EMPLOYMENT_YEARS"] = derived["EMPLOYMENT_YEARS"]
    correlations = (
        numeric.corr(numeric_only=True)["TARGET"]
        .drop("TARGET")
        .dropna()
        .sort_values()
    )
    correlations.rename("与TARGET的Pearson相关系数").to_csv(
        table_dir / "target_numeric_correlations.csv",
        encoding="utf-8-sig"
    )

    # Graph 1: label distribution.
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    ax.bar(
        ["TARGET=0", "TARGET=1"],
        [summary["negative_count"], summary["positive_count"]],
        color=["#679BC0", "#E69A65"]
    )
    ax.set_ylabel("Applications")
    ax.set_title(
        "Outcome distribution (observed TARGET)",
        loc="left",
        fontweight="bold"
    )
    for index, value in enumerate([
        summary["negative_count"],
        summary["positive_count"]
    ]):
        ax.text(
            index,
            value + 3000,
            f"{value:,} ({value / len(data):.2%})",
            ha="center",
            fontsize=10
        )
    ax.set_ylim(0, summary["negative_count"] * 1.13)
    save_plot("01_target_distribution", figure_dir)

    # Graph 2: missing fields.
    plot_miss = fields.sort_values("缺失率", ascending=False).head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9.5, 6.1))
    ax.barh(plot_miss["字段名"], plot_miss["缺失率"] * 100, color="#679BC0")
    ax.set_xlabel("Missing (%)")
    ax.set_title("Top 15 fields by missing-data rate", loc="left", fontweight="bold")
    ax.set_xlim(0, 100)
    save_plot("02_missing_rate_top15", figure_dir)

    def plot_groups(
        field: str,
        title: str,
        filename: str,
        minimum: int = MIN_GROUP_SIZE
    ) -> None:
        mask = risk_table["变量"].eq(field) & risk_table["样本量"].ge(minimum)
        subset = risk_table.loc[mask].copy()
        subset["分组"] = subset["分组"].astype(str)
        subset = subset.sort_values("TARGET为1比例")
        fig, ax = plt.subplots(figsize=(9.6, max(3.5, 0.42 * len(subset) + 1.8)))
        ax.barh(subset["分组"], subset["TARGET为1比例"] * 100, color="#679BC0")
        ax.axvline(
            summary["positive_rate"] * 100,
            color="#DB8B6E",
            linestyle="--",
            linewidth=1.5,
            label="Overall rate"
        )
        ax.set_xlabel("Share with TARGET=1 (%)")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.legend(loc="lower right")
        for index, (_, row) in enumerate(subset.iterrows()):
            ax.text(
                row["TARGET为1比例"] * 100 + 0.12,
                index,
                f"n={row['样本量']:,}",
                va="center",
                fontsize=8
            )
        ax.set_xlim(0, max(12, subset["TARGET为1比例"].max() * 115))
        save_plot(filename, figure_dir)

    plot_groups(
        "NAME_CONTRACT_TYPE",
        "Observed TARGET rate by loan contract type",
        "03_contract_type"
    )
    plot_groups(
        "NAME_INCOME_TYPE",
        "Observed TARGET rate by income type (n>=500)",
        "04_income_type"
    )
    plot_groups(
        "CREDIT_INCOME_BUCKET",
        "Observed TARGET rate by credit/income ratio",
        "05_credit_income_ratio"
    )
    plot_groups(
        "EXT_SOURCE_2_BUCKET",
        "Observed TARGET rate by EXT_SOURCE_2 quintile",
        "06_ext_source_quintiles"
    )
    plot_groups(
        "AGE_BUCKET",
        "Observed TARGET rate by age (descriptive only)",
        "07_age_descriptive"
    )

    top_names = correlations.abs().sort_values(ascending=False).head(14).index
    top_corr = correlations.reindex(top_names).sort_values()
    fig, ax = plt.subplots(figsize=(9.1, 6.3))
    ax.barh(
        top_corr.index,
        top_corr.values,
        color=["#DB8B6E" if v < 0 else "#679BC0" for v in top_corr.values]
    )
    ax.set_xlabel("Pearson correlation with TARGET (association only)")
    ax.set_title(
        "Top 14 absolute numeric correlations with TARGET",
        loc="left",
        fontweight="bold"
    )
    save_plot("08_target_correlations", figure_dir)

    cross = derived.groupby(
        ["NAME_CONTRACT_TYPE", "CREDIT_INCOME_BUCKET"],
        observed=True
    )["TARGET"].agg(["size", "mean"]).reset_index()
    cross.columns = ["申请类型", "信用额收入比分组", "样本量", "TARGET为1比例"]
    cross.to_csv(
        table_dir / "contract_by_credit_income.csv",
        index=False,
        encoding="utf-8-sig"
    )
    heat = cross.pivot(
        index="申请类型",
        columns="信用额收入比分组",
        values="TARGET为1比例"
    )
    fig, ax = plt.subplots(figsize=(8.8, 3.5))
    image = ax.imshow(heat.values * 100, aspect="auto", cmap="Blues", vmin=0, vmax=15)
    ax.set_xticks(range(heat.shape[1]), [str(value) for value in heat.columns])
    ax.set_yticks(range(heat.shape[0]), heat.index)
    ax.set_xlabel("Credit/income ratio")
    ax.set_title(
        "TARGET rate (%) by contract and credit/income",
        loc="left",
        fontweight="bold"
    )
    for (row_idx, col_idx), value in np.ndenumerate(heat.values):
        if pd.notna(value):
            ax.text(
                col_idx,
                row_idx,
                f"{value:.1%}",
                ha="center",
                va="center",
                fontsize=10
            )
    fig.colorbar(image, ax=ax, label="TARGET rate (%)")
    save_plot("09_cross_segment_heatmap", figure_dir)

    return summary


def main() -> None:
    """Parse command line options and execute the data analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_analysis(args.data, args.dictionary, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
