"""Assemble report variables and tables directly from saved numerical evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


MODEL_LABELS = {
    "WOE_Logistic": "WOE逻辑回归",
    "LGBM_Baseline": "LightGBM基线",
    "LGBM_Final_Raw": "优化模型（未校准）",
    "LGBM_Final_Calibrated": "优化模型（已校准）",
    "surrogate_rules": "研究者示例规则",
    "A_conservative": "方案A：审慎",
    "B_balanced": "方案B：平衡",
}


def load_table(root: Path, relative: str) -> pd.DataFrame:
    """Read a generated CSV as a report data source.

    Args:
        root: Repository root.
        relative: Output path relative to the root.

    Returns:
        The complete aggregate result table.
    """
    return pd.read_csv(root / relative)


def display_value(value: object, pattern: str = "text") -> str:
    """Format one report value with explicit units and missing-value handling.

    Args:
        value: Scalar from an audited output table.
        pattern: text, pct, f2, f4, int, million, or hundred_million.

    Returns:
        Human-readable cell text without unsupported citation tokens.
    """
    if pd.isna(value):
        return "-"
    if pattern == "pct":
        return f"{float(value):.2%}"
    if pattern == "int":
        return f"{int(value):,}"
    if pattern == "million":
        return f"{float(value) / 1000000:,.2f}"
    if pattern == "hundred_million":
        return f"{float(value) / 100000000:,.3f}"
    if pattern.startswith("f"):
        return f"{float(value):.{int(pattern[1:])}f}"
    return MODEL_LABELS.get(str(value), str(value))


def report_table(frame: pd.DataFrame, columns: list[tuple]) -> list[list[str]]:
    """Convert an aggregate table to ready-to-render display rows.

    Args:
        frame: Source result rows in desired order.
        columns: Tuples of source field, displayed header, and number format.

    Returns:
        Header row followed by formatted data rows.
    """
    rows = [[header for _, header, _ in columns]]
    for _, record in frame.iterrows():
        rows.append([display_value(record[field], pattern)
                     for field, _, pattern in columns])
    return rows


def build_context(root: Path) -> dict:
    """Bind all PDF claims to current CSV/JSON outputs rather than template numbers.

    Args:
        root: Completed project root.

    Returns:
        Scalar template variables and named display tables for every report.
    """
    summary = json.loads((root / "outputs/run_summary.json").read_text())
    manifest = json.loads((
        root / "outputs/phase2/final_feature_manifest.json").read_text())
    policies = json.loads((root / "outputs/phase3/frozen_policies.json").read_text())
    metrics = load_table(root, "outputs/phase3/test_model_metrics.csv")
    measured = metrics[metrics["operating_point"].eq("policy_KS")].set_index("model")
    strategy = load_table(root, "outputs/phase3/strategy_test_metrics.csv")
    strategy_index = strategy.set_index("strategy")
    correlation = load_table(root, "outputs/phase1/correlation_diagnostics.csv")
    ablation = load_table(root, "outputs/phase3/ext3_validation_ablation.csv")
    importance = load_table(root, "outputs/phase3/shap_global_importance.csv")
    cv = load_table(root, "outputs/phase3/development_cv.csv")
    coefficients = load_table(root, "outputs/phase2/logistic_coefficients.csv")
    intervals = load_table(root, "outputs/phase3/auc_bootstrap_intervals.csv")
    comparison = load_table(root, "outputs/phase4/economic_base_scenario.csv")
    sensitivity = load_table(root, "outputs/phase4/economic_sensitivity.csv")
    final = measured.loc["LGBM_Final_Calibrated"]
    raw = measured.loc["LGBM_Final_Raw"]
    tree = measured.loc["LGBM_Baseline"]
    first = measured.loc["WOE_Logistic"]
    correlation_value = correlation[
        correlation["first"].eq("EXT_SOURCE_2")
        & correlation["second"].eq("EXT_SOURCE_3")
    ].iloc[0]
    ci = intervals.set_index("model").loc["LGBM_Final_Calibrated"]
    context = {
        "n_raw": f"{summary['phase1']['rows']:,}",
        "n_test": f"{int(final['n']):,}",
            "n_features": summary["phase1"]["raw_columns"],
        "target_rate": f"{summary['phase1']['target_rate']:.2%}",
        "employment_sentinel": f"{summary['phase1']['employment_sentinel']:,}",
        "selected_woe": summary["selected_woe_features"],
        "final_feature_count": len(manifest["final_matrix_columns"]),
        "model_auc": f"{final['auc']:.4f}", "model_ks": f"{final['ks']:.4f}",
        "model_gini": f"{final['gini']:.4f}",
            "model_ap": f"{final['average_precision']:.4f}",
        "logistic_auc": f"{first['auc']:.4f}", "baseline_auc": f"{tree['auc']:.4f}",
        "delta_lr": f"{final['auc'] - first['auc']:.4f}",
        "delta_tree": f"{final['auc'] - tree['auc']:.4f}",
        "ci_low": f"{ci['auc_ci_lower']:.4f}", "ci_high": f"{ci['auc_ci_upper']:.4f}",
        "delta_tree_low": f"{ci.get('delta_vs_tree_lower', 0):.5f}",
        "delta_tree_high": f"{ci.get('delta_vs_tree_upper', 0):.5f}",
        "cv_mean": f"{cv['auc'].mean():.4f}", "cv_sd": f"{cv['auc'].std():.4f}",
        "brier_raw": f"{raw['brier']:.6f}", "brier_cal": f"{final['brier']:.6f}",
        "corr_23": f"{correlation_value['pearson']:.4f}",
        "corr_23_n": f"{int(correlation_value['pairwise_n']):,}",
        "ext2_missing": f"{summary['phase1']['ext2_missing']:.2%}",
        "ext3_missing": f"{summary['phase1']['ext3_missing']:.2%}",
        "ablation_without": f"{ablation.iloc[0]['auc']:.4f}",
        "ablation_with": f"{ablation.iloc[1]['auc']:.4f}",
        "ablation_delta": f"{ablation.iloc[1]['auc'] - ablation.iloc[0]['auc']:.4f}",
        "winner": summary["winner"]["name"],
        "winner_tune_auc": f"{summary['winner']['auc']:.4f}",
        "winner_iterations": summary["winner"]["best_iteration"],
        "winner_parameters": json.dumps(summary["winner"]["parameters"],
            ensure_ascii=False),
        "positive_coefficients": int(coefficients["sign_review_required"].sum()),
        "calibration_slope": f"{manifest['calibration_slope']:.4f}",
        "calibration_intercept": f"{manifest['calibration_intercept']:.4f}",
        "score_psi": f"{summary['psi']:.5f}",
        "shap_n": summary["shap"]["global_sample_n"],
        "shap_error": f"{summary['shap']['calibrated_additivity_max_error']:.2e}",
        "sha_short": summary["provenance"]["input_sha256"][:16],
        "python_version": summary["provenance"]["python"],
        "top_shap_names": "、".join(importance.head(5)["feature"].tolist()),
        "scenario_count": len(sensitivity),
    }
    for policy, prefix in zip(policies, ("a", "b")):
        row = strategy_index.loc[policy["name"]]
        baseline_manual = strategy_index.loc["surrogate_rules", "manual_share"]
        manual_reduction = 1 - row["manual_share"] / baseline_manual
        context.update({
            f"{prefix}_lower": f"{policy['lower']:.4%}",
            f"{prefix}_upper": f"{policy['upper']:.4%}",
            f"{prefix}_auto": f"{row['auto_approve_share']:.2%}",
            f"{prefix}_review": f"{row['manual_share']:.2%}",
            f"{prefix}_rate": f"{row['auto_approve_target_rate']:.2%}",
            f"{prefix}_manual_reduction": f"{manual_reduction:.2%}",
        })
    tables: dict = {}
    tables["overview"] = [
        ["数据与方法", "本轮实际口径"],
        ["公开主表", f"{context['n_raw']}条 / {context['n_features']}个原始字段"],
        ["TARGET=1", context["target_rate"] + "；还款困难标签，非机构真实坏账率"],
        ["最终测试集", context["n_test"] + "条；固定随机回溯留出"],
        ["最终模型", f"AUC {context['model_auc']} / KS {context['model_ks']}"],
    ]
    tables["metrics"] = report_table(measured.reset_index()[measured.index.isin([
        "WOE_Logistic", "LGBM_Baseline", "LGBM_Final_Calibrated"
    ])], [
        ("model", "模型", "text"), ("auc", "AUC", "f4"),
        ("ks", "KS", "f4"), ("gini", "Gini", "f4"),
        ("average_precision", "AP", "f4"), ("brier", "Brier", "f5"),
    ])
    tables["operating"] = report_table(metrics[metrics["model"].isin([
        "WOE_Logistic", "LGBM_Final_Calibrated"
    ])], [
        ("model", "模型", "text"), ("operating_point", "口径", "text"),
        ("threshold", "阈值", "f4"), ("precision", "精确率", "pct"),
        ("recall", "召回率", "pct"), ("fp", "FP", "int"), ("fn", "FN", "int"),
    ])
    tables["splits"] = report_table(load_table(root,
        "outputs/phase2/split_counts.csv"), [
        ("partition", "分区", "text"), ("n", "记录数", "int"),
        ("positives", "正例数", "int"), ("target_rate", "正例占比", "pct"),
    ])
    tables["correlations"] = report_table(correlation.iloc[[0, 1, 2, 3, 4, 9]], [
        ("first", "变量一", "text"), ("second", "变量二", "text"),
        ("pairwise_n", "有效样本", "int"), ("pearson", "Pearson r", "f4"),
    ])
    tables["signals"] = report_table(load_table(root,
        "outputs/phase1/signals_holdout_check.csv"), [
        ("EXT_SOURCE_2", "外部评分", "text"), ("EMPLOYMENT_YEARS", "就业时长", "text"),
        ("CREDIT_INCOME_RATIO", "贷款收入比", "text"), ("n", "留出样本", "int"),
        ("target_rate", "标签比例", "pct"), ("risk_lift", "风险倍数", "f2"),
    ])
    ranking = load_table(root, "outputs/phase2/iv_ranking_fit.csv")
    vif = load_table(root, "outputs/phase2/vif_selected_fit.csv")
    selection = ranking.merge(vif, on="feature").merge(coefficients, on="feature")
    tables["selected_features"] = report_table(selection.head(12), [
        ("feature", "入模特征（部分）", "text"), ("iv", "拟合IV", "f4"),
        ("vif", "拟合VIF", "f2"), ("coefficient", "LR系数", "f4"),
    ])
    tables["encoding"] = report_table(load_table(root,
        "outputs/phase2/encoding_imbalance_validation.csv"), [
        ("method", "实验方法", "text"), ("auc", "验证AUC", "f4"),
        ("precision", "P@0.5", "pct"), ("recall", "R@0.5", "pct"),
        ("brier", "Brier", "f4"),
    ])
    iteration = load_table(root, "outputs/phase3/iteration_validation_metrics.csv")
    round_best = iteration.sort_values("auc",
        ascending=False).groupby("stage").head(1).sort_values("stage")
    tables["iterations"] = report_table(round_best, [
        ("name", "每轮最佳实验", "text"), ("auc", "验证AUC", "f4"),
        ("ks", "KS", "f4"), ("brier", "Brier", "f4"),
        ("best_iteration", "树数", "int"),
    ])
    tables["ablation"] = report_table(ablation, [
        ("variant", "验证集对照", "text"), ("auc", "AUC", "f4"),
        ("ks", "KS", "f4"), ("average_precision", "AP", "f4"),
    ])
    tables["shap"] = report_table(importance.head(10), [
        ("feature", "主要模型驱动因素", "text"),
        ("mean_absolute_shap", "平均|SHAP|", "f4"),
        ("spearman_value_vs_shap", "取值/贡献关联", "f4"),
    ])
    tables["local_cases"] = report_table(load_table(root,
        "outputs/phase3/shap_local_cases.csv"), [
        ("case", "案例", "text"), ("score_quantile", "分数分位", "pct"),
        ("predicted_probability", "校准概率", "pct"), ("target", "标签", "int"),
        ("reconstructed_probability", "SHAP重构", "pct"),
    ])
    policy_rows = [{"name": policy["name"], "lower": policy["lower"],
        "upper": policy["upper"],
                    "cap": policy["target_rate_cap"],
                        "met": "满足" if policy["cap_met"] else "未满足"}
                   for policy in policies]
    tables["thresholds"] = report_table(pd.DataFrame(policy_rows), [
        ("name", "策略", "text"), ("lower", "低/中阈值", "pct"),
        ("upper", "中/高阈值", "pct"), ("cap", "低风险上限", "pct"),
        ("met", "策略集约束", "text"),
    ])
    tables["strategies"] = report_table(strategy, [
        ("strategy", "策略", "text"), ("auto_approve_share", "自动通过", "pct"),
        ("manual_share", "人工复核", "pct"), ("auto_reject_share", "高风险拒绝", "pct"),
        ("auto_approve_target_rate", "通过组标签率", "pct"),
    ])
    tables["approval_projection"] = report_table(strategy, [
        ("strategy", "策略", "text"), ("projected_approval_share", "假设最终通过率", "pct"),
        ("projected_approved_target_rate", "获批组标签率代理", "pct"),
        ("high_target_capture", "高风险组覆盖正例", "pct"),
    ])
    fairness = load_table(root, "outputs/phase3/fairness_test_groups.csv")
    tables["fairness"] = report_table(fairness, [
        ("attribute", "审计维度", "text"), ("group", "群体", "text"), ("n", "样本", "int"),
        ("auto_approve_share", "自动通过", "pct"), ("high_risk_tpr", "高风险TPR", "pct"),
        ("high_risk_fpr", "高风险FPR", "pct"),
    ])
    tables["economics"] = report_table(comparison, [
        ("strategy", "假设策略", "text"), ("approved_exposure_cny_assumed", "敞口(亿元)",
            "hundred_million"),
        ("proxy_loss_cny_assumed", "损失代理(百万元)", "million"),
        ("net_value_cny_assumed", "净值(百万元)", "million"),
        ("net_per_100m_approved_exposure", "每亿元净值(百万元)", "million"),
    ])
    ranges = sensitivity.groupby("strategy")["net_difference_to_surrogate"].agg(
        minimum="min", maximum="max", positive_share=lambda values: (values > 0).mean()
    ).reset_index()
    tables["sensitivity"] = report_table(ranges, [
        ("strategy", "相对示例规则", "text"), ("minimum", "最小差额(百万元)", "million"),
        ("maximum", "最大差额(百万元)", "million"), ("positive_share", "正差额情景占比", "pct"),
    ])
    tables["uncertainty"] = report_table(intervals, [
        ("model", "模型", "text"), ("auc", "AUC", "f4"),
        ("auc_ci_lower", "95%下界", "f4"), ("auc_ci_upper", "95%上界", "f4"),
    ])
    context["tables"] = tables
    return context
