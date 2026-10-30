"""Run stages 1-4 from uploaded application data with a recorded protocol."""

from __future__ import annotations

import argparse
import logging
import platform
import time
from pathlib import Path

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from credit_risk.config import RunConfig
from credit_risk.economics import run_economics
from credit_risk.evaluation import (
    binary_metrics, calibrated_probability, calibration_table, ks_threshold,
    paired_bootstrap, population_stability,
)
from credit_risk.explanations import explain_final
from credit_risk.features import add_interactions, build_features
from credit_risk.io_utils import file_sha256, write_json, write_table
from credit_risk.plots import evaluation_plots, result_plots
from credit_risk.stage1 import core_bands, risk_cells, run_stage1
from credit_risk.strategy import (
    assign_tiers, fairness_diagnostics, select_policies, summarise_strategy,
    surrogate_rule_tiers,
)
from credit_risk.training import develop_models, development_cv, refit_models

LOGGER = logging.getLogger(__name__)


def make_splits(target: pd.Series, seed: int = 42) -> dict:
    """Reserve 30% test data and separate development, calibration, and policy.

    Args:
        target: Binary labels indexed by raw input row.
        seed: Random seed; every split is class-stratified.

    Returns:
        Five disjoint arrays: fit, tune, calibration, policy, and test.
    """
    development, test = train_test_split(
        target.index.to_numpy(), test_size=0.3, stratify=target, random_state=seed
    )
    fit, other = train_test_split(
        development, test_size=0.3, stratify=target.loc[development],
            random_state=seed + 1
    )
    tune, reserve = train_test_split(
        other, test_size=0.5, stratify=target.loc[other], random_state=seed + 2
    )
    calibration, policy = train_test_split(
        reserve, test_size=0.5, stratify=target.loc[reserve], random_state=seed + 3
    )
    result = {"fit": fit, "tune": tune, "calibration": calibration,
              "policy": policy, "test": test}
    combined = np.concatenate(list(result.values()))
    if len(np.unique(combined)) != len(target) or len(combined) != len(target):
        raise AssertionError("Splits must be disjoint and exhaustive")
    return result


def score_partition(bundle: dict, features: pd.DataFrame) -> dict:
    """Score a partition without referring to its outcomes.

    Args:
        bundle: Refit models and frozen preprocessing/calibration objects.
        features: Engineered data; TARGET, if present, is not used.

    Returns:
        Arrays for logistic, baseline tree, final raw tree, and calibrated tree.
    """
    base_x = bundle["preprocessor"].transform(features)
    woe_x = bundle["woe"].transform(base_x)[bundle["woe_features"]]
    final_data = add_interactions(features) if bundle["interactions"] else features
    final_x = bundle["final_preprocessor"].transform(final_data)
    margin = bundle["final_tree"].predict(final_x, raw_score=True)
    return {
        "WOE_Logistic": bundle["logistic"].predict_proba(woe_x)[:, 1],
        "LGBM_Baseline": bundle["baseline_tree"].predict_proba(base_x)[:, 1],
        "LGBM_Final_Raw": bundle["final_tree"].predict_proba(final_x)[:, 1],
        "LGBM_Final_Calibrated": calibrated_probability(bundle["calibrator"], margin),
    }


def run_pipeline(config: RunConfig) -> dict:
    """Execute and persist all attainable application-only project stages.

    Args:
        config: Local dataset path, root output path, and run controls.

    Returns:
        Run summary with data identity, metrics, policies, and limitations.

    Raises:
        FileNotFoundError: The application CSV is absent.
        ValueError: The supplied labels or identifiers are invalid.
    """
    start = time.perf_counter()
    for folder in ("phase1", "phase2", "phase3", "phase4", "qa"):
        (config.root / "outputs" / folder).mkdir(parents=True, exist_ok=True)
    (config.root / "models").mkdir(parents=True, exist_ok=True)
    if not config.data_path.exists():
        raise FileNotFoundError(
            "Place application_train.csv in data/raw or pass --data")
    raw = pd.read_csv(config.data_path)
    if raw["SK_ID_CURR"].duplicated().any() or not raw["TARGET"].isin([0, 1]).all():
        raise ValueError("Unique IDs and binary nonmissing TARGET are required")
    indices = make_splits(raw["TARGET"], config.seed)
    partition_rows = [
        {"partition": name, "n": len(values),
         "positives": int(raw.loc[values, "TARGET"].sum()),
         "target_rate": float(raw.loc[values, "TARGET"].mean())}
        for name, values in indices.items()
    ]
    write_table(pd.DataFrame(partition_rows),
        config.root / "outputs/phase2/split_counts.csv")
    provenance = {
        "input_filename": config.data_path.name,
        "input_sha256": file_sha256(config.data_path),
        "seed": config.seed, "outer_split": "70% development / 30% retrospective test",
        "refit": "fit+tune only; calibration and policy remain separate",
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "sklearn": sklearn.__version__,
        "lightgbm": lightgbm.__version__, "partitions": partition_rows,
        "known_limitation": "Prior V2 EDA inspected full data; no virgin external test",
    }
    write_json(provenance, config.root / "outputs/phase2/split_manifest.json")
    LOGGER.info("Read %s rows; splits: %s", len(raw), partition_rows)
    description = config.data_path.parent / "HomeCredit_columns_description.csv"
    phase1 = run_stage1(raw, indices, config.root, description)
    features = build_features(raw)
    development = develop_models(features, indices, config)
    LOGGER.info("Frozen winner: %s, tuning AUC %.5f",
                development.winner["name"], development.winner["auc"])
    bundle = refit_models(features, indices, development, config)
    development_cv(features, indices, development.winner, config)
    policy_features = features.loc[indices["policy"]]
    policy_y = policy_features["TARGET"].to_numpy()
    policy_predictions = score_partition(bundle, policy_features)
    p3 = config.root / "outputs/phase3"
    policies = select_policies(policy_predictions["LGBM_Final_Calibrated"],
        policy_y, p3)
    thresholds = {name: ks_threshold(policy_y, values)
                  for name, values in policy_predictions.items()}
    write_json(thresholds, p3 / "frozen_diagnostic_thresholds.json")
    # All selection is above this line. Test evaluation cannot alter the winner.
    test_features = features.loc[indices["test"]]
    target = test_features["TARGET"].to_numpy()
    bands = core_bands(test_features, phase1["boundaries"])
    dimensions = list(phase1["boundaries"])
    candidate_rows = pd.read_csv(
        config.root / "outputs/phase1/candidate_signals_fit.csv"
    )
    confirmation = candidate_rows[dimensions].merge(
        risk_cells(bands, dimensions), on=dimensions, how="left"
    )
    write_table(confirmation, config.root / "outputs/phase1/signals_holdout_check.csv")
    predictions = score_partition(bundle, test_features)
    metric_rows = []
    for name, probability in predictions.items():
        for operating_point, threshold in (("fixed_0.5", 0.5), ("policy_KS",
            thresholds[name])):
            metric_rows.append({"model": name, "operating_point": operating_point,
                                **binary_metrics(target, probability, threshold)})
    metric_table = pd.DataFrame(metric_rows)
    write_table(metric_table, p3 / "test_model_metrics.csv")
    main_predictions = {name: predictions[name] for name in (
        "WOE_Logistic", "LGBM_Baseline", "LGBM_Final_Calibrated"
    )}
    intervals = paired_bootstrap(target, main_predictions,
                                 config.bootstrap_repeats, config.seed)
    write_table(intervals, p3 / "auc_bootstrap_intervals.csv")
    for name, suffix in (("LGBM_Final_Raw", "raw"), ("LGBM_Final_Calibrated", "final")):
        write_table(calibration_table(target, predictions[name]),
                    p3 / f"calibration_{suffix}.csv")
    stability = population_stability(
        policy_predictions["LGBM_Final_Calibrated"],
            predictions["LGBM_Final_Calibrated"]
    )
    write_json(stability, p3 / "score_psi_random_split.json")
    strategy_labels = {"surrogate_rules": surrogate_rule_tiers(test_features)}
    for policy in policies:
        strategy_labels[policy.name] = assign_tiers(
            predictions["LGBM_Final_Calibrated"], policy
        )
    strategies, tier_tables = [], []
    for name, tiers in strategy_labels.items():
        summary, table = summarise_strategy(
            name, tiers, target, predictions["LGBM_Final_Calibrated"]
        )
        strategies.append(summary)
        tier_tables.append(table)
    write_table(pd.DataFrame(strategies), p3 / "strategy_test_metrics.csv")
    write_table(pd.concat(tier_tables, ignore_index=True),
        p3 / "strategy_tiers_test.csv")
    fairness = fairness_diagnostics(
        test_features, predictions["LGBM_Final_Calibrated"], policies[1]
    )
    write_table(fairness, p3 / "fairness_test_groups.csv")
    final_data = add_interactions(test_features) if bundle[
        "interactions"] else test_features
    final_x = bundle["final_preprocessor"].transform(final_data)
    explanation = explain_final(bundle, final_x, target, p3, config.seed)
    economics = run_economics(strategy_labels, target, config.root / "outputs/phase4")
    bundle["policies"] = policies
    bundle["data_sha256"] = provenance["input_sha256"]
    joblib.dump(bundle, config.root / "models/model_bundle.joblib", compress=3)
    bundle["final_tree"].booster_.save_model(str(config.root / "models/lgbm_final.txt"))
    evaluation_plots(target, main_predictions, config.root)
    result_plots(config.root)
    summary = {
        "provenance": provenance, "phase1": phase1,
        "winner": development.winner,
        "selected_woe_features": len(development.woe_features),
        "test_metrics": metric_table.to_dict("records"), "strategies": strategies,
        "shap": explanation, "psi": stability["psi"], "economics": economics,
        "runtime_seconds": time.perf_counter() - start,
        "missing_history_scope": "Application-only; no actual securities data",
    }
    write_json(summary, config.root / "outputs/run_summary.json")
    LOGGER.info("Finished numerical stages in %.1f seconds", summary["runtime_seconds"])
    return summary


def main() -> None:
    """Run the numerical pipeline with command-line settings.

    Args:
        None; options are read from the command line.

    Returns:
        None; results, models, and execution logs are written to disk.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
        default=Path("data/raw/application_train.csv"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--threads", type=int, default=4)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s")
    config = RunConfig(arguments.data, arguments.root, threads=arguments.threads,
                       bootstrap_repeats=arguments.bootstrap)
    run_pipeline(config)


if __name__ == "__main__":
    main()
