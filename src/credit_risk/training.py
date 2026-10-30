"""Development-only model selection, imbalance trials, and final fixed refitting."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_risk.config import RunConfig
from credit_risk.evaluation import binary_metrics, ks_threshold
from credit_risk.features import add_interactions, model_columns
from credit_risk.io_utils import write_json, write_table
from credit_risk.preprocessing import ApplicationPreprocessor
from credit_risk.woe import WoeEncoder, select_features

LOGGER = logging.getLogger(__name__)


@dataclass
class DevelopmentResult:
    """Carry frozen development choices into refitting, never using test labels.

    Attributes:
        winner: Chosen candidate metadata.
        baseline: Fixed LightGBM baseline metadata.
        woe_features: Selected names for logistic regression.
        logistic_c: Logistic regularisation selected on tuning observations.
        experiments: Complete tuning-set comparison table.
    """

    winner: dict
    baseline: dict
    woe_features: list[str]
    logistic_c: float
    experiments: pd.DataFrame


def tree_parameters(config: RunConfig) -> dict:
    """Construct deterministic model defaults independent of outcomes.

    Args:
        config: Run configuration, random seed, and CPU-thread limit.

    Returns:
        LightGBM constructor parameters used by all comparable tree runs.
    """
    return {
        "objective": "binary", "n_estimators": 500, "learning_rate": 0.05,
        "num_leaves": 31, "max_depth": 7, "min_child_samples": 200,
        "reg_lambda": 3.0, "colsample_bytree": 0.85,
        "subsample": 0.9, "subsample_freq": 1, "verbosity": -1,
        "random_state": config.seed, "n_jobs": config.threads,
        "deterministic": True, "force_col_wise": True,
    }


def fit_tree(
    fit_x: pd.DataFrame, fit_y: pd.Series, tune_x: pd.DataFrame,
    tune_y: pd.Series, parameters: dict,
) -> tuple[lgb.LGBMClassifier, dict]:
    """Fit one LightGBM candidate and evaluate solely on the tuning partition.

    Args:
        fit_x: Fitting features with native categorical columns.
        fit_y: Fitting labels.
        tune_x: Frozen-preprocessed tuning features.
        tune_y: Tuning labels for early stopping and comparison.
        parameters: Fixed learner settings for this candidate.

    Returns:
        Trained model and tuning metrics, selected iterations, and runtime.
    """
    start = time.perf_counter()
    model = lgb.LGBMClassifier(**parameters)
    model.fit(
        fit_x, fit_y, eval_set=[(tune_x, tune_y)], eval_metric="auc",
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )
    probability = model.predict_proba(tune_x)[:, 1]
    metrics = binary_metrics(tune_y.to_numpy(), probability)
    threshold = ks_threshold(tune_y.to_numpy(), probability)
    adjusted = binary_metrics(tune_y.to_numpy(), probability, threshold)
    metrics.update({
        "best_iteration": model.best_iteration_,
        "runtime_seconds": time.perf_counter() - start,
        "ks_cutoff": threshold, "recall_at_ks": adjusted["recall"],
        "precision_at_ks": adjusted["precision"],
    })
    return model, metrics


def encoding_trials(
    processed: dict, targets: dict, encoded: dict, selected: list[str], seed: int,
) -> pd.DataFrame:
    """Benchmark WOE, class weighting, SMOTE, and one-hot logistic encodings.

    Args:
        processed: Native feature frames for fit and tune partitions.
        targets: Binary fit and tune labels.
        encoded: Training-only WOE transforms for fit and tune.
        selected: Supervised fitting-only selected WOE columns.
        seed: Reproducibility seed for synthetic resampling.

    Returns:
        Tuning metrics for four encoding/imbalance alternatives.
    """
    rows = []
    fit_x, tune_x = encoded["fit"][selected], encoded["tune"][selected]
    for method in ("woe", "woe_balanced", "woe_smote"):
        data_x, data_y = fit_x, targets["fit"]
        if method == "woe_smote":
            sampler = SMOTE(sampling_strategy=0.30, random_state=seed)
            data_x, data_y = sampler.fit_resample(data_x, data_y)
        weight = "balanced" if method == "woe_balanced" else None
        learner = LogisticRegression(C=1.0, max_iter=1000, class_weight=weight)
        learner.fit(data_x, data_y)
        probability = learner.predict_proba(tune_x)[:, 1]
        row = binary_metrics(targets["tune"].to_numpy(), probability)
        row.update({"method": method, "resampled_n": len(data_y)})
        rows.append(row)
    numerical = processed["fit"].select_dtypes(include="number").columns.tolist()
    categorical = [name for name in processed["fit"] if name not in numerical]
    transform = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numerical),
        ("categories", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    learner = Pipeline([
        ("preprocessing", transform),
        ("logistic", LogisticRegression(C=0.1, max_iter=1000, solver="lbfgs")),
    ])
    learner.fit(processed["fit"], targets["fit"])
    probability = learner.predict_proba(processed["tune"])[:, 1]
    row = binary_metrics(targets["tune"].to_numpy(), probability)
    row.update({"method": "numeric_scaled_plus_one_hot", "resampled_n": len(fit_x)})
    rows.append(row)
    return pd.DataFrame(rows)


def develop_models(
    features: pd.DataFrame, indices: dict, config: RunConfig,
) -> DevelopmentResult:
    """Complete Stage 2 and Stage 3 development without accessing test outcomes.

    Args:
        features: Raw and engineered application fields including TARGET.
        indices: Predeclared disjoint partition indices.
        config: Output destinations and experiment controls.

    Returns:
        Frozen model and feature choices selected only from fitting/tuning data.
    """
    p2, p3 = config.root / "outputs/phase2", config.root / "outputs/phase3"
    columns = model_columns(features)
    preprocessor = ApplicationPreprocessor().fit(features.loc[indices["fit"], columns])
    processed = {
        part: preprocessor.transform(features.loc[indices[part], columns])
        for part in ("fit", "tune")
    }
    target = {part: features.loc[indices[part], "TARGET"] for part in processed}
    encoder = WoeEncoder().fit(processed["fit"], target["fit"])
    encoded = {part: encoder.transform(frame) for part, frame in processed.items()}
    selected, screening, vif = select_features(encoded["fit"], encoder.ranking)
    write_table(pd.DataFrame(preprocessor.statistics), p2 / "preprocessing_fit.csv")
    write_table(encoder.ranking, p2 / "iv_ranking_fit.csv")
    write_table(encoder.bin_table, p2 / "woe_bins_fit.csv")
    write_table(screening, p2 / "feature_screening.csv")
    write_table(vif, p2 / "vif_selected_fit.csv")
    write_table(pd.DataFrame({"feature": selected}), p2 / "selected_woe_features.csv")
    logistic_trials = []
    for c_value in (0.1, 1.0, 10.0):
        learner = LogisticRegression(C=c_value, max_iter=1000)
        learner.fit(encoded["fit"][selected], target["fit"])
        probability = learner.predict_proba(encoded["tune"][selected])[:, 1]
        row = binary_metrics(target["tune"].to_numpy(), probability)
        row["c"] = c_value
        logistic_trials.append(row)
    trials = pd.DataFrame(logistic_trials)
    logistic_c = float(trials.sort_values("auc", ascending=False).iloc[0]["c"])
    write_table(trials, p2 / "logistic_tuning.csv")
    LOGGER.info("WOE screening selected %d features; C=%s", len(selected), logistic_c)
    base = tree_parameters(config)
    records, choices = [], []

    def experiment(name: str, stage: str, settings: dict,
                   matrices: dict, interactions: bool = False) -> dict:
        """Run and record one declared tuning-only tree experiment.

        Args:
            name: Unique experiment identifier.
            stage: Iteration round.
            settings: LightGBM constructor parameters.
            matrices: Fit and tune feature frames.
            interactions: Whether cross-business features were included.

        Returns:
            Serializable selection metadata and tuning results.
        """
        LOGGER.info("Training %s", name)
        _, metrics = fit_tree(
            matrices["fit"], target["fit"], matrices["tune"], target["tune"], settings
        )
        parameters = dict(settings)
        parameters["n_estimators"] = int(metrics["best_iteration"])
        result = {
            "name": name, "stage": stage, "interactions": interactions,
            "parameters": parameters, **metrics,
        }
        records.append({key: value for key,
            value in result.items() if key != "parameters"})
        choices.append(result)
        return result

    baseline = experiment("R0_baseline", "R0", base, processed)
    grid = ParameterGrid({
        "learning_rate": [0.03, 0.06], "num_leaves": [15, 31],
        "reg_lambda": [1.0, 5.0],
    })
    grid_choices = []
    for number, parameters in enumerate(grid, 1):
        settings = {**base, **parameters}
        settings["max_depth"] = 5 if parameters["num_leaves"] == 15 else 7
        grid_choices.append(experiment(f"R1_grid_{number:02d}", "R1", settings,
            processed))
    grid_winner = max(grid_choices, key=lambda entry: entry["auc"])
    for weight in (3.0, float((target["fit"] == 0).sum() / target["fit"].sum())):
        settings = {**grid_winner["parameters"], "n_estimators": 500}
        settings["scale_pos_weight"] = weight
        experiment(f"R2_weight_{weight:.2f}", "R2", settings, processed)
    enriched = add_interactions(features)
    enriched_columns = model_columns(enriched)
    enriched_preprocessor = ApplicationPreprocessor().fit(
        enriched.loc[indices["fit"], enriched_columns]
    )
    enriched_frames = {
        part: enriched_preprocessor.transform(enriched.loc[indices[part],
            enriched_columns])
        for part in ("fit", "tune")
    }
    settings = {**grid_winner["parameters"], "n_estimators": 500}
    experiment("R3_business_interactions", "R3", settings, enriched_frames, True)
    winner = max(choices, key=lambda entry: (entry["auc"], -entry["brier"]))
    # EXT_SOURCE_3 is a diagnostic challenger, not an allowed final predictor.
    ablation_columns = model_columns(features, include_ext3=True)
    ablation_preprocessor = ApplicationPreprocessor().fit(
        features.loc[indices["fit"], ablation_columns]
    )
    ablation_frames = {
        part: ablation_preprocessor.transform(features.loc[indices[part],
            ablation_columns])
        for part in ("fit", "tune")
    }
    _, ablation_metrics = fit_tree(
        ablation_frames["fit"], target["fit"], ablation_frames["tune"], target["tune"],
        {**grid_winner["parameters"], "n_estimators": 500},
    )
    write_table(pd.DataFrame([
        {"variant": "EXT_SOURCE_2_only",
            **{key: grid_winner[key] for key in ablation_metrics}},
        {"variant": "EXT_SOURCE_2_plus_3", **ablation_metrics},
    ]), p3 / "ext3_validation_ablation.csv")
    write_table(encoding_trials(processed, target, encoded, selected, config.seed),
                p2 / "encoding_imbalance_validation.csv")
    table = pd.DataFrame(records)
    write_table(table, p3 / "iteration_validation_metrics.csv")
    write_json({"winner": winner, "baseline": baseline, "logistic_c": logistic_c,
                "selection_rule": "maximum tuning AUC; lower Brier breaks ties",
                "ext3_excluded_by_design": True}, p3 / "frozen_model_selection.json")
    return DevelopmentResult(winner, baseline, selected, logistic_c, table)


def refit_models(
    features: pd.DataFrame, indices: dict, development: DevelopmentResult,
    config: RunConfig,
) -> dict:
    """Refit frozen choices on fit+tune, leaving calibration and policy separate.

    Args:
        features: Engineered application features and TARGET.
        indices: Disjoint predeclared partitions.
        development: Choices fixed before test metrics are calculated.
        config: Output root and deterministic controls.

    Returns:
        Self-contained predictors and frozen preprocessing objects, with no raw data.
    """
    refit_index = np.r_[indices["fit"], indices["tune"]]
    columns = model_columns(features)
    preprocessor = ApplicationPreprocessor().fit(features.loc[refit_index, columns])
    refit_x = preprocessor.transform(features.loc[refit_index, columns])
    refit_y = features.loc[refit_index, "TARGET"]
    encoder = WoeEncoder().fit(refit_x, refit_y)
    selected = development.woe_features
    logistic = LogisticRegression(C=development.logistic_c, max_iter=1000)
    logistic.fit(encoder.transform(refit_x)[selected], refit_y)
    baseline = lgb.LGBMClassifier(**development.baseline["parameters"])
    baseline.fit(refit_x, refit_y)
    final_features = add_interactions(features) if development.winner[
        "interactions"] else features
    final_columns = model_columns(final_features)
    final_preprocessor = ApplicationPreprocessor().fit(
        final_features.loc[refit_index, final_columns]
    )
    final = lgb.LGBMClassifier(**development.winner["parameters"])
    final.fit(final_preprocessor.transform(final_features.loc[refit_index,
        final_columns]), refit_y)
    cal_x = final_preprocessor.transform(final_features.loc[indices["calibration"],
        final_columns])
    margin = final.predict(cal_x, raw_score=True)
    calibrator = LogisticRegression(C=1000000.0, max_iter=1000)
    calibrator.fit(margin.reshape(-1, 1), features.loc[indices["calibration"],
        "TARGET"])
    if calibrator.coef_[0, 0] <= 0:
        raise ValueError("Calibration reversed model ranking; stop and investigate")
    coefficients = pd.DataFrame({
        "feature": selected, "coefficient": logistic.coef_[0],
        "expected_sign": "negative: higher good/bad WOE should reduce risk",
    })
    coefficients["sign_review_required"] = coefficients["coefficient"].gt(0)
    write_table(coefficients, config.root / "outputs/phase2/logistic_coefficients.csv")
    write_table(encoder.bin_table, config.root / "outputs/phase2/woe_bins_refit.csv")
    scorecard = encoder.bin_table.merge(coefficients[["feature", "coefficient"]],
        on="feature")
    scorecard["bin_points"] = -(20 / np.log(2)) * scorecard[
        "coefficient"] * scorecard["woe"]
    write_table(scorecard, config.root / "outputs/phase2/logistic_scorecard.csv")
    write_json({
        "refit_n": len(refit_index), "final_features": final_preprocessor.columns,
        "final_matrix_columns": final.feature_name_, "woe_selected": selected,
        "calibration_slope": float(calibrator.coef_[0, 0]),
        "calibration_intercept": float(calibrator.intercept_[0]),
        "scorecard_base_points": float(600 - (20 / np.log(2)) * (
            np.log(50) + logistic.intercept_[0])),
        "score_scale": "600 points at good:bad odds 50:1, PDO=20; illustrative",
    }, config.root / "outputs/phase2/final_feature_manifest.json")
    return {
        "preprocessor": preprocessor, "woe": encoder, "logistic": logistic,
        "woe_features": selected, "baseline_tree": baseline,
        "final_preprocessor": final_preprocessor, "final_tree": final,
        "calibrator": calibrator, "interactions": development.winner["interactions"],
        "source": "Home Credit public data; research only", "version": "3.0.0",
    }


def development_cv(
    features: pd.DataFrame, indices: dict, chosen: dict, config: RunConfig,
) -> pd.DataFrame:
    """Perform three-fold stability diagnostics entirely inside fit+tune data.

    Args:
        features: Engineered data; test, calibration, and policy are excluded.
        indices: Declared partition indices.
        chosen: Frozen final-model metadata.
        config: Seed and destination settings.

    Returns:
        Per-fold AUC and related metrics, not used to reselect the model.
    """
    subset = features.loc[np.r_[indices["fit"], indices["tune"]]]
    if chosen["interactions"]:
        subset = add_interactions(subset)
    columns = model_columns(subset)
    folds = StratifiedKFold(3, shuffle=True, random_state=config.seed + 10)
    rows = []
    for number, (train, valid) in enumerate(folds.split(subset, subset["TARGET"]), 1):
        transformer = ApplicationPreprocessor().fit(subset.iloc[train][columns])
        learner = lgb.LGBMClassifier(**chosen["parameters"])
        learner.fit(transformer.transform(subset.iloc[train][columns]),
            subset.iloc[train]["TARGET"])
        probability = learner.predict_proba(transformer.transform(subset.iloc[
            valid][columns]))[:, 1]
        metrics = binary_metrics(subset.iloc[valid]["TARGET"].to_numpy(), probability)
        rows.append({"fold": number, **metrics})
    result = pd.DataFrame(rows)
    write_table(result, config.root / "outputs/phase3/development_cv.csv")
    return result
