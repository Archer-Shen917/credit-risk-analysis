"""Unit tests for data safety, WOE, split integrity, policies, and economics."""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from credit_risk.config import EconomicAssumptions, PolicySpec
from credit_risk.economics import economic_case
from credit_risk.evaluation import (
    binary_metrics, calibration_table, population_stability, wilson_interval,
)
from credit_risk.features import (
    add_interactions,
    build_features,
    model_columns,
    safe_ratio,
)
from credit_risk.pipeline import make_splits
from credit_risk.preprocessing import ApplicationPreprocessor
from credit_risk.strategy import assign_tiers, surrogate_rule_tiers
from credit_risk.woe import WoeEncoder, select_features


@pytest.fixture
def application_frame() -> pd.DataFrame:
    """Provide synthetic records; no user or customer data enter the tests.

    Args:
        None.

    Returns:
        Minimal synthetic application table.
    """
    return pd.DataFrame({
        "AMT_CREDIT": [100.0, 200.0, 300.0, 400.0],
        "AMT_INCOME_TOTAL": [50.0, 0.0, np.nan, 100.0],
        "AMT_ANNUITY": [10.0, 20.0, 30.0, 40.0],
        "DAYS_EMPLOYED": [-365.25, 365243, -730.5, -3652.5],
        "DAYS_BIRTH": [-10000, -12000, -16000, -20000],
        "EXT_SOURCE_2": [0.2, 0.7, 0.5, 0.8],
        "EXT_SOURCE_3": [0.4, 0.2, 0.3, 0.4],
        "CODE_GENDER": ["M", "F", "M", "F"],
        "SK_ID_CURR": [1, 2, 3, 4], "TARGET": [0, 1, 0, 1],
    })


def test_safe_ratio_nonpositive() -> None:
    """Reject zero and negative denominator values.

    Args:
        None.

    Returns:
        None; assertions establish the invariant.
    """
    result = safe_ratio(pd.Series([1.0] * 4), pd.Series([1.0, 0.0, -1.0, np.nan]))
    assert result.iloc[0] == 1.0
    assert result.iloc[1:].isna().all()


def test_feature_builder_preserves_input(application_frame: pd.DataFrame) -> None:
    """Check nonmutation and the employment sentinel interpretation.

    Args:
        application_frame: Synthetic pytest fixture.

    Returns:
        None; original data and derived features are checked.
    """
    before = application_frame.copy(deep=True)
    result = build_features(application_frame)
    pd.testing.assert_frame_equal(before, application_frame)
    assert pd.isna(result.loc[1, "EMPLOYMENT_YEARS"])
    assert result.loc[1, "EMPLOYED_SENTINEL"] == 1
    assert result.loc[0, "EMPLOYMENT_YEARS"] == 1.0


def test_required_fields_fail_clearly() -> None:
    """Reject data that cannot support the declared feature definitions.

    Args:
        None.

    Returns:
        None; the exception is verified.
    """
    with pytest.raises(ValueError, match="required"):
        build_features(pd.DataFrame({"unrelated": [1]}))


def test_governance_exclusions(application_frame: pd.DataFrame) -> None:
    """Keep identifiers, target, gender, and the second score out of main models.

    Args:
        application_frame: Synthetic pytest fixture.

    Returns:
        None; both production-candidate and ablation settings are checked.
    """
    features = build_features(application_frame)
    columns = model_columns(features)
    for name in ("SK_ID_CURR", "TARGET", "CODE_GENDER", "AGE_YEARS", "EXT_SOURCE_3"):
        assert name not in columns
    assert "EXT_SOURCE_3" in model_columns(features, include_ext3=True)


def test_interactions_do_not_create_infinity(application_frame: pd.DataFrame) -> None:
    """Verify the safe cross-business feature formulas.

    Args:
        application_frame: Synthetic pytest fixture.

    Returns:
        None; all finite nonmissing numeric values are checked.
    """
    result = add_interactions(build_features(application_frame))
    values = result.select_dtypes(include="number").to_numpy()
    assert not np.isinf(values).any()


def test_preprocessor_frozen_caps() -> None:
    """Ensure extreme validation inputs do not modify learnt training caps.

    Args:
        None.

    Returns:
        None; fit parameters and transformed validation values are verified.
    """
    data = pd.DataFrame({"amount": np.arange(100, dtype=float)})
    transformer = ApplicationPreprocessor().fit(data)
    limits = transformer.parameters["amount"]
    result = transformer.transform(pd.DataFrame({"amount": [1e9]}))
    assert transformer.parameters["amount"] == limits
    assert np.isclose(result.loc[0, "amount"], limits[1])


def test_preprocessor_unknown_category() -> None:
    """Map an unseen category to a reserved training vocabulary entry.

    Args:
        None.

    Returns:
        None; the frozen categorical vocabulary is checked.
    """
    data = pd.DataFrame({"kind": ["a"] * 60 + ["b"] * 60})
    transformer = ApplicationPreprocessor().fit(data)
    result = transformer.transform(pd.DataFrame({"kind": ["new"]}))
    assert str(result.loc[0, "kind"]) == "__OTHER__"


def test_preprocessor_forbids_target() -> None:
    """Prevent accidental target inclusion in preprocessing.

    Args:
        None.

    Returns:
        None; a clear exception is required.
    """
    with pytest.raises(ValueError, match="TARGET"):
        ApplicationPreprocessor().fit(pd.DataFrame({"TARGET": [0, 1]}))


def test_preprocessor_drops_constants() -> None:
    """Remove all-missing and constant columns without changing variable columns.

    Args:
        None.

    Returns:
        None; fitted schema is checked.
    """
    data = pd.DataFrame({"fixed": [1, 1, 1], "empty": [np.nan] * 3, "x": [1, 2, 3]})
    transformer = ApplicationPreprocessor().fit(data)
    assert transformer.columns == ["x"]


def test_woe_sign_and_distribution() -> None:
    """Validate WOE sign, count totals, and probability normalisation.

    Args:
        None.

    Returns:
        None; high observed risk must receive negative WOE.
    """
    data = pd.DataFrame({"x": ["low"] * 10 + ["high"] * 10})
    target = pd.Series([0] * 9 + [1] + [0] + [1] * 9)
    encoder = WoeEncoder().fit(data, target)
    assert encoder.mappings["x"]["high"] < 0
    assert encoder.mappings["x"]["low"] > 0
    assert np.isclose(encoder.bin_table["dist_good"].sum(), 1)
    assert np.isclose(encoder.bin_table["dist_bad"].sum(), 1)
    assert encoder.bin_table["n"].sum() == len(data)


def test_woe_unseen_neutral() -> None:
    """Ensure unseen categorical levels do not use validation labels.

    Args:
        None.

    Returns:
        None; unseen values have neutral zero WOE.
    """
    encoder = WoeEncoder().fit(pd.DataFrame({"x": ["a", "b", "a", "b"]}),
                               pd.Series([0, 1, 0, 1]))
    result = encoder.transform(pd.DataFrame({"x": ["new"]}))
    assert result.iloc[0, 0] == 0


def test_woe_extreme_numeric_values() -> None:
    """Cover values outside training extrema with infinite endpoint intervals.

    Args:
        None.

    Returns:
        None; transformations remain finite and nonmissing.
    """
    data = pd.DataFrame({"x": np.arange(100, dtype=float)})
    encoder = WoeEncoder().fit(data, pd.Series([0, 1] * 50))
    result = encoder.transform(pd.DataFrame({"x": [-1e10, 1e10, np.nan]}))
    assert np.isfinite(result.to_numpy()).all()


@pytest.mark.parametrize("target", [[0, 0, 0], [0, 2, 1], [0, np.nan, 1]])
def test_invalid_woe_target(target: list) -> None:
    """Reject invalid classes and missing targets.

    Args:
        target: Parametrised invalid label values.

    Returns:
        None; a ValueError is expected.
    """
    with pytest.raises(ValueError):
        WoeEncoder().fit(pd.DataFrame({"x": [1, 2, 3]}), pd.Series(target))


def test_screening_removes_duplicate() -> None:
    """Eliminate strongly correlated duplicate WOE predictors.

    Args:
        None.

    Returns:
        None; only the higher-ranked duplicate is retained.
    """
    data = pd.DataFrame({"a": np.arange(100), "b": np.arange(100),
                         "c": np.tile([0, 1], 50)})
    ranking = pd.DataFrame({"feature": ["a", "b", "c"], "iv": [0.2, 0.1, 0.02]})
    chosen, _, vif = select_features(data, ranking)
    assert "a" in chosen and "b" not in chosen
    assert vif["vif"].max() <= 5


def test_splits_are_disjoint() -> None:
    """Confirm stratified 70:30 development/test splitting and disjoint subsets.

    Args:
        None.

    Returns:
        None; coverage, sample counts, and repeatability are checked.
    """
    target = pd.Series([0] * 900 + [1] * 100)
    splits = make_splits(target)
    combined = np.concatenate(list(splits.values()))
    assert len(combined) == len(np.unique(combined)) == 1000
    assert len(splits["test"]) == 300
    assert np.array_equal(make_splits(target)["fit"], splits["fit"])


def test_policy_partition_boundaries() -> None:
    """Test exact lower and upper boundary membership.

    Args:
        None.

    Returns:
        None; every probability has exactly one tier.
    """
    policy = PolicySpec("test", 0.1, 0.3, 0.04, True)
    result = assign_tiers(np.array([0.0, 0.099, 0.1, 0.299, 0.3, 1.0]), policy)
    assert result.tolist() == ["low", "low", "review", "review", "high", "high"]


@pytest.mark.parametrize("lower,upper", [(0.3, 0.1), (-0.1, 0.2), (0.2, 1.2)])
def test_policy_invalid_thresholds(lower: float, upper: float) -> None:
    """Reject invalid policy definitions before they can be used.

    Args:
        lower: Invalid lower threshold candidate.
        upper: Invalid upper threshold candidate.

    Returns:
        None; an exception is required.
    """
    with pytest.raises(ValueError):
        assign_tiers(np.array([0.1]), PolicySpec("bad", lower, upper, 0.04, False))


def test_rule_missing_defaults_review() -> None:
    """Do not interpret missing fields as a low-risk pass.

    Args:
        None.

    Returns:
        None; an all-missing record stays in the review tier.
    """
    data = pd.DataFrame({"EXT_SOURCE_2": [np.nan], "CREDIT_INCOME_RATIO": [np.nan],
                         "EMPLOYMENT_YEARS": [np.nan]})
    assert surrogate_rule_tiers(data)[0] == "review"


def test_metric_perfect_classifier() -> None:
    """Check standard metric identities on a perfect synthetic classifier.

    Args:
        None.

    Returns:
        None; AUC, KS, Gini, and confusion counts are verified.
    """
    result = binary_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert result["auc"] == result["ks"] == result["gini"] == 1
    assert result["tp"] == 2 and result["fp"] == 0


def test_wilson_bounds() -> None:
    """Check sensible confidence limits at ordinary and zero-event proportions.

    Args:
        None.

    Returns:
        None; intervals bracket the observed rate.
    """
    lower, upper = wilson_interval(8, 100)
    assert 0 <= lower < 0.08 < upper <= 1
    assert wilson_interval(0, 100)[1] > 0


def test_psi_identical_distribution() -> None:
    """Confirm identical distributions produce zero PSI.

    Args:
        None.

    Returns:
        None; the stability identity is asserted.
    """
    probability = np.linspace(0, 1, 100)
    assert population_stability(probability, probability)["psi"] == 0


def test_calibration_counts_exhaustive() -> None:
    """Ensure all observations are represented in calibration bins.

    Args:
        None.

    Returns:
        None; counts sum to the evaluation population.
    """
    table = calibration_table(np.tile([0, 1], 50), np.linspace(0.01, 0.99, 100))
    assert table["n"].sum() == 100


def test_economic_identity_and_sensitivity() -> None:
    """Verify the scenario accounting identity and LGD sensitivity.

    Args:
        None.

    Returns:
        None; larger assumed loss severity must not increase net value.
    """
    tiers = np.array(["low", "review", "high", "low"])
    target = np.array([1, 0, 1, 0])
    assumptions = EconomicAssumptions()
    first = economic_case(tiers, target, assumptions)
    second = economic_case(tiers, target, replace(assumptions, lgd=0.60))
    assert second["net_value_cny_assumed"] < first["net_value_cny_assumed"]
    assert np.isclose(first["net_value_cny_assumed"],
                      first["interest_margin_cny_assumed"] - first[
                          "proxy_loss_cny_assumed"]
                      - first["manual_cost_cny_assumed"] - first[
                          "fixed_cost_cny_assumed"])


def test_all_source_functions_documented() -> None:
    """Audit the teacher's required Args/Returns sections and type annotations.

    Args:
        None.

    Returns:
        None; every explicit source function must satisfy the contract.
    """
    root = Path(__file__).parents[1] / "src" / "credit_risk"
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assert ast.get_docstring(tree), str(path)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            documentation = ast.get_docstring(node) or ""
            assert "Args:" in documentation and "Returns:" in documentation, (
                f"{path.name}:{node.name} missing input/output documentation"
            )
            assert node.returns is not None, f"Missing return annotation: {node.name}"
            for argument in node.args.args:
                if argument.arg not in {"self", "cls"}:
                    assert argument.annotation is not None, argument.arg


def test_economic_no_approvals() -> None:
    """Represent the no-approval scenario without a division error.

    Args:
        None.

    Returns:
        None; zero exposure and undefined per-exposure ratios are checked.
    """
    result = economic_case(
        np.array(["high", "high"]),
        np.array([0, 1]),
        EconomicAssumptions(),
    )
    assert result["approved_exposure_cny_assumed"] == 0
    assert np.isnan(result["net_per_100m_approved_exposure"])
