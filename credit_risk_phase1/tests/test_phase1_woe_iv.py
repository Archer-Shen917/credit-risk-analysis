"""Unit tests for numeric bins, smoothed IV and high-risk intersection rules."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phase1_woe_iv import (
    add_derived_features,
    calculate_woe_iv,
    cross_risk_table,
    make_numeric_bins,
    wilson_lower,
)


def test_woe_sums_and_missing() -> None:
    """IV is non-negative, smoothing finite and coverage includes missing."""
    variable = pd.Series([1, 2, 3, 4, None, 1, 2, 3], name="SAMPLE")
    target = pd.Series([0, 0, 1, 1, 0, 0, 1, 1])
    bins, rank = calculate_woe_iv(variable, target)
    assert bins["n"].sum() == len(variable)
    assert np.isfinite(bins["woe"]).all()
    assert np.isclose(bins["iv_bin"].sum(), rank["iv"])
    assert rank["iv"] >= 0
    assert "Missing" in bins["bin"].values


def test_constant_bin() -> None:
    """Identical values produce a valid single bin with IV approximately zero."""
    variable = pd.Series([3.0] * 10, name="CONSTANT")
    target = pd.Series([0, 1] * 5)
    bins, rank = calculate_woe_iv(variable, target)
    assert len(bins) == 1
    assert abs(rank["iv"]) < 1e-10


def test_sentinel_employment_removed() -> None:
    """Sentinel employment must not become a plausible positive employment age."""
    frame = pd.DataFrame({
        "SK_ID_CURR": [1, 2], "TARGET": [1, 0],
        "DAYS_EMPLOYED": [365243, -730], "DAYS_BIRTH": [-10000, -13000],
        "AMT_CREDIT": [50, 100], "AMT_ANNUITY": [5, 10],
        "AMT_INCOME_TOTAL": [10, 20]
    })
    numeric = add_derived_features(frame)
    assert pd.isna(numeric.loc[0, "EMPLOYMENT_YEARS"])
    assert numeric.loc[1, "EMPLOYMENT_YEARS"] > 0


def test_risk_support_gate() -> None:
    """High-risk claims must be filtered when any cell has fewer than 500 rows."""
    frame = pd.DataFrame({"SEGMENT": ["A"] * 4 + ["B"] * 4,
                          "TARGET": [1, 1, 1, 0, 0, 0, 0, 0]})
    table = cross_risk_table(frame, ["SEGMENT"])
    assert not table["candidate_signal"].any()
    assert (table["min_n_pass"] == False).all()


def test_wilson_less_than_observed() -> None:
    """Wilson lower bounds cannot exceed observed frequencies."""
    lows = wilson_lower(pd.Series([5, 50]), pd.Series([10, 100]))
    assert (lows < pd.Series([0.5, 0.5])).all()


def test_make_bins_missing_all() -> None:
    """An entirely null input is represented as Missing, not an exception."""
    result = make_numeric_bins(pd.Series([np.nan, np.nan], name="NULL"))
    assert result.eq("Missing").all()
