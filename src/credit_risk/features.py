"""Application-only feature definitions and explicit unavailable-data boundaries."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Remove sensitive direct attributes from model inputs, but retain for auditing.
GOVERNANCE_EXCLUSIONS = {
    "SK_ID_CURR", "TARGET", "CODE_GENDER", "DAYS_BIRTH", "AGE_YEARS",
    "NAME_FAMILY_STATUS", "CNT_CHILDREN", "EXT_SOURCE_1", "EXT_SOURCE_3",
}
DERIVED_REPLACEMENTS = {
    "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH",
    "DAYS_LAST_PHONE_CHANGE",
}


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide only by positive denominators without mutating either input.

    Args:
        numerator: Numerator indexed by application.
        denominator: Denominator; nonpositive or missing entries are invalid.

    Returns:
        Ratio with invalid or nonfinite values replaced by NaN.
    """
    result = numerator / denominator.where(denominator.gt(0))
    return result.replace([np.inf, -np.inf], np.nan)


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Add application-level proxies; never invent unavailable account history.

    Args:
        raw: Uploaded Home Credit application records, with optional TARGET.

    Returns:
        A copied frame containing raw fields and documented derived features.

    Raises:
        ValueError: A required application field is absent.
    """
    needed = {"AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_ANNUITY", "DAYS_EMPLOYED"}
    missing = needed.difference(raw.columns)
    if missing:
        raise ValueError(f"Missing required application fields: {sorted(missing)}")
    data = raw.copy()
    employment = data["DAYS_EMPLOYED"]
    data["EMPLOYED_SENTINEL"] = employment.eq(365243).astype("int8")
    data["EMPLOYMENT_YEARS"] = -employment.mask(employment.eq(365243)) / 365.25
    data["CREDIT_INCOME_RATIO"] = safe_ratio(
        data["AMT_CREDIT"], data["AMT_INCOME_TOTAL"]
    )
    data["ANNUITY_INCOME_RATIO"] = safe_ratio(
        data["AMT_ANNUITY"], data["AMT_INCOME_TOTAL"]
    )
    data["ANNUITY_CREDIT_RATIO"] = safe_ratio(
        data["AMT_ANNUITY"], data["AMT_CREDIT"]
    )
    if "AMT_GOODS_PRICE" in data:
        data["CREDIT_GOODS_RATIO"] = safe_ratio(
            data["AMT_CREDIT"], data["AMT_GOODS_PRICE"]
        )
    if "CNT_FAM_MEMBERS" in data:
        data["INCOME_PER_MEMBER"] = safe_ratio(
            data["AMT_INCOME_TOTAL"], data["CNT_FAM_MEMBERS"]
        )
    day_fields = {
        "DAYS_BIRTH": "AGE_YEARS",
        "DAYS_REGISTRATION": "REGISTRATION_YEARS",
        "DAYS_ID_PUBLISH": "ID_AGE_YEARS",
        "DAYS_LAST_PHONE_CHANGE": "PHONE_CHANGE_YEARS",
    }
    for original, derived in day_fields.items():
        if original in data:
            data[derived] = -data[original] / 365.25
    document_columns = [c for c in data if c.startswith("FLAG_DOCUMENT_")]
    if document_columns:
        data["DOCUMENT_COUNT"] = data[document_columns].sum(axis=1)
    return data.replace([np.inf, -np.inf], np.nan)


def model_columns(data: pd.DataFrame, include_ext3: bool = False) -> list[str]:
    """Apply the declared governance and redundant-field exclusion policy.

    Args:
        data: Frame returned by build_features.
        include_ext3: Include EXT_SOURCE_3 for validation-only ablation.

    Returns:
        Ordered candidate names; selection uses no outcomes or correlations.
    """
    exclusions = GOVERNANCE_EXCLUSIONS | DERIVED_REPLACEMENTS
    if include_ext3:
        exclusions = exclusions - {"EXT_SOURCE_3"}
    return [
        c for c in data.columns
        if c not in exclusions and not c.endswith(("_MODE", "_MEDI"))
    ]


def add_interactions(data: pd.DataFrame) -> pd.DataFrame:
    """Add cross-business features without using labels or learnt cut-points.

    Args:
        data: Engineered application data containing the three core dimensions.

    Returns:
        Copied data with four interaction features; NaNs remain explicit.
    """
    result = data.copy()
    score = 1 - result["EXT_SOURCE_2"]
    credit = result["CREDIT_INCOME_RATIO"].clip(0, 30)
    tenure = result["EMPLOYMENT_YEARS"].clip(0, 50)
    result["EXT2_RISK_X_CREDIT_INCOME"] = score * credit
    result["EXT2_RISK_X_TENURE"] = score * np.log1p(tenure)
    result["CREDIT_INCOME_PER_TENURE"] = credit / (1 + tenure)
    result["EXT2_RISK_X_ANNUITY_INCOME"] = (
        score * result["ANNUITY_INCOME_RATIO"].clip(0, 3)
    )
    return result
