"""Score local CSV records with the trusted research model; no lending automation."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from credit_risk.features import build_features
from credit_risk.pipeline import score_partition
from credit_risk.strategy import assign_tiers


def predict_research(raw: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Produce research probabilities, illustrative scores, and two policy tiers.

    Args:
        raw: Application records with the same fields as training, without TARGET.
        bundle: Trusted locally loaded model bundle from this project.

    Returns:
        A frame with probabilities, illustrative PDO scores, and simulation tiers.
    """
    features = build_features(raw)
    probabilities = score_partition(bundle, features)["LGBM_Final_Calibrated"]
    clipped = np.clip(probabilities, 1e-9, 1 - 1e-9)
    score = 600 + (20 / np.log(2)) * (np.log((1 - clipped) / clipped) - np.log(50))
    result = pd.DataFrame({"risk_probability": probabilities,
                           "illustrative_score": score}, index=raw.index)
    if "SK_ID_CURR" in raw:
        result.insert(0, "SK_ID_CURR", raw["SK_ID_CURR"])
    for policy in bundle["policies"]:
        result[policy.name] = assign_tiers(probabilities, policy)
    return result


def main() -> None:
    """Score records into a local, git-ignored private output file.

    Args:
        None; trusted model, input CSV, and output paths come from the CLI.

    Returns:
        None; predictions are written locally, not uploaded or used for decisions.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path,
        default=Path("models/model_bundle.joblib"))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/private/research_predictions.csv"))
    arguments = parser.parse_args()
    bundle = joblib.load(arguments.model)
    raw = pd.read_csv(arguments.input)
    results = predict_research(raw, bundle)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(arguments.output, index=False)


if __name__ == "__main__":
    main()
