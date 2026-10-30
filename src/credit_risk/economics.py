"""Explicit hypothetical economic scenarios; no measured bank ROI is claimed."""

from __future__ import annotations

from dataclasses import asdict, replace
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from credit_risk.config import EconomicAssumptions
from credit_risk.io_utils import write_json, write_table


def economic_case(
    tiers: np.ndarray, target: np.ndarray, assumptions: EconomicAssumptions,
) -> dict:
    """Value a normalised hypothetical cohort under explicit proxy assumptions.

    Args:
        tiers: Low, review, and high labels from a frozen research strategy.
        target: Historical payment-difficulty label, NOT verified annual default.
        assumptions: Declared yield, funding, severity, review, and mapping inputs.

    Returns:
        Hypothetical expected approvals, exposure, proxy loss, costs, and net value.

    Raises:
        ValueError: Assumptions violate probability or exposure constraints.
    """
    if not 0 <= assumptions.manual_acceptance <= 1:
        raise ValueError("manual_acceptance must be in [0, 1]")
    if not 0 <= assumptions.lgd <= 1 or assumptions.unit_exposure <= 0:
        raise ValueError("LGD must be in [0, 1] and exposure must be positive")
    low, review = (tiers == "low").astype(float), (tiers == "review").astype(float)
    approval_weight = low + review * assumptions.manual_acceptance
    exposure_weight = (
        low + review * assumptions.manual_acceptance * assumptions.manual_limit
    )
    scale = assumptions.applications / len(target)
    exposure = exposure_weight.sum() * assumptions.unit_exposure * scale
    proxy_event_exposure = (
        np.sum(target * exposure_weight) * assumptions.unit_exposure * scale
    )
    proxy_loss = (
        proxy_event_exposure * assumptions.lgd * assumptions.target_pd_multiplier
    )
    interest_margin = exposure * (assumptions.loan_yield - assumptions.funding_cost)
    review_cost = review.sum() * scale * assumptions.review_cost
    net = interest_margin - proxy_loss - review_cost - assumptions.fixed_cost
    return {
        "applications": assumptions.applications,
        "projected_approved_count": approval_weight.sum() * scale,
        "review_count": review.sum() * scale,
        "approved_exposure_cny_assumed": exposure,
        "interest_margin_cny_assumed": interest_margin,
        "proxy_loss_cny_assumed": proxy_loss,
        "manual_cost_cny_assumed": review_cost,
        "fixed_cost_cny_assumed": assumptions.fixed_cost,
        "net_value_cny_assumed": net,
        "proxy_loss_rate_on_exposure": proxy_loss / exposure if exposure else np.nan,
        "net_per_100m_approved_exposure": (
            net / exposure * 100000000 if exposure else np.nan
        ),
        "lgd": assumptions.lgd,
        "target_pd_multiplier": assumptions.target_pd_multiplier,
        "manual_acceptance": assumptions.manual_acceptance,
        "net_margin_rate": assumptions.loan_yield - assumptions.funding_cost,
    }


def run_economics(strategies: dict, target: np.ndarray, destination: Path) -> dict:
    """Evaluate one base scenario and a complete, disclosed sensitivity grid.

    Args:
        strategies: Strategy-name to tier-label arrays.
        target: Held-out historical TARGET outcomes.
        destination: Stage 4 output folder.

    Returns:
        Base-case rows, comparison rows, and the assumptions used.
    """
    base = EconomicAssumptions()
    rows, sensitivity = [], []
    for name, tiers in strategies.items():
        assumptions = (
            replace(base, fixed_cost=0.0) if name == "surrogate_rules" else base
        )
        rows.append({"strategy": name, **economic_case(tiers, target, assumptions)})
    table = pd.DataFrame(rows)
    baseline_value = table.loc[
        table["strategy"].eq("surrogate_rules"), "net_value_cny_assumed"
    ].iloc[0]
    table["net_difference_to_surrogate"] = (
        table["net_value_cny_assumed"] - baseline_value
    )
    for lgd, multiplier, acceptance, margin in product(
        (0.30, 0.45, 0.60), (0.5, 1.0, 1.5), (0.4, 0.6, 0.8), (0.04, 0.07, 0.10)
    ):
        for name, tiers in strategies.items():
            settings = replace(
                base, lgd=lgd, target_pd_multiplier=multiplier,
                manual_acceptance=acceptance, loan_yield=base.funding_cost + margin,
                fixed_cost=0.0 if name == "surrogate_rules" else base.fixed_cost,
            )
            sensitivity.append(
                {"strategy": name, **economic_case(tiers, target, settings)}
            )
    grid = pd.DataFrame(sensitivity)
    keys = ["lgd", "target_pd_multiplier", "manual_acceptance", "net_margin_rate"]
    baseline = grid[grid["strategy"].eq("surrogate_rules")][
        keys + ["net_value_cny_assumed"]
    ]
    baseline = baseline.rename(columns={"net_value_cny_assumed": "surrogate_net"})
    grid = grid.merge(baseline, on=keys)
    grid["net_difference_to_surrogate"] = (
        grid["net_value_cny_assumed"] - grid["surrogate_net"]
    )
    write_table(table, destination / "economic_base_scenario.csv")
    write_table(grid, destination / "economic_sensitivity.csv")
    write_json(asdict(base), destination / "economic_assumptions.json")
    summary = {"base": table.to_dict("records"), "assumptions": asdict(base),
               "scenario_count": len(grid),
               "warning": "TARGET-to-annual-PD mapping is unverified; not realised ROI"}
    write_json(summary, destination / "summary.json")
    return summary
