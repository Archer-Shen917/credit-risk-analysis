"""Configuration contracts and fixed, pre-evaluation research assumptions."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RunConfig:
    """Store paths and reproducible experiment controls.

    Attributes:
        data_path: Local application_train.csv; never committed to the repository.
        root: Repository root containing outputs, models, and reports.
        seed: Seed used by splitting, learners, and bootstrap inference.
        threads: Maximum model CPU threads.
        bootstrap_repeats: Paired, stratified test bootstrap iterations.
    """

    data_path: Path
    root: Path
    seed: int = 42
    threads: int = 4
    bootstrap_repeats: int = 200


@dataclass
class EconomicAssumptions:
    """Explicit hypothetical one-year inputs, not observed bank economics.

    Attributes:
        applications: Number of hypothetical applications to normalise results.
        unit_exposure: Assumed CNY exposure per standard approved application.
        loan_yield: Assumed annual gross yield.
        funding_cost: Assumed annual funding cost.
        lgd: Assumed loss given default under the proxy mapping.
        review_cost: Assumed CNY cost per manually reviewed application.
        manual_acceptance: Assumed fraction approved after manual review.
        manual_limit: Relative exposure for manually approved applications.
        target_pd_multiplier: Hypothetical mapping of TARGET rate to annual PD.
        fixed_cost: Assumed incremental model cost per evaluation cohort.
    """

    applications: int = 10000
    unit_exposure: float = 100000.0
    loan_yield: float = 0.12
    funding_cost: float = 0.05
    lgd: float = 0.45
    review_cost: float = 50.0
    manual_acceptance: float = 0.60
    manual_limit: float = 0.70
    target_pd_multiplier: float = 1.0
    fixed_cost: float = 50000.0


@dataclass
class PolicySpec:
    """Describe a policy selected without test outcomes.

    Attributes:
        name: Human-readable policy identifier.
        lower: Calibrated probability below which an application is low risk.
        upper: Calibrated probability at or above which it is high risk.
        target_rate_cap: Policy-set upper confidence bound constraint.
        cap_met: Whether the declared constraint was achievable on policy data.
        metadata: Supporting selection information.
    """

    name: str
    lower: float
    upper: float
    target_rate_cap: float
    cap_met: bool
    metadata: dict = field(default_factory=dict)
