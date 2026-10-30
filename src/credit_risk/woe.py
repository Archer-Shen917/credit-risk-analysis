"""Frozen WOE encoding, bin audit tables, and supervised feature screening."""

from __future__ import annotations

import numpy as np
import pandas as pd


class WoeEncoder:
    """Fit unsupervised bins and smoothed supervised mappings on fitting data.

    Attributes:
        specifications: Bin boundaries or low-cardinality vocabularies.
        mappings: Fitted log(good distribution / bad distribution).
        bin_table: Auditable per-bin counts, WOE, and IV components.
        ranking: Fitted IV values sorted in descending order.
    """

    def __init__(self, n_bins: int = 5, smoothing: float = 0.5) -> None:
        """Set binning controls.

        Args:
            n_bins: Desired quantile bins for non-discrete numeric variables.
            smoothing: Positive additive count to handle zero event cells.

        Returns:
            None.

        Raises:
            ValueError: Controls fall outside supported ranges.
        """
        if n_bins < 2 or smoothing <= 0:
            raise ValueError("n_bins >= 2 and smoothing > 0 are required")
        self.n_bins = n_bins
        self.smoothing = smoothing
        self.specifications: dict = {}
        self.mappings: dict = {}
        self.bin_table = pd.DataFrame()
        self.ranking = pd.DataFrame()

    def labels(self, series: pd.Series) -> pd.Series:
        """Apply one fitted bin specification, including missing and extremes.

        Args:
            series: Named feature whose specification was fitted previously.

        Returns:
            String bin labels; unseen discrete categories are kept unmapped.
        """
        kind, specification = self.specifications[series.name]
        if kind == "interval":
            values = pd.to_numeric(series, errors="coerce")
            codes = pd.cut(values, specification, labels=False, include_lowest=True)
            return codes.map(
                lambda value: "Missing" if pd.isna(value) else f"B{int(value)}"
            )
        if kind == "modal":
            return series.map(
                lambda value: (
                    "Missing" if pd.isna(value)
                    else "Modal" if value == specification else "Other"
                )
            ).astype(str)
        return series.astype("object").where(series.notna(), "Missing").astype(str)

    def fit(self, data: pd.DataFrame, target: pd.Series) -> WoeEncoder:
        """Estimate bins, class-conditional distributions, and information value.

        Args:
            data: Fitting-partition features; no TARGET column allowed.
            target: Binary payment-difficulty outcomes aligned to data.index.

        Returns:
            Fitted encoder with per-bin and feature-level diagnostics.

        Raises:
            ValueError: Target alignment or binary-class requirements fail.
        """
        if "TARGET" in data or not target.index.equals(data.index):
            raise ValueError("Target must be separate and aligned")
        if not target.isin([0, 1]).all() or target.nunique() != 2:
            raise ValueError("A nonmissing binary target with both classes is required")
        self.specifications, self.mappings = {}, {}
        rows, ranking = [], []
        for name in data:
            series = data[name]
            valid = series.dropna()
            numeric = pd.api.types.is_numeric_dtype(series)
            if numeric and valid.nunique() > self.n_bins:
                values = np.unique(
                    valid.quantile(np.linspace(0, 1, self.n_bins + 1)).values
                )
                if len(values) > 2:
                    edges = np.r_[-np.inf, values[1:-1], np.inf]
                    self.specifications[name] = ("interval", edges)
                else:
                    self.specifications[name] = ("modal", valid.mode().iloc[0])
            else:
                self.specifications[name] = ("category", None)
            labels = self.labels(series)
            counts = pd.DataFrame({"bin": labels, "target": target}).groupby(
                "bin", observed=True
            )["target"].agg(n="size", bad="sum")
            counts["good"] = counts["n"] - counts["bad"]
            k = len(counts)
            alpha = self.smoothing
            counts["dist_good"] = (counts["good"] + alpha) / (
                counts["good"].sum() + alpha * k
            )
            counts["dist_bad"] = (counts["bad"] + alpha) / (
                counts["bad"].sum() + alpha * k
            )
            counts["woe"] = np.log(counts["dist_good"] / counts["dist_bad"])
            counts["iv_bin"] = (
                counts["dist_good"] - counts["dist_bad"]
            ) * counts["woe"]
            counts["target_rate"] = counts["bad"] / counts["n"]
            counts["feature"] = name
            kind, specification = self.specifications[name]
            definitions = []
            for label in counts.index:
                if kind == "interval" and label != "Missing":
                    number = int(label[1:])
                    lower, upper = specification[number:number + 2]
                    definitions.append(f"({lower:.8g}, {upper:.8g}]")
                elif kind == "modal" and label != "Missing":
                    definitions.append(
                        f"value = {specification}" if label == "Modal"
                        else f"value != {specification}"
                    )
                else:
                    definitions.append(str(label))
            counts["bin_definition"] = definitions
            counts["small_cell"] = counts["n"] < 500
            self.mappings[name] = counts["woe"].to_dict()
            rows.append(counts.reset_index())
            ranking.append({
                "feature": name, "iv": float(counts["iv_bin"].sum()),
                "missing_rate": float(series.isna().mean()),
                "n_bins": k, "min_bin_n": int(counts["n"].min()),
            })
        self.bin_table = pd.concat(rows, ignore_index=True)
        self.ranking = pd.DataFrame(ranking).sort_values("iv", ascending=False)
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Encode using fixed training mappings; unseen categories receive zero.

        Args:
            data: Frame containing all fitted columns, without target access.

        Returns:
            Finite float matrix in fitted feature order.

        Raises:
            ValueError: Encoder has not been fitted.
        """
        if not self.mappings:
            raise ValueError("Call fit before transform")
        return pd.DataFrame({
            name: self.labels(data[name]).map(mapping).fillna(0.0)
            for name, mapping in self.mappings.items()
        }, index=data.index).astype("float32")


def select_features(
    encoded: pd.DataFrame, ranking: pd.DataFrame, max_features: int = 45
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    """Screen fitting-only IV, variance, correlation, then iterative VIF.

    Args:
        encoded: WOE matrix from the fitting partition only.
        ranking: Fitted IV ranking containing feature and iv columns.
        max_features: Upper bound before the VIF elimination pass.

    Returns:
        Selected feature names, a complete selection log, and final VIF values.
    """
    selected: list[str] = []
    log: list[dict] = []
    for row in ranking.itertuples(index=False):
        name, iv = row.feature, row.iv
        reason = "selected"
        if iv < 0.005:
            reason = "iv_below_0.005"
        elif encoded[name].var() < 1e-6:
            reason = "near_zero_variance"
        elif len(selected) >= max_features:
            reason = "feature_count_cap"
        elif selected:
            correlations = encoded[selected].corrwith(encoded[name]).abs()
            if correlations.max() >= 0.85:
                reason = f"correlation_with_{correlations.idxmax()}"
        if reason == "selected":
            selected.append(name)
        log.append({"feature": name, "iv": iv, "decision": reason})
    while len(selected) > 1:
        correlation = encoded[selected].corr().values
        inverse = np.linalg.pinv(correlation)
        values = np.diag(inverse)
        # A singular correlation matrix also needs explicit removal.
        if np.linalg.matrix_rank(correlation) < len(selected):
            index = len(selected) - 1
        elif values.max() > 5:
            index = int(values.argmax())
        else:
            break
        removed = selected.pop(index)
        for record in log:
            if record["feature"] == removed:
                record["decision"] = "vif_over_5_or_singular"
    if not selected:
        raise ValueError("No features survived supervised screening")
    final_correlation = encoded[selected].corr().fillna(0).values
    final_vif = np.diag(np.linalg.pinv(final_correlation))
    vif = pd.DataFrame({"feature": selected, "vif": final_vif})
    return selected, pd.DataFrame(log), vif
