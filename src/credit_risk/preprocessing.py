"""Training-only preprocessing for native-categorical LightGBM and WOE inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd


class ApplicationPreprocessor:
    """Learn caps, imputations, categories, and missing indicators on fit data.

    Attributes:
        columns: Nonconstant, not-all-missing fitted input columns.
        numeric: Fitted numeric column names.
        categories: Training category vocabularies with reserved fallback labels.
        statistics: Per-column preprocessing decisions for audit export.
    """

    def __init__(self) -> None:
        """Initialise an unfitted transformer.

        Args:
            None.

        Returns:
            None.
        """
        self.columns: list[str] = []
        self.numeric: list[str] = []
        self.categories: dict = {}
        self.parameters: dict = {}
        self.statistics: list[dict] = []

    def fit(self, data: pd.DataFrame) -> ApplicationPreprocessor:
        """Learn transformations from training inputs only, without TARGET.

        Args:
            data: Candidate feature frame from the designated fitting partition.

        Returns:
            Fitted transformer.

        Raises:
            ValueError: TARGET is present, or no usable features remain.
        """
        if "TARGET" in data:
            raise ValueError("TARGET must not enter feature preprocessing")
        self.columns = []
        self.numeric = []
        self.categories = {}
        self.parameters = {}
        self.statistics = []
        for name in data:
            series = data[name]
            rate = float(series.isna().mean())
            unique = int(series.nunique(dropna=True))
            record = {"feature": name, "missing_rate": rate, "n_unique": unique}
            if unique < 2:
                record["action"] = "drop_constant_or_all_missing"
                self.statistics.append(record)
                continue
            self.columns.append(name)
            if pd.api.types.is_numeric_dtype(series):
                self.numeric.append(name)
                bounds = (
                    series.quantile([0.005, 0.995]).tolist()
                    if unique > 20 else [-np.inf, np.inf]
                )
                median = float(series.median())
                self.parameters[name] = (bounds[0], bounds[1], median, rate)
                record.update({
                    "kind": "numeric", "cap_lower": bounds[0],
                    "cap_upper": bounds[1], "fill": median,
                    "action": "median" if rate < 0.10 else "NaN_and_indicator",
                })
            else:
                values = series.dropna().astype(str)
                counts = values.value_counts()
                cutoff = max(50, int(len(series) * 0.001))
                kept = sorted(counts[counts >= cutoff].index.tolist())
                mode = str(values.mode().iloc[0])
                self.categories[name] = kept + ["__MISSING__", "__OTHER__"]
                self.parameters[name] = (mode, rate)
                record.update({
                    "kind": "categorical", "retained_levels": len(kept),
                    "action": "mode_or_missing_with_rare_bucket",
                })
            self.statistics.append(record)
        if not self.columns:
            raise ValueError("No usable features after preprocessing")
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply frozen parameters; unseen categories map to __OTHER__.

        Args:
            data: Training, validation, calibration, policy, or test inputs.

        Returns:
            Numeric and pandas-categorical frame with stable column order.

        Raises:
            ValueError: Transformer is unfitted or a required column is absent.
        """
        if not self.columns:
            raise ValueError("Call fit before transform")
        missing = set(self.columns).difference(data.columns)
        if missing:
            raise ValueError(f"Required feature columns absent: {sorted(missing)}")
        result: dict = {}
        for name in self.columns:
            series = data[name]
            if name in self.numeric:
                lower, upper, median, rate = self.parameters[name]
                cleaned = pd.to_numeric(series, errors="coerce")
                cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
                if rate >= 0.10:
                    result[f"{name}__MISSING"] = cleaned.isna().astype("int8")
                clipped = cleaned.clip(lower, upper)
                if rate < 0.10:
                    clipped = clipped.fillna(median)
                result[name] = clipped.astype("float32")
            else:
                mode, rate = self.parameters[name]
                fill = mode if rate < 0.10 else "__MISSING__"
                labels = series.astype("object").where(series.notna(), fill)
                labels = labels.astype(str)
                vocabulary = self.categories[name]
                labels = labels.where(labels.isin(vocabulary), "__OTHER__")
                result[name] = pd.Series(
                    pd.Categorical(labels, categories=vocabulary), index=data.index
                )
        return pd.DataFrame(result, index=data.index)
