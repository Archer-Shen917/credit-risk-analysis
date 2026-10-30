"""Synthetic integration tests; the real data are not required by CI."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from credit_risk.preprocessing import ApplicationPreprocessor
from credit_risk.woe import WoeEncoder


def test_fit_transform_predict_pipeline() -> None:
    """Smoke-test preprocessing, WOE fitting, and logistic prediction end to end.

    Args:
        None.

    Returns:
        None; transformed dimensions and finite probability ranges are checked.
    """
    rng = np.random.default_rng(42)
    data = pd.DataFrame({
        "amount": rng.lognormal(4, 1, 1200),
        "score": rng.uniform(0, 1, 1200),
        "type": rng.choice(["A", "B"], 1200),
    })
    target = pd.Series((data["score"] < rng.uniform(0, 0.8, 1200)).astype(int))
    train, valid = data.iloc[:900], data.iloc[900:]
    transformer = ApplicationPreprocessor().fit(train)
    train_x, valid_x = transformer.transform(train), transformer.transform(valid)
    encoder = WoeEncoder().fit(train_x, target.iloc[:900])
    learner = LogisticRegression(max_iter=500)
    learner.fit(encoder.transform(train_x), target.iloc[:900])
    prediction = learner.predict_proba(encoder.transform(valid_x))[:, 1]
    assert prediction.shape == (300,)
    assert np.isfinite(prediction).all()
    assert ((prediction > 0) & (prediction < 1)).all()
