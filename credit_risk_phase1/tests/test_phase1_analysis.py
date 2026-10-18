"""Unit tests for the stage-one data-quality functions."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phase1_analysis import classify_field, grouped_rates, missing_level


def test_missing_tiers() -> None:
    """Respect the documented thresholds, including boundary values."""
    assert missing_level(0.0999) == "低缺失 <10%"
    assert missing_level(0.10) == "中缺失 10%-30%"
    assert missing_level(0.30) == "中缺失 10%-30%"
    assert missing_level(0.3001) == "高缺失 >30%"


def test_field_classification() -> None:
    """Map familiar Home Credit fields to stable business domains."""
    assert classify_field("SK_ID_CURR") == "标识与标签"
    assert classify_field("EXT_SOURCE_2") == "外部信用与查询"
    assert classify_field("AMT_CREDIT") == "申请与财务"


def test_grouped_rate_denominator() -> None:
    """Verify group denominators and label counts."""
    data = pd.DataFrame({"TARGET": [1, 0, 0, 1], "TYPE": ["A", "A", "B", "B"]})
    output = grouped_rates(data, "TYPE", min_n=2)
    assert output.set_index("分组").loc["A", "TARGET为1比例"] == 0.5
    assert output["样本量"].sum() == 4
