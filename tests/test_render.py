"""Unit tests for the evaluation render functions."""

from pathlib import Path

import pandas as pd
import pytest
from test_analysis import (
    data_result_1,
    data_result_2,
    test_score_1,
    test_score_2,
)

from analysis.analysis import (
    DesiredDirection,
    EvaluationResult,
    EvaluationScore,
    EvaluationScoreCI,
    EvaluationScoreComparison,
    EvaluationScoreDataType,
)
from analysis.render import (
    fmt_badge,
    fmt_ci,
    fmt_control_badge,
    fmt_hyperlink,
    fmt_image,
    fmt_metric_value,
    fmt_pvalue,
    fmt_table_ci,
    fmt_table_compare,
    fmt_treatment_badge,
)


def test_fmt_metric_value():
    """Test formatting of metric values."""
    # Test ordinal formatting
    assert fmt_metric_value(1.2345, EvaluationScoreDataType.ORDINAL) == "1.23"

    # Test continuous formatting
    assert fmt_metric_value(0.0012345, EvaluationScoreDataType.CONTINUOUS) == "0.00123"

    # Test boolean formatting (percentage)
    assert fmt_metric_value(0.75, EvaluationScoreDataType.BOOLEAN) == "75.0%"

    # Test with sign
    assert fmt_metric_value(0.75, EvaluationScoreDataType.ORDINAL, sign=True) == "+0.75"


def test_fmt_pvalue():
    """Test formatting of p-values."""
    # Test small p-value
    assert fmt_pvalue(0.0001) == "1e-4"

    # Test regular p-value
    assert fmt_pvalue(0.034) == "0.034"

    # Test zero p-value
    assert fmt_pvalue(0) == "≈0"


def test_fmt_image():
    """Test formatting of image markdown."""
    assert (
        fmt_image("https://example.com/image.png", "Alt text")
        == '![Alt text](https://example.com/image.png "")'
    )


@pytest.mark.parametrize(
    "test_case, text, url, tooltip",
    [
        ("without-tooltip", "GitHub", "https://github.com", ""),
        ("with-tooltip", "GitHub", "https://github.com", "Visit GitHub"),
        ("quotes-tooltip", "GitHub", "https://github.com", 'Visit "GitHub"'),
        ("newline-tooltip", "GitHub", "https://github.com", "Visit\nGitHub"),
    ],
)
def test_fmt_hyperlink(test_case, text, url, tooltip, snapshot):
    """Test formatting of hyperlinks."""
    output = fmt_hyperlink(text, url, tooltip)

    snapshot.snapshot_dir = Path("tests", "snapshots", "fmt_hyperlink")
    snapshot.assert_match(output, f"{test_case}.md")


@pytest.mark.parametrize(
    "test_case, label, message, color, tooltip",
    [
        ("improved-strong", "Improved", "+5.3%", "ImprovedStrong", ""),
        ("improved-weak", "Improved", "+5.3%", "ImprovedWeak", ""),
        ("degraded-strong", "Degraded", "+5.3%", "DegradedStrong", ""),
        ("degraded-weak", "Degraded", "+5.3%", "DegradedWeak", ""),
        ("changed-strong", "Changed", "+5.3%", "ChangedStrong", ""),
        ("changed-weak", "Changed", "+5.3%", "ChangedWeak", ""),
        ("inconclusive", "Inconclusive", "+5.3%", "Inconclusive", ""),
        ("warning", "Zero samples", "0%", "Warning", "My tooltip"),
        ("pass", "Test", "Passed", "Pass", ""),
        ("fail", "Test", "Failed", "Fail", ""),
        ("hex-color", "Hex", "Color", "#4C6CE4", ""),
        ("special-characters", "A_B", "C-D", "Pass", ""),
    ],
)
# pylint: disable-next=too-many-arguments, too-many-positional-arguments
def test_fmt_badge(test_case, label, message, color, tooltip, snapshot):
    """Test formatting of badges."""
    output = fmt_badge(label, message, color, tooltip)

    snapshot.snapshot_dir = Path("tests", "snapshots", "fmt_badge")
    snapshot.assert_match(output, f"{test_case}.md")


@pytest.mark.parametrize(
    "test_case, result_1, result_2",
    [
        (
            "too-few-samples",
            {"inputs.id": [1, 2, 3], "outputs.fluency.score": [0.8, 0.9, 0.85]},
            {"inputs.id": [1, 2, 3], "outputs.fluency.score": [0.6, 0.5, 0.75]},
        ),
        (
            "improve-weak",
            {
                "inputs.id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "outputs.fluency.score": [
                    0.6,
                    0.5,
                    0.75,
                    0.6,
                    0.5,
                    0.75,
                    0.6,
                    0.5,
                    0.75,
                    0.85,
                ],
            },
            {
                "inputs.id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "outputs.fluency.score": [
                    0.8,
                    0.9,
                    0.85,
                    0.8,
                    0.9,
                    0.85,
                    0.8,
                    0.9,
                    0.85,
                    0.85,
                ],
            },
        ),
        (
            "degraded-weak",
            {
                "inputs.id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "outputs.fluency.score": [
                    0.8,
                    0.9,
                    0.85,
                    0.8,
                    0.9,
                    0.85,
                    0.8,
                    0.9,
                    0.85,
                    0.85,
                ],
            },
            {
                "inputs.id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "outputs.fluency.score": [
                    0.6,
                    0.5,
                    0.75,
                    0.6,
                    0.5,
                    0.75,
                    0.6,
                    0.5,
                    0.75,
                    0.85,
                ],
            },
        ),
        (
            "degraded-strong",
            {
                "inputs.id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "outputs.fluency.score": [
                    0.8,
                    0.9,
                    0.85,
                    0.8,
                    0.9,
                    0.85,
                    0.8,
                    0.9,
                    0.85,
                    0.85,
                ],
            },
            {
                "inputs.id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "outputs.fluency.score": [
                    0.1,
                    0.2,
                    0.15,
                    0.1,
                    0.2,
                    0.15,
                    0.1,
                    0.2,
                    0.15,
                    0.15,
                ],
            },
        ),
        (
            "improve-strong",
            {
                "inputs.id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "outputs.fluency.score": [
                    0.1,
                    0.2,
                    0.15,
                    0.1,
                    0.2,
                    0.15,
                    0.1,
                    0.2,
                    0.15,
                    0.15,
                ],
            },
            {
                "inputs.id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "outputs.fluency.score": [
                    0.8,
                    0.9,
                    0.85,
                    0.8,
                    0.9,
                    0.85,
                    0.8,
                    0.9,
                    0.85,
                    0.85,
                ],
            },
        ),
    ],
)
def test_fmt_treatment_badge(test_case, result_1, result_2, snapshot):
    """Test formatting of badges."""

    control_result = EvaluationResult(
        variant="test_variant_1", df_result=pd.DataFrame(result_1)
    )
    treatment_result = EvaluationResult(
        variant="test_variant_2", df_result=pd.DataFrame(result_2)
    )

    comparison = EvaluationScoreComparison(
        control_result, treatment_result, test_score_1
    )

    output = fmt_treatment_badge(comparison)

    snapshot.snapshot_dir = Path("tests", "snapshots", "fmt_treatment_badge")
    snapshot.assert_match(output, f"{test_case}.md")


def test_fmt_control_badge(snapshot):
    """Test formatting of control badges."""

    control_result = EvaluationResult(
        variant="test_variant_1", df_result=pd.DataFrame(data_result_1)
    )
    treatment_result = EvaluationResult(
        variant="test_variant_2", df_result=pd.DataFrame(data_result_2)
    )

    comparison = EvaluationScoreComparison(
        control_result, treatment_result, test_score_1
    )

    output = fmt_control_badge(comparison)

    snapshot.snapshot_dir = Path("tests", "snapshots", "fmt_control_badge")
    snapshot.assert_match(output, "test.md")


@pytest.mark.parametrize(
    "test_case, result, evaluator, score_data_type, expected_contains",
    [
        (
            "too-few-samples",
            {"inputs.id": [1, 2, 3], "outputs.fluency.score": [0.8, 0.9, 0.85]},
            "fluency",
            EvaluationScoreDataType.CONTINUOUS,
            "Too few samples",
        ),
        (
            "not-applicable",
            {
                "inputs.id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "outputs.ordinal.score": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
            },
            "ordinal",
            EvaluationScoreDataType.ORDINAL,
            "N/A",
        ),
        (
            "has-ci",
            {
                "inputs.id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "outputs.fluency.score": [
                    0.8,
                    0.9,
                    0.85,
                    0.8,
                    0.9,
                    0.85,
                    0.8,
                    0.9,
                    0.85,
                    0.85,
                ],
            },
            "fluency",
            EvaluationScoreDataType.CONTINUOUS,
            "0.821",  # Just check for a numerical value since the format doesn't match exactly
        ),
    ],
)
# pylint: disable-next=unused-argument
def test_fmt_ci(test_case, result, evaluator, score_data_type, expected_contains):
    """Test formatting of confidence intervals."""

    result_obj = EvaluationResult(
        variant="test_variant", df_result=pd.DataFrame(result)
    )
    score = EvaluationScore(
        name="test_score",
        evaluator=evaluator,
        field="score",
        data_type=score_data_type,
        desired_direction=DesiredDirection.INCREASE,
    )
    ci = EvaluationScoreCI(result_obj, score)

    output = fmt_ci(ci)
    # Check that the output contains the expected text
    assert expected_contains.lower() in output.lower()


def test_fmt_table_compare(snapshot):
    """Test formatting of table comparison."""

    result_1 = EvaluationResult(
        variant="test_variant_1", df_result=pd.DataFrame(data_result_1)
    )
    result_2 = EvaluationResult(
        variant="test_variant_2", df_result=pd.DataFrame(data_result_2)
    )
    scores = [test_score_1, test_score_2]
    results = {"test_variant_1": result_1, "test_varaint_2": result_2}

    output = fmt_table_compare(scores, results, result_1.variant)

    snapshot.snapshot_dir = Path("tests", "snapshots", "fmt_table_compare")
    snapshot.assert_match(output, "test.md")


def test_fmt_table_ci(snapshot):
    """Test formatting of confidence interval table."""

    result = EvaluationResult(
        variant="test_variant",
        df_result=pd.DataFrame(
            {
                "inputs.id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "outputs.fluency.score": [
                    0.8,
                    0.9,
                    0.85,
                    0.8,
                    0.9,
                    0.85,
                    0.8,
                    0.9,
                    0.85,
                    0.85,
                ],
                "outputs.accuracy.score": [4, 5, 4, 4, 5, 4, 4, 5, 4, 5],
            }
        ),
    )

    scores = [test_score_1, test_score_2]

    output = fmt_table_ci(scores, result)

    snapshot.snapshot_dir = Path("tests", "snapshots", "fmt_table_ci")
    snapshot.assert_match(output, "test.md")
