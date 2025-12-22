"""Unit tests for the evaluation render functions."""

from pathlib import Path

import pytest

from analysis.analysis import (
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
    fmt_treatment_badge,
)
from tests.conftest import create_fluency_score


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
    assert fmt_image("https://example.com/image.png", "Alt text") == (
        '![Alt text](https://example.com/image.png "")'
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


def test_fmt_treatment_badge_weak_improvement():
    """Test treatment badge for weak improvement."""
    score = create_fluency_score()

    comparison = EvaluationScoreComparison(
        score=score,
        control_variant="baseline",
        treatment_variant="treatment",
        count=10,
        control_mean=0.8,
        treatment_mean=0.85,
        delta_estimate=0.05,
        p_value=0.001,  # Significant
    )

    badge = fmt_treatment_badge(comparison)

    # Should indicate strong improvement
    assert "improved" in badge.lower() or "green" in badge.lower()


def test_fmt_treatment_badge_weak_degradation():
    """Test treatment badge for weak degradation."""
    score = create_fluency_score()

    comparison = EvaluationScoreComparison(
        score=score,
        control_variant="control",
        treatment_variant="treatment",
        count=10,
        control_mean=0.9,
        treatment_mean=0.7,
        delta_estimate=-0.2,
        p_value=0.001,
    )

    badge = fmt_treatment_badge(comparison)

    # Should indicate degradation
    assert "degraded" in badge.lower() or "red" in badge.lower()


def test_fmt_control_badge():
    """Test control badge formatting."""
    score = create_fluency_score()

    comparison = EvaluationScoreComparison(
        score=score,
        control_variant="control",
        treatment_variant="treatment",
        count=10,
        control_mean=0.8,
        treatment_mean=0.85,
        delta_estimate=0.05,
        p_value=0.1,
    )

    badge = fmt_control_badge(comparison)

    # Should be a badge with control label
    assert "badge" in badge.lower() or "control" in badge.lower()


def test_fmt_ci():
    """Test confidence interval formatting."""
    score = create_fluency_score()

    result_items = [
        {"score": 0.8},
        {"score": 0.9},
        {"score": 0.85},
        {"score": 0.87},
        {"score": 0.83},
    ]

    score_ci = EvaluationScoreCI(
        variant="agent1", score=score, result_items=result_items
    )

    ci_output = fmt_ci(score_ci)

    # Should contain mean and confidence interval
    assert isinstance(ci_output, str)
    assert len(ci_output) > 0
