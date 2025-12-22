"""Tests for the analysis module functionality"""

import numpy as np
import pytest
from scipy import stats

from analysis.analysis import (
    DesiredDirection,
    EvaluationScore,
    EvaluationScoreCI,
    EvaluationScoreComparison,
    EvaluationScoreDataType,
)
from tests.conftest import create_fluency_score


def compute_comparison_stats(
    control_values: list, treatment_values: list, data_type: EvaluationScoreDataType
) -> tuple:
    """Helper function to compute comparison statistics for testing."""
    control_arr = np.array(control_values)
    treatment_arr = np.array(treatment_values)

    count = len(control_values)
    control_mean = float(np.mean(control_arr))
    treatment_mean = float(np.mean(treatment_arr))
    delta_estimate = treatment_mean - control_mean

    # Compute p-value based on data type
    if data_type == EvaluationScoreDataType.BOOLEAN:
        _, p_value = stats.ttest_rel(
            treatment_arr.astype(float), control_arr.astype(float)
        )
    elif data_type == EvaluationScoreDataType.ORDINAL:
        _, p_value = stats.wilcoxon(treatment_arr, control_arr)
    else:
        _, p_value = stats.ttest_rel(treatment_arr, control_arr)

    return count, control_mean, treatment_mean, delta_estimate, float(p_value)


def test_create_score():
    """Test creating an evaluation score with all required fields."""
    score = create_fluency_score()

    assert score.name == "fluency"
    assert score.evaluator == "fluency"
    assert score.field == "score"
    assert score.data_type == EvaluationScoreDataType.CONTINUOUS
    assert score.desired_direction == DesiredDirection.INCREASE


def test_evaluation_score_ci():
    """Test creating a confidence interval from result items."""
    result_items = [
        {"score": 0.8},
        {"score": 0.9},
        {"score": 0.85},
    ]
    score = create_fluency_score()

    ci = EvaluationScoreCI(
        variant="test_variant", score=score, result_items=result_items
    )

    assert ci.variant == "test_variant"
    assert ci.count == 3
    assert ci.mean == pytest.approx(0.85, rel=1e-2)


def test_evaluation_score_ci_boolean():
    """Test creating a confidence interval with boolean data type."""
    # Test with passed/failed boolean results
    result_items = [
        {"passed": True, "score": 1},
        {"passed": False, "score": 0},
        {"passed": True, "score": 1},
    ]
    score = EvaluationScore(
        name="pass_fail",
        evaluator="pass_fail",
        field="passed",
        data_type=EvaluationScoreDataType.BOOLEAN,
        desired_direction=DesiredDirection.INCREASE,
    )

    ci = EvaluationScoreCI(
        variant="test_variant", score=score, result_items=result_items
    )

    assert ci.variant == "test_variant"
    assert ci.count == 3
    assert ci.mean == pytest.approx(2 / 3, rel=1e-2)  # 2 out of 3 passed
    assert ci.ci_lower is not None
    assert ci.ci_upper is not None


def test_evaluation_score_comparison_continuous():
    """Test comparing two variants with continuous scores."""
    control_values = [0.8, 0.9, 0.85]
    treatment_values = [0.6, 0.5, 0.75]

    score = create_fluency_score()

    count, control_mean, treatment_mean, delta_estimate, p_value = (
        compute_comparison_stats(
            control_values, treatment_values, EvaluationScoreDataType.CONTINUOUS
        )
    )

    comparison = EvaluationScoreComparison(
        score=score,
        control_variant="baseline",
        treatment_variant="treatment",
        count=count,
        control_mean=control_mean,
        treatment_mean=treatment_mean,
        delta_estimate=delta_estimate,
        p_value=p_value,
    )

    assert comparison.score.name == "fluency"
    assert comparison.control_variant == "baseline"
    assert comparison.treatment_variant == "treatment"
    assert comparison.count == 3
    assert comparison.control_mean == pytest.approx(0.85, rel=1e-2)
    assert comparison.treatment_mean == pytest.approx(0.62, rel=1e-2)
    assert comparison.delta_estimate == pytest.approx(-0.233, rel=1e-2)
    assert comparison.treatment_effect == "Too few samples"


def test_evaluation_score_comparison_ordinal():
    """Test comparing two variants with ordinal scores."""
    control_values = [1, 2, 1]
    treatment_values = [2, 4, 5]

    score = EvaluationScore(
        name="rating",
        evaluator="rating",
        field="score",
        data_type=EvaluationScoreDataType.ORDINAL,
        desired_direction=DesiredDirection.INCREASE,
    )

    count, control_mean, treatment_mean, delta_estimate, p_value = (
        compute_comparison_stats(
            control_values, treatment_values, EvaluationScoreDataType.ORDINAL
        )
    )

    comparison = EvaluationScoreComparison(
        score=score,
        control_variant="baseline",
        treatment_variant="treatment",
        count=count,
        control_mean=control_mean,
        treatment_mean=treatment_mean,
        delta_estimate=delta_estimate,
        p_value=p_value,
    )

    assert comparison.count == 3
    assert comparison.control_mean == pytest.approx(1.333, rel=1e-2)
    assert comparison.treatment_mean == pytest.approx(3.667, rel=1e-2)
    assert comparison.delta_estimate == pytest.approx(2.333, rel=1e-2)
    assert comparison.treatment_effect == "Too few samples"


def test_evaluation_score_comparison_boolean():
    """Test comparing two variants with boolean scores."""
    control_values = [True, False, False]
    treatment_values = [True, True, True]

    score = EvaluationScore(
        name="pass",
        evaluator="pass",
        field="score",
        data_type=EvaluationScoreDataType.BOOLEAN,
        desired_direction=DesiredDirection.INCREASE,
    )

    count, control_mean, treatment_mean, delta_estimate, p_value = (
        compute_comparison_stats(
            control_values, treatment_values, EvaluationScoreDataType.BOOLEAN
        )
    )

    comparison = EvaluationScoreComparison(
        score=score,
        control_variant="baseline",
        treatment_variant="treatment",
        count=count,
        control_mean=control_mean,
        treatment_mean=treatment_mean,
        delta_estimate=delta_estimate,
        p_value=p_value,
    )

    assert comparison.count == 3
    assert comparison.control_mean == pytest.approx(0.333, rel=1e-2)
    assert comparison.treatment_mean == pytest.approx(1.0, rel=1e-2)
    assert comparison.delta_estimate == pytest.approx(0.667, rel=1e-2)
    assert comparison.treatment_effect == "Too few samples"


def test_evaluation_score_comparison_boolean_statistically_significant():
    """Test comparison with enough samples to show statistical significance."""
    control_values = [True, False, False, True, False, False, True, False, False, False]
    treatment_values = [False, True, True, True, True, True, True, True, True, True]

    score = EvaluationScore(
        name="pass",
        evaluator="pass",
        field="score",
        data_type=EvaluationScoreDataType.BOOLEAN,
        desired_direction=DesiredDirection.INCREASE,
    )

    count, control_mean, treatment_mean, delta_estimate, p_value = (
        compute_comparison_stats(
            control_values, treatment_values, EvaluationScoreDataType.BOOLEAN
        )
    )

    comparison = EvaluationScoreComparison(
        score=score,
        control_variant="baseline",
        treatment_variant="treatment",
        count=count,
        control_mean=control_mean,
        treatment_mean=treatment_mean,
        delta_estimate=delta_estimate,
        p_value=p_value,
    )

    assert comparison.count == 10
    assert comparison.control_mean == pytest.approx(0.3, rel=1e-2)
    assert comparison.treatment_mean == pytest.approx(0.9, rel=1e-2)
    assert comparison.delta_estimate == pytest.approx(0.600, rel=1e-2)
    assert comparison.p_value < 0.05  # Statistically significant
    assert comparison.treatment_effect == "Improved"
