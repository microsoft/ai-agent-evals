"""Tests for the analysis module functionality"""

import pandas as pd
import pytest
from action import convert_pass_fail_to_boolean

from analysis.analysis import (
    DesiredDirection,
    EvaluationResult,
    EvaluationScore,
    EvaluationScoreCI,
    EvaluationScoreComparison,
    EvaluationScoreDataType,
)

data_result_1 = {
    "inputs.id": [1, 2, 3],
    "outputs.fluency.score": [0.8, 0.9, 0.85],
    "outputs.accuracy.score": [4, 5, 4],
}

data_result_2 = {
    "inputs.id": [1, 2, 3],
    "outputs.fluency.score": [0.6, 0.5, 0.75],
    "outputs.accuracy.score": [3, 4, 5],
}

test_score_1 = EvaluationScore(
    name="fluency",
    evaluator="fluency",
    field="score",
    data_type=EvaluationScoreDataType.CONTINUOUS,
    desired_direction=DesiredDirection.INCREASE,
)

test_score_2 = EvaluationScore(
    name="accuracy",
    evaluator="accuracy",
    field="score",
    data_type=EvaluationScoreDataType.ORDINAL,
    desired_direction=DesiredDirection.DECREASE,
)


def test_create_score():
    """Test creating an evaluation score."""
    assert test_score_1.name == "fluency"
    assert test_score_1.evaluator == "fluency"
    assert test_score_1.field == "score"
    assert test_score_1.data_type == EvaluationScoreDataType.CONTINUOUS
    assert test_score_1.desired_direction == DesiredDirection.INCREASE


def test_create_evaluation_result():
    """Test creating an evaluation result with multiple scores"""
    result = EvaluationResult(
        variant="test_variant",
        df_result=pd.DataFrame(data_result_1),
        ai_foundry_url="test_url",
    )
    assert result.variant == "test_variant"
    assert result.df_result.shape == (3, 3)  # 3 rows, 3 columns
    assert result.ai_foundry_url == "test_url"


def test_evaluation_confidence_interval():
    """Test creating a confidence interval for an evaluation result"""
    result = EvaluationResult(
        variant="test_variant",
        df_result=pd.DataFrame(data_result_1),
        ai_foundry_url="test_url",
    )
    score = EvaluationScore(
        name="fluency",
        evaluator="fluency",
        field="score",
        data_type=EvaluationScoreDataType.CONTINUOUS,
        desired_direction=DesiredDirection.INCREASE,
    )
    ci = EvaluationScoreCI(result, score)
    assert ci.ci_lower == pytest.approx(0.73, rel=1e-2)
    assert ci.ci_upper == pytest.approx(0.97, rel=1e-2)
    assert ci.mean == pytest.approx(0.85, rel=1e-2)


def test_evaluation_score_comparison():
    """Test comparing two evaluation results"""
    control_result = EvaluationResult(
        variant="test_variant_1",
        df_result=pd.DataFrame(data_result_1),
        ai_foundry_url="test_url_1",
    )
    treatment_result = EvaluationResult(
        variant="test_variant_2",
        df_result=pd.DataFrame(data_result_2),
        ai_foundry_url="test_url_2",
    )
    score = EvaluationScore(
        name="fluency",
        evaluator="fluency",
        field="score",
        data_type=EvaluationScoreDataType.CONTINUOUS,
        desired_direction=DesiredDirection.INCREASE,
    )

    comparison = EvaluationScoreComparison(control_result, treatment_result, score)

    assert comparison.score.name == "fluency"
    assert comparison.control_variant == "test_variant_1"
    assert comparison.treatment_variant == "test_variant_2"
    assert comparison.count == 3
    assert comparison.control_mean == pytest.approx(0.85, rel=1e-2)
    assert comparison.treatment_mean == pytest.approx(0.62, rel=1e-2)
    assert comparison.delta_estimate == pytest.approx(-0.233, rel=1e-2)
    assert comparison.p_value == pytest.approx(0.118, rel=1e-2)
    assert comparison.treatment_effect == "Too few samples"


def test_boolean_conversion():
    """Test that string pass/fail values are converted to boolean values correctly."""
    # Create test data with string pass/fail values
    test_data = {
        "inputs.id": ["test1", "test2", "test3"],
        "outputs.fluency.result": ["pass", "fail", "PASS"],  # Case variations
        "outputs.safety.result": ["FAIL", "pass", "fail"],
    }

    # Create a mock eval_result_data dictionary as it appears in action.py
    eval_result_data = {"rows": []}
    for i, test_id in enumerate(test_data["inputs.id"]):
        row = {
            "inputs.id": test_id,
            "outputs.fluency.result": test_data["outputs.fluency.result"][i],
            "outputs.safety.result": test_data["outputs.safety.result"][i],
        }
        eval_result_data["rows"].append(row)

    # Test the conversion logic from action.py
    convert_pass_fail_to_boolean(eval_result_data)

    # Verify conversion worked correctly
    assert eval_result_data["rows"][0]["outputs.fluency.result"] is True
    assert eval_result_data["rows"][0]["outputs.safety.result"] is False
    assert eval_result_data["rows"][1]["outputs.fluency.result"] is False
    assert eval_result_data["rows"][1]["outputs.safety.result"] is True
    assert eval_result_data["rows"][2]["outputs.fluency.result"] is True
    assert eval_result_data["rows"][2]["outputs.safety.result"] is False
