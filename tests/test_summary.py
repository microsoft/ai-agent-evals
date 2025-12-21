"""Tests for summary module."""

import pytest
from analysis.summary import summarize
from analysis.analysis import (
    EvaluationScore,
    EvaluationScoreCI,
    EvaluationScoreComparison,
)
from analysis.analysis import EvaluationScoreDataType, DesiredDirection


class MockAgent:
    """Mock agent for testing."""

    def __init__(self, name, version):
        self.name = name
        self.version = version


def test_summarize_single_variant():
    """Test summarize with single variant (no comparisons)."""
    # Create baseline results
    score = EvaluationScore(
        name="fluency",
        evaluator="fluency",
        field="score",
        data_type=EvaluationScoreDataType.CONTINUOUS,
        desired_direction=DesiredDirection.INCREASE,
    )

    result_items = [{"score": 0.8}, {"score": 0.9}, {"score": 0.85}]

    score_ci = EvaluationScoreCI(
        variant="agent1:v1", score=score, result_items=result_items
    )

    baseline_results = {
        "evaluation_scores": {"fluency": score_ci},
        "agent": MockAgent("agent1", "v1"),
        "evaluator_names": ["fluency"],
    }

    # Generate summary
    summary = summarize(baseline_results)

    # Verify summary contains key elements
    assert "agent1:v1" in summary
    assert "fluency" in summary
    assert "Observability in generative AI" in summary


def test_summarize_with_comparisons():
    """Test summarize with variant comparisons."""
    # Create baseline results
    score = EvaluationScore(
        name="fluency",
        evaluator="fluency",
        field="score",
        data_type=EvaluationScoreDataType.CONTINUOUS,
        desired_direction=DesiredDirection.INCREASE,
    )

    control_items = [{"score": 0.7}, {"score": 0.75}, {"score": 0.72}]
    treatment_items = [{"score": 0.85}, {"score": 0.88}, {"score": 0.87}]

    score_ci = EvaluationScoreCI(
        variant="control:v1", score=score, result_items=control_items
    )

    comparison = EvaluationScoreComparison(
        score=score,
        control_variant="control:v1",
        treatment_variant="treatment:v1",
        count=3,
        control_mean=0.72,
        treatment_mean=0.87,
        delta_estimate=0.15,
        p_value=0.01,
    )

    baseline_results = {
        "evaluation_scores": {"fluency": score_ci},
        "agent": MockAgent("control", "v1"),
        "evaluator_names": ["fluency"],
    }

    comparisons_by_evaluator = {"fluency": [comparison]}

    # Generate summary
    summary = summarize(baseline_results, comparisons_by_evaluator)

    # Verify summary contains comparison elements
    assert "control:v1" in summary
    assert "treatment:v1" in summary
    assert "fluency" in summary


def test_summarize_with_report_urls():
    """Test summarize with report URLs."""
    score = EvaluationScore(
        name="relevance",
        evaluator="relevance",
        field="score",
        data_type=EvaluationScoreDataType.CONTINUOUS,
        desired_direction=DesiredDirection.INCREASE,
    )

    result_items = [{"score": 0.9}]

    score_ci = EvaluationScoreCI(
        variant="agent1:v1", score=score, result_items=result_items
    )

    baseline_results = {
        "evaluation_scores": {"relevance": score_ci},
        "agent": MockAgent("agent1", "v1"),
        "evaluator_names": ["relevance"],
    }

    report_urls = {"agent1_v1": "https://example.com/report/agent1_v1"}

    # Generate summary
    summary = summarize(
        baseline_results,
        report_urls=report_urls,
        eval_url="https://example.com/eval",
        compare_url="https://example.com/compare",
    )

    # Verify summary contains URLs
    assert "https://example.com" in summary or "example.com" in summary


def test_summarize_multiple_evaluators():
    """Test summarize with multiple evaluators."""
    # Create scores for multiple evaluators
    fluency_score = EvaluationScore(
        name="fluency",
        evaluator="fluency",
        field="score",
        data_type=EvaluationScoreDataType.CONTINUOUS,
        desired_direction=DesiredDirection.INCREASE,
    )

    relevance_score = EvaluationScore(
        name="relevance",
        evaluator="relevance",
        field="score",
        data_type=EvaluationScoreDataType.CONTINUOUS,
        desired_direction=DesiredDirection.INCREASE,
    )

    fluency_ci = EvaluationScoreCI(
        variant="agent:v1", score=fluency_score, result_items=[{"score": 0.8}]
    )

    relevance_ci = EvaluationScoreCI(
        variant="agent:v1", score=relevance_score, result_items=[{"score": 0.9}]
    )

    baseline_results = {
        "evaluation_scores": {"fluency": fluency_ci, "relevance": relevance_ci},
        "agent": MockAgent("agent", "v1"),
        "evaluator_names": ["fluency", "relevance"],
    }

    summary = summarize(baseline_results)

    # Verify both evaluators appear in summary
    assert "fluency" in summary
    assert "relevance" in summary
