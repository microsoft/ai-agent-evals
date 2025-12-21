"""Shared test fixtures and utilities."""

from analysis.analysis import (
    DesiredDirection,
    EvaluationScore,
    EvaluationScoreDataType,
)


def create_fluency_score() -> EvaluationScore:
    """Create a standard fluency score for testing.

    This reduces duplicate code across test files.
    """
    return EvaluationScore(
        name="fluency",
        evaluator="fluency",
        field="score",
        data_type=EvaluationScoreDataType.CONTINUOUS,
        desired_direction=DesiredDirection.INCREASE,
    )
