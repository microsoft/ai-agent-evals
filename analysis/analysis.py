# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Results from an offline evaluation"""

from dataclasses import dataclass
from enum import Enum
from math import isnan
from typing import Literal

import pandas as pd
from scipy.stats import binomtest, t

SAMPLE_SIZE_THRESHOLD = 10


class EvaluationResultView(Enum):
    """Different views for displaying evaluation results

    Controls how evaluation results are presented to users,
    with options for different levels of detail.
    """

    DEFAULT = "default"  # Default view, showing only passing/defect rate
    ALL = "all-scores"  # All scores view, showing all evaluation scores
    RAW_SCORES = "raw-scores-only"  # Raw scores view, showing only raw metrics


class EvaluationScoreDataType(Enum):
    """Data type of the evaluation score"""

    ORDINAL = "Ordinal"
    CONTINUOUS = "Continuous"
    BOOLEAN = "Boolean"


class DesiredDirection(Enum):
    """Desired direction of the evaluation score"""

    INCREASE = "Increase"
    DECREASE = "Decrease"
    NEUTRAL = "Neutral"


@dataclass
class EvaluationScore:
    """Metadata about an evaluation score"""

    name: str
    evaluator: str
    field: str
    data_type: EvaluationScoreDataType
    desired_direction: DesiredDirection

    def __post_init__(self):
        if self.name is None or self.name == "":
            raise ValueError("name cannot be empty or missing")
        if self.evaluator is None or self.evaluator == "":
            raise ValueError("evaluator cannot be empty or missing")
        if self.field is None or self.field == "":
            raise ValueError("field cannot be empty or missing")

        if isinstance(self.data_type, str):
            self.data_type = EvaluationScoreDataType(self.data_type)
        if isinstance(self.desired_direction, str):
            self.desired_direction = DesiredDirection(self.desired_direction)


# pylint: disable-next=too-few-public-methods
class EvaluationScoreCI:
    """Confidence interval for an evaluation score"""

    def __init__(self, variant: str, score: EvaluationScore, result_items: list):
        """
        Initialize EvaluationScoreCI from result items.
        
        Args:
            variant: Name/identifier of the variant
            score: Metadata about the evaluation score
            result_items: List of evaluation result items from the API
        """
        if not result_items:
            raise ValueError("result_items cannot be empty")

        self.score = score
        self.variant = variant
        self.result_items = result_items
        self.count = len(result_items)
        
        # Extract scores from result items
        scores = self._extract_scores_from_items()
        self._compute_ci(scores)
        self._summarize_items()

    def _extract_scores_from_items(self) -> pd.Series:
        """Extract scores from result items based on the score field"""
        scores = []
        for item in self.result_items:
            # Try to get the score based on the field name
            if self.score.field == 'score' and 'score' in item:
                scores.append(item['score'])
            elif self.score.field == 'passed' and 'passed' in item:
                scores.append(item['passed'])
            elif self.score.field in item:
                scores.append(item[self.score.field])
            else:
                # Default to score field if field not found
                scores.append(item.get('score'))
        
        return pd.Series(scores)

    def _compute_ci(self, data: pd.Series, confidence_level: float = 0.95):
        """Compute the confidence interval for the given data"""
        ci_lower = None
        ci_upper = None
        
        # Remove None values
        data = data.dropna()
        
        if len(data) == 0:
            self.mean = None
            self.ci_lower = None
            self.ci_upper = None
            return
        
        if self.score.data_type == EvaluationScoreDataType.BOOLEAN:
            result = binomtest(data.sum(), data.count())
            mean = result.proportion_estimate
            ci = result.proportion_ci(
                confidence_level=confidence_level, method="wilsoncc"
            )
            ci_lower = ci.low
            ci_upper = ci.high

        elif self.score.data_type == EvaluationScoreDataType.CONTINUOUS:
            # NOTE: parametric CI does not respect score bounds (use bootstrapping if needed)
            mean = data.mean()
            if len(data) > 1:
                stderr = data.std() / (len(data)**0.5)
                z_ao2 = t.ppf(1 - (1 - confidence_level) / 2, df=len(data) - 1)
                ci_lower = mean - z_ao2 * stderr
                ci_upper = mean + z_ao2 * stderr

        elif self.score.data_type == EvaluationScoreDataType.ORDINAL:
            # NOTE: ordinal data has non-linear intervals, so we omit CI
            mean = data.mean()
            ci_lower = None
            ci_upper = None

        self.mean = mean
        self.ci_lower = ci_lower
        self.ci_upper = ci_upper

    def _summarize_items(self):
        """Summarize evaluation result items"""
        if not self.result_items:
            self.item_summary = None
            return

        # Extract key metrics from result items
        passed_count = sum(1 for item in self.result_items if item.get('passed', False))
        failed_count = len(self.result_items) - passed_count
        
        scores = [item.get('score') for item in self.result_items if item.get('score') is not None]
        avg_score = sum(scores) / len(scores) if scores else None
        
        # Collect reasons for failures
        fail_reasons = [item.get('reason', '') for item in self.result_items if not item.get('passed', False)]
        
        # Collect usage statistics if available
        total_prompt_tokens = 0
        total_completion_tokens = 0
        for item in self.result_items:
            sample = item.get('sample', {})
            usage = sample.get('usage', {})
            total_prompt_tokens += usage.get('prompt_tokens', 0)
            total_completion_tokens += usage.get('completion_tokens', 0)
        
        self.item_summary = {
            'total_items': len(self.result_items),
            'passed_count': passed_count,
            'failed_count': failed_count,
            'pass_rate': passed_count / len(self.result_items) if self.result_items else 0,
            'average_score': avg_score,
            'fail_reasons': fail_reasons,
            'total_prompt_tokens': total_prompt_tokens,
            'total_completion_tokens': total_completion_tokens,
            'total_tokens': total_prompt_tokens + total_completion_tokens
        }


# pylint: disable-next=too-few-public-methods,too-many-instance-attributes
class EvaluationScoreComparison:
    """Comparison of paired evaluation scores from two variants"""

    def __init__(
        self,
        score: EvaluationScore,
        control_variant: str,
        treatment_variant: str,
        count: int,
        control_mean: float,
        treatment_mean: float,
        delta_estimate: float,
        p_value: float,
        treatment_effect_result: str | None = None,
    ):
        """Initialize comparison with pre-computed statistics.
        
        Args:
            score: Metadata about the evaluation score
            control_variant: Name of the baseline/control variant
            treatment_variant: Name of the treatment variant
            count: Number of samples
            control_mean: Mean score for control variant
            treatment_mean: Mean score for treatment variant
            delta_estimate: Difference between treatment and control means
            p_value: Statistical test p-value
            treatment_effect_result: Optional pre-computed treatment effect
        """
        self.score = score
        self.control_variant = control_variant
        self.treatment_variant = treatment_variant
        self.count = count
        self.control_mean = control_mean
        self.treatment_mean = treatment_mean
        self.delta_estimate = delta_estimate
        self.p_value = p_value
        self._treatment_effect_result = treatment_effect_result

    @classmethod
    def from_insight_comparison(
        cls,
        comparison_data: dict,
        control_variant: str,
        treatment_variant: str,
        score: EvaluationScore,
    ):
        """Create comparison from Azure AI comparison insight result.
        
        Args:
            comparison_data: Single comparison item from insight result
            control_variant: Name of the baseline/control variant
            treatment_variant: Name of the treatment variant
            score: Metadata about the evaluation score
            
        Returns:
            EvaluationScoreComparison instance
            
        Example comparison_data structure:
            {
                'testingCriteria': 'fluency',
                'metric': 'fluency',
                'evaluator': 'builtin.fluency',
                'baselineRunSummary': {
                    'runId': 'evalrun_...',
                    'sampleCount': '3',
                    'average': 4.333,
                    'standardDeviation': 1.154
                },
                'compareItems': [{
                    'treatmentRunSummary': {
                        'runId': 'evalrun_...',
                        'sampleCount': '3',
                        'average': 3.666,
                        'standardDeviation': 1.527
                    },
                    'deltaEstimate': -0.666,
                    'pValue': 0.183,
                    'treatmentEffect': 'TooFewSamples'
                }]
            }
        """
        baseline_summary = comparison_data['baselineRunSummary']
        # Get first treatment comparison item
        compare_item = comparison_data['compareItems'][0]
        treatment_summary = compare_item['treatmentRunSummary']
        
        # Map treatment effect from API format to our format
        treatment_effect_map = {
            'TooFewSamples': 'Too few samples',
            'ZeroSamples': 'Zero samples',
            'Inconclusive': 'Inconclusive',
            'Changed': 'Changed',
            'Improved': 'Improved',
            'Degraded': 'Degraded',
        }
        treatment_effect = treatment_effect_map.get(
            compare_item.get('treatmentEffect'),
            None
        )
        
        return cls(
            score=score,
            control_variant=control_variant,
            treatment_variant=treatment_variant,
            count=int(baseline_summary['sampleCount']),
            control_mean=float(baseline_summary['average']),
            treatment_mean=float(treatment_summary['average']),
            delta_estimate=float(compare_item['deltaEstimate']),
            p_value=float(compare_item['pValue']),
            treatment_effect_result=treatment_effect,
        )

    @property
    # pylint: disable-next=too-many-return-statements
    def treatment_effect(
        self,
    ) -> Literal[
        "Zero samples",
        "Too few samples",
        "Inconclusive",
        "Changed",
        "Improved",
        "Degraded",
    ]:
        """Treatment effect based on the p-value and desired direction.
        
        Returns pre-computed result if available, otherwise computes from statistics.
        """
        # Return pre-computed result if available
        if self._treatment_effect_result:
            return self._treatment_effect_result
        
        # Otherwise compute from statistics
        if self.count == 0:
            return "Zero samples"
        if self.count < SAMPLE_SIZE_THRESHOLD:
            return "Too few samples"
        if isnan(self.p_value):
            print("Encountered NaN p-value")
            return "Inconclusive"
        if self.p_value > 0.05:
            return "Inconclusive"
        if self.score.desired_direction == DesiredDirection.NEUTRAL:
            return "Changed"
        if (
            self.score.desired_direction == DesiredDirection.INCREASE
            and self.treatment_mean > self.control_mean
        ):
            return "Improved"
        if (
            self.score.desired_direction == DesiredDirection.DECREASE
            and self.treatment_mean < self.control_mean
        ):
            return "Improved"
        return "Degraded"
