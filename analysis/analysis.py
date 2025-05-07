# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Results from an offline evaluation"""

from dataclasses import dataclass
from enum import Enum
from math import isnan
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import binom, binomtest, t, ttest_rel, wilcoxon
from scipy.stats.contingency import crosstab

SAMPLE_SIZE_THRESHOLD = 10
TEST_ID = "inputs.id"


def mcnemar(contingency_table: np.ndarray) -> float:
    """McNemar's test for paired boolean data.

    We choose the mid-p version of the test, which finds a balance between the
    statistical characteristics of the exact test and the asymptotic test.

    Citation: https://doi.org/10.1186/1471-2288-13-91
    """
    n12 = contingency_table[0, 1]
    n21 = contingency_table[1, 0]
    n = n12 + n21

    pvalue_exact_conditional = 2 * binom.cdf(k=min(n12, n21), n=n, p=0.5)
    pvalue_midp = pvalue_exact_conditional - binom.pmf(k=n12, n=n, p=0.5)

    return float(pvalue_midp)


@dataclass
class EvaluationResult:
    """Result from an AI evaluation"""

    variant: str
    df_result: pd.DataFrame
    ai_foundry_url: str | None = None

    def __post_init__(self):
        if self.variant is None or self.variant == "":
            raise ValueError("variant cannot be empty or missing")
        if self.df_result.empty:
            raise ValueError("df_result cannot be empty")
        if TEST_ID not in self.df_result.columns:
            raise ValueError(f"{TEST_ID} column is required in df_result")
        if self.df_result[TEST_ID].duplicated().any():
            raise ValueError(f"{TEST_ID} column must be unique")


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

    def __init__(self, result: EvaluationResult, score: EvaluationScore):
        # Ensure the evaluation key is present in the dataframe
        col_score = f"outputs.{score.evaluator}.{score.field}"
        if col_score not in result.df_result.columns:
            raise ValueError(f"{col_score} column is required in result")

        self.score = score
        self.variant = result.variant
        self.count = result.df_result.shape[0]
        self._compute_ci(result.df_result[col_score])

    def _compute_ci(self, data: pd.Series, confidence_level: float = 0.95):
        """Compute the confidence interval for the given data"""
        ci_lower = None
        ci_upper = None
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
            stderr = data.std() / (self.count**0.5)
            z_ao2 = t.ppf(1 - (1 - confidence_level) / 2, df=self.count - 1)
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


# pylint: disable-next=too-few-public-methods,too-many-instance-attributes
class EvaluationScoreComparison:
    """Comparison of paired evaluation scores from two variants"""

    def __init__(
        self,
        control: EvaluationResult,
        treatment: EvaluationResult,
        score: EvaluationScore,
    ):
        # Ensure the evaluation key is present in both dataframes
        col_score = f"outputs.{score.evaluator}.{score.field}"
        if (
            col_score not in control.df_result.columns
            or col_score not in treatment.df_result.columns
        ):
            raise ValueError(f"{col_score} column is required in both results")

        df_c = control.df_result[[TEST_ID, col_score]].rename(
            columns={col_score: "score"}
        )
        df_t = treatment.df_result[[TEST_ID, col_score]].rename(
            columns={col_score: "score"}
        )

        df_paired = df_c.merge(
            df_t, how="inner", on=TEST_ID, suffixes=("_c", "_t"), validate="one_to_one"
        )

        # raise exception if there are unmatched rows (will cause contradictions)
        if df_paired.shape[0] < max(df_c.shape[0], df_t.shape[0]):
            raise ValueError("Variants have unmatched evaluation results")

        if df_c["score"].isnull().any() or df_t["score"].isnull().any():
            raise ValueError("Variants have NaN evaluation results")

        self.score = score
        self.control_variant = control.variant
        self.treatment_variant = treatment.variant
        self.count = df_paired.shape[0]

        self.control_mean = float(df_paired["score_c"].mean())
        self.treatment_mean = float(df_paired["score_t"].mean())

        self.delta_estimate = self.treatment_mean - self.control_mean
        self.p_value = float(self._stat_test(df_paired))

    def _stat_test(self, df_paired: pd.DataFrame) -> float:
        """Perform statistical test on the paired scores"""
        if self.score.data_type == EvaluationScoreDataType.ORDINAL:
            diff = (df_paired["score_t"] - df_paired["score_c"]).round()
            if (diff == 0).all():
                p_value = 1.0
            else:
                # Wilcoxon signed-rank test with Pratt zero procedure
                # NOTE: compares medians, which circumvents unequal intervals
                result = wilcoxon(diff, zero_method="pratt")
                p_value = result.pvalue

        elif self.score.data_type == EvaluationScoreDataType.CONTINUOUS:
            diff = df_paired["score_t"] - df_paired["score_c"]
            if (diff == 0).all():
                p_value = 1.0
            elif diff.std() == 0:
                p_value = 0.0
            else:
                # Paired t-test
                # NOTE: assumes normality of the differences
                # (may not be true for bounded scores with small samples)
                result = ttest_rel(df_paired["score_c"], df_paired["score_t"])
                p_value = result.pvalue

        elif self.score.data_type == EvaluationScoreDataType.BOOLEAN:
            contingency_table = crosstab(
                df_paired["score_c"],
                df_paired["score_t"],
                levels=([False, True], [False, True]),
            ).count

            # McNemar's test for paired nominal data
            p_value = mcnemar(contingency_table)

        else:
            raise ValueError(f"Unsupported data type: {self.score.data_type}")

        return p_value

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
        """Treatment effect based on the p-value and desired direction"""
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
