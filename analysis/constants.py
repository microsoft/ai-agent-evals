# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Constants used across the analysis module."""

from azure.ai.projects.models import EvaluatorMetricDirection, EvaluatorMetricType

# Statistical thresholds
SAMPLE_SIZE_THRESHOLD = (
    10  # Minimum samples required for statistical significance testing
)
SS_THRESHOLD = 0.05  # Statistical significance threshold (p-value)
HSS_THRESHOLD = 0.001  # Highly statistical significance threshold (p-value)

# Default evaluator metadata structure
DEFAULT_EVALUATOR_METADATA = {
    "metrics": {
        "score": {
            "data_type": EvaluatorMetricType.CONTINUOUS,
            "desired_direction": EvaluatorMetricDirection.INCREASE,
            "field": "score",
        }
    },
    "categories": [],
    "init_parameters": None,
    "data_schema": None,
}
