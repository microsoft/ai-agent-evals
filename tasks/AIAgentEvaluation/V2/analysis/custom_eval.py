# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Evaluator to pass through operational metrics."""


# pylint: disable-next=too-few-public-methods
class OperationalMetricsEvaluator:
    """Propagate operational metrics to the final evaluation results.

    This evaluator passes any values found in the "metrics" field of the input
    data to the output data without any changes.

    E.g., If the input dataset has field: {"metrics": {"token-count": 100}}
    and the evaluator is named "my_metrics", then the evaluation results will
    have the field: {"output.my_metrics.token-count": 100}
    """

    def __init__(self):
        pass

    def __call__(self, *, metrics: dict, **kwargs):
        return metrics
