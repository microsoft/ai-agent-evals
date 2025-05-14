"""
Evaluation Analysis Package for AI Agents.

This package provides tools and utilities for analyzing and summarizing evaluation
results from AI agent performance tests. It includes functionality for:

- Processing evaluation results with confidence intervals
- Custom evaluators for operational metrics
- Summarization of evaluation findings
- Visualization and reporting capabilities

The module exposes key classes and functions for working with evaluation data.
"""

# ruff: noqa: F401
# flake8: noqa: F401
from .analysis import EvaluationResult, EvaluationResultView
from .custom_eval import OperationalMetricsEvaluator
from .summary import summarize
