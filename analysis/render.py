# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Render analysis results as a markdown summary"""

from urllib.parse import quote

import pandas as pd

from .analysis import (
    EvaluationScoreCI,
    EvaluationScoreComparison,
    EvaluationScoreDataType,
)
from .constants import HSS_THRESHOLD, SS_THRESHOLD

DARK_GREEN = "157e3b"
PALE_GREEN = "a1d99b"
DARK_RED = "d03536"
PALE_RED = "fcae91"
DARK_BLUE = "1c72af"
PALE_BLUE = "9ecae1"
PALE_YELLOW = "f0e543"
PALE_GREY = "e6e6e3"
WHITE = "ffffff"

COLOR_MAP = {
    "ImprovedStrong": DARK_GREEN,
    "ImprovedWeak": PALE_GREEN,
    "DegradedStrong": DARK_RED,
    "DegradedWeak": PALE_RED,
    "ChangedStrong": DARK_BLUE,
    "ChangedWeak": PALE_BLUE,
    "Inconclusive": PALE_GREY,
    "Warning": PALE_YELLOW,
    "Pass": DARK_GREEN,
    "Fail": DARK_RED,
    "Information": PALE_GREY,
}


def fmt_metric_value(
    x: float, data_type: EvaluationScoreDataType, sign: bool = False
) -> str:
    """Format a metric value"""
    if data_type == EvaluationScoreDataType.ORDINAL:
        spec = ".2f"
    elif data_type == EvaluationScoreDataType.CONTINUOUS:
        spec = ".3g"
    elif data_type == EvaluationScoreDataType.BOOLEAN:
        spec = ".1%"
    else:
        raise ValueError(f"Unsupported data type: {data_type}")

    if sign:
        spec = "+" + spec
    return format(x, spec)


def fmt_pvalue(x: float) -> str:
    """Format a p-value"""
    if x <= 0:
        return "≈0"

    spec = ".0e" if x < 0.001 else ".3f"
    return format(x, spec).replace("e-0", "e-")


def fmt_hyperlink(text: str, url: str, tooltip: str = "") -> str:
    """Markdown to render a hyperlink"""
    tooltip = tooltip.replace("\n", "&#013;").replace('"', "&quot;")
    return f'[{text}]({url} "{tooltip}")'


def fmt_image(url: str, alt_text: str, tooltip: str = "") -> str:
    """Markdown to render an image"""
    return "!" + fmt_hyperlink(alt_text, url, tooltip)


def fmt_badge(label: str, message: str, color: str, tooltip: str = "") -> str:
    """Markdown to render a badge

    Parameters
    ----------
    label : str
        Left-hand side of the badge.
    message : str
        Right-hand side of the badge.
    color : str
        Badge color. Accepts hex, rgb, hsl, hsla, css named color, or a preset
    tooltip : str, optional
        Tooltip. Default: standard message for color presets, otherwise none.
    """
    if not tooltip:
        if color.endswith("Strong"):
            tooltip = "Highly statistically significant."
        elif color.endswith("Weak"):
            tooltip = "Marginally statistically significant."
        elif color == "Inconclusive":
            tooltip = "Not statistically significant."

    color = COLOR_MAP.get(
        color, color
    )  # If color isn't in map, keep the original value

    def escape(s: str) -> str:
        return quote(s, safe="").replace("-", "--").replace("_", "__")

    badge_content = "-".join(map(escape, [label, message, color]))
    url = f"https://img.shields.io/badge/{badge_content}"
    alt_text = f"{label}: {message}"

    return fmt_image(url, alt_text, tooltip)


def fmt_treatment_badge(x: EvaluationScoreComparison) -> str:
    """Format a treatment effect as a badge"""
    effect = x.treatment_effect

    if effect in ["Improved", "Degraded", "Changed"]:
        if x.p_value <= HSS_THRESHOLD:
            color = f"{effect}Strong"
            tooltip_stat = "Highly statistically significant"
        elif x.p_value <= SS_THRESHOLD:
            color = f"{effect}Weak"
            tooltip_stat = "Marginally statistically significant"
        else:
            color = "Warning"
            tooltip_stat = "Unexpected classification"
        tooltip_stat += f" (p-value: {fmt_pvalue(x.p_value)})."
    elif effect == "Inconclusive":
        if x.p_value > SS_THRESHOLD:
            color = effect
            tooltip_stat = "Not statistically significant"
        else:
            color = "Warning"
            tooltip_stat = "Unexpected classification"
        tooltip_stat += f" (p-value: {fmt_pvalue(x.p_value)})."
    elif effect == "Too few samples":
        color = "Warning"
        tooltip_stat = "Insufficient observations to determine statistical significance"
    elif effect == "Zero samples":
        color = "Warning"
        tooltip_stat = "Zero observations might indicate a problem with data collection"
    else:
        color = PALE_GREY
        tooltip_stat = ""

    value = fmt_metric_value(x.treatment_mean, x.score.data_type)
    delta = fmt_metric_value(x.delta_estimate, x.score.data_type, sign=True)
    return fmt_badge(effect, f"{value} ({delta})", color, tooltip_stat)


def fmt_control_badge(x: EvaluationScoreComparison) -> str:
    """Format a control value"""
    value = fmt_metric_value(x.control_mean, x.score.data_type)
    return fmt_badge("Baseline", value, WHITE)


def fmt_ci(x: EvaluationScoreCI) -> str:
    """Format a confidence interval as a badge"""
    if x.ci_lower is None or x.ci_upper is None:
        color = "Information"
        tooltip_stat = "Confidence interval not available"
        return fmt_badge("", "N/A", color, tooltip_stat)

    if x.count < 10:
        color = "Information"
        tooltip_stat = "Too few samples to determine confidence interval"
        return fmt_badge("", "Too few samples", color, tooltip_stat)

    md_lower = fmt_metric_value(x.ci_lower, x.score.data_type)
    md_upper = fmt_metric_value(x.ci_upper, x.score.data_type)
    md_ci = f"({md_lower}, {md_upper})"
    return md_ci


def fmt_table_compare(
    comparisons_by_evaluator: dict[str, list[EvaluationScoreComparison]],
    baseline_name: str,
) -> str:
    """Render a table comparing evaluation results from multiple agent variants.

    Args:
        comparisons_by_evaluator: Dictionary mapping evaluator names to lists of
            EvaluationScoreComparison objects (one per treatment agent)
        baseline_name: Name of the baseline agent

    Returns:
        Markdown formatted comparison table
    """
    if not comparisons_by_evaluator:
        raise ValueError("No comparison results provided")

    records = []
    for score_key, comparisons in comparisons_by_evaluator.items():
        try:
            # The key is already formatted properly from processing.py
            # It's either "evaluator" or "evaluator:metric" for multiple metrics
            first_comp = comparisons[0] if comparisons else None

            row = {"Evaluation metric": score_key}

            # Create a control badge using first comparison (same baseline for all)
            if first_comp:
                # Create a self-comparison for the baseline
                baseline_comp = EvaluationScoreComparison(
                    score=first_comp.score,
                    control_variant=baseline_name,
                    treatment_variant=baseline_name,
                    count=first_comp.count,
                    control_mean=first_comp.control_mean,
                    treatment_mean=first_comp.control_mean,
                    delta_estimate=0.0,
                    p_value=1.0,
                    treatment_effect_result="Inconclusive",
                )
                row[baseline_name] = fmt_control_badge(baseline_comp)

                # Add treatment badges for each comparison
                for comp in comparisons:
                    row[comp.treatment_variant] = fmt_treatment_badge(comp)

            records.append(row)

        except (ValueError, KeyError) as e:
            print(f"Error comparing score {score_key}: {e}")

    df_summary = pd.DataFrame.from_records(records)
    return df_summary.to_markdown(index=False)


def fmt_table_ci(evaluation_scores: dict[str, any], agent_name: str) -> str:
    """Render a table of confidence intervals for the evaluation scores

    Args:
        evaluation_scores: Dictionary mapping evaluator names to EvaluationScoreCI objects
        agent_name: Name of the agent being evaluated

    Returns:
        Markdown formatted table string
    """
    if not evaluation_scores:
        raise ValueError("No evaluation scores provided")

    records = []
    for score_key, score_ci in evaluation_scores.items():
        try:
            # The key is already formatted properly from processing.py
            # It's either "evaluator" or "evaluator:metric" for multiple metrics
            eval_score_label = score_key

            records.append(
                {
                    "Evaluation metric": eval_score_label,
                    "Pass Rate": (
                        f"{score_ci.item_summary['pass_rate']:.1%}"
                        if score_ci.item_summary
                        else "N/A"
                    ),
                    "Score": (
                        fmt_metric_value(score_ci.mean, score_ci.score.data_type)
                        if score_ci.mean is not None
                        else "N/A"
                    ),
                    "95% Confidence Interval": fmt_ci(score_ci),
                }
            )
        except (ValueError, KeyError) as e:
            print(f"Error formatting score {score_key}: {e}")

    df_summary = pd.DataFrame.from_records(records)

    if not df_summary.empty:
        # First column (Evaluation metric) left-aligned, all other columns right-aligned
        alignments = ["left"] + ["right"] * (len(df_summary.columns) - 1)
        return df_summary.to_markdown(index=False, colalign=alignments)

    return ""
