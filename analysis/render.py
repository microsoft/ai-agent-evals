# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Render analysis results as a markdown summary"""

from urllib.parse import quote

import pandas as pd

from .analysis import (
    EvaluationResult,
    EvaluationScore,
    EvaluationScoreCI,
    EvaluationScoreComparison,
    EvaluationScoreDataType,
)

SS_THRESHOLD = 0.05
HSS_THRESHOLD = 0.001

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

    color = COLOR_MAP.get(color, color)  # If color isn't in map, keep the original value

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
    """Format a confidence interval"""
    if x.ci_lower is None or x.ci_upper is None:
        md_ci = "n/a"
    elif x.count < 10:
        md_ci = "Too few samples"
    else:
        md_lower = fmt_metric_value(x.ci_lower, x.score.data_type)
        md_upper = fmt_metric_value(x.ci_upper, x.score.data_type)
        md_ci = f"({md_lower}, {md_upper})"

    return md_ci


def fmt_table_compare(
    scores: list[EvaluationScore],
    results: dict[str, EvaluationResult],
    baseline: str,
) -> str:
    """Render a table comparing the evaluation results from multiple agent variants"""
    if not results:
        raise ValueError("No evaluation results provided")

    if not scores:
        raise ValueError("No evaluator scores provided")

    records = []
    for score in scores:
        try:
            row = {"Evaluation score": score.name}

            compare_result = EvaluationScoreComparison(
                results[baseline], results[baseline], score=score
            )
            row[results[baseline].variant] = fmt_control_badge(compare_result)

            for variant, variant_result in results.items():
                if variant == baseline:
                    continue

                compare_result = EvaluationScoreComparison(
                    results[baseline], variant_result, score=score
                )
                row[variant_result.variant] = fmt_treatment_badge(compare_result)

            records.append(row)

        except ValueError as e:
            print(f"Error comparing score {score.name}: {e}")

    df_summary = pd.DataFrame.from_records(records)
    return df_summary.to_markdown(index=False)


def fmt_table_ci(scores: list[EvaluationScore], result: EvaluationResult) -> str:
    """Render a table of confidence intervals for the evaluation result"""
    if not scores:
        raise ValueError("No evaluator scores provided")

    records = []
    for score in scores:
        try:
            result_ci = EvaluationScoreCI(result, score=score)
            records.append(
                {
                    "Evaluation score": score.name,
                    result.variant: fmt_metric_value(
                        result_ci.mean, result_ci.score.data_type
                    ),
                    "95% CI": fmt_ci(result_ci),
                }
            )
        except ValueError as e:
            print(f"Error comparing score {score.name}: {e}")

    df_summary = pd.DataFrame.from_records(records)
    return df_summary.to_markdown(index=False)
