# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Summary module for formatting and generating evaluation result summaries.

This module provides functionality to generate formatted markdown summaries
of evaluation results for AI agents. It includes functions to create
tables comparing multiple agent variants or displaying confidence intervals
for a single agent's performance metrics.
"""
from pathlib import Path

import yaml
from azure.ai.projects.models import Agent

from .analysis import (
    EvaluationResult,
    EvaluationResultView,
    EvaluationScore,
    EvaluationScoreDataType,
)
from .render import fmt_hyperlink, fmt_table_ci, fmt_table_compare


def should_include_score(
    score: dict, evaluator: dict, result_view: EvaluationResultView
) -> bool:
    """
    Determines if a score should be included in the result view.

    Args:
        score: Score metadata from evaluator-scores.yaml
        evaluator: Evaluator metadata from evaluator-scores.yaml
        result_view: The current view mode for evaluation results

    Returns:
        True if the score should be included, False otherwise
    """
    # Always include operational metrics
    if evaluator["class"] == "OperationalMetricsEvaluator":
        return True

    if result_view == EvaluationResultView.ALL:
        return True

    if score["type"] == EvaluationScoreDataType.BOOLEAN.value:
        return result_view == EvaluationResultView.DEFAULT

    return result_view == EvaluationResultView.RAW_SCORES


# pylint: disable-next=too-many-locals, too-many-arguments, too-many-positional-arguments
def summarize(
    eval_results: dict[str, EvaluationResult],
    agents: dict[str, Agent],
    baseline: str,
    evaluators: list[str],
    agent_base_url: str,
    result_view: EvaluationResultView,
) -> str:
    """Generate a markdown summary of evaluation results.

    Args:
        eval_results: Dictionary mapping agent IDs to their evaluation results
        agents: Dictionary mapping agent IDs to Agent objects
        baseline: ID of the baseline agent for comparisons
        evaluators: List of evaluator class names to include in the summary
        agent_base_url: Base URL for agent links
        result_view: The view mode to use for displaying evaluation results

    Returns:
        Formatted markdown string with evaluation summary
    """
    md = []
    view_label = (
        "" if result_view == EvaluationResultView.DEFAULT else f"({result_view.value})"
    )
    md.append(f"## Azure AI Evaluation {view_label}\n")

    def format_agent_row(agent: Agent, agent_url: str) -> str:
        result_url = eval_results[agent.id].ai_foundry_url
        result_link = fmt_hyperlink("Click here", result_url) if result_url else ""
        return (
            f"| {agent.name} | "
            f"{fmt_hyperlink(agent.id, agent_url)} | "
            f"{result_link} |"
        )

    md.append("### Agent variants\n")
    md.append("| Agent name | Agent ID | Evaluation results |")
    md.append("|:-----------|:---------|:-------------------|")
    md.append(format_agent_row(agents[baseline], agent_base_url + agents[baseline].id))

    for agent in agents.values():
        if agent.id != baseline:
            md.append(format_agent_row(agent, agent_base_url + agent.id))

    # load hardcoded evaluator score metadata
    metadata_path = Path(__file__).parent / "evaluator-scores.yaml"
    with open(metadata_path, encoding="utf-8") as f:
        score_metadata = yaml.safe_load(f)

    if len(eval_results) >= 2:
        md.append("\n### Compare evaluation scores between variants\n")
    elif len(eval_results) == 1:
        md.append("\n### Evaluation results\n")

    for section in score_metadata["sections"]:
        section_evals = [x["class"] for x in section["evaluators"]]
        if any(x in evaluators for x in section_evals):
            append_eval_section(
                eval_results, baseline, evaluators, result_view, md, section
            )

    md.append("### References\n")
    md.append(
        "- See [evaluator-scores.yaml](https://github.com/microsoft/ai-agent-evals/blob/main/"
        "analysis/evaluator-scores.yaml) for the full list of evaluators supported "
        "and the definitions of the scores"
    )
    md.append(
        "- For in-depth details on evaluators, please see the "
        "[Agent Evaluation SDK section in the Azure AI documentation]"
        "(https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/develop/agent-evaluate-sdk)"
    )
    md.append("")

    return "\n".join(md)


# pylint: disable-next=too-many-arguments, too-many-positional-arguments
def append_eval_section(eval_results, baseline, evaluators, result_view, md, section):
    """Append a section of evaluation scores to the markdown summary."""
    eval_scores = []
    for evaluator in section["evaluators"]:
        if evaluator["class"] not in evaluators:
            continue
        for score in evaluator["scores"]:
            if should_include_score(score, evaluator, result_view):
                eval_scores.append(
                    EvaluationScore(
                        name=score["name"],
                        evaluator=evaluator["key"],
                        field=score["key"],
                        data_type=score["type"],
                        desired_direction=score["desired_direction"],
                    )
                )

    if len(eval_scores) > 0:
        md_table = ""
        if len(eval_results) >= 2:
            md_table = fmt_table_compare(eval_scores, eval_results, baseline)
        elif len(eval_results) == 1:
            md_table = fmt_table_ci(eval_scores, eval_results[baseline])

        md.append(f"#### {section['name']}\n")
        md.append(md_table)
        md.append("")
