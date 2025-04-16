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

from .analysis import EvaluationResult, EvaluationScore
from .render import fmt_hyperlink, fmt_table_ci, fmt_table_compare


# pylint: disable-next=too-many-locals
def summarize(
    eval_results: dict[str, EvaluationResult],
    agents: dict[str, Agent],
    baseline: str,
    evaluators: list[str],
    agent_base_url: str,
) -> str:
    """Generate a markdown summary of evaluation results.

    Args:
        eval_results: Dictionary mapping agent IDs to their evaluation results
        agents: Dictionary mapping agent IDs to Agent objects
        baseline: ID of the baseline agent for comparisons
        evaluators: List of evaluator class names to include in the summary
        agent_base_url: Base URL for agent links

    Returns:
        Formatted markdown string with evaluation summary
    """
    md = []
    md.append("## Azure AI Evaluation\n")

    def format_agent_row(agent: Agent, agent_url: str) -> str:
        result_url = eval_results[agent.id].ai_foundry_url
        return (
            f"| {agent.name} | "
            f"{fmt_hyperlink(agent.id, agent_url)} | "
            f"{fmt_hyperlink('Click here', result_url) if result_url else ""} |"
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
    with open(metadata_path, "r", encoding="utf-8") as f:
        score_metadata = yaml.safe_load(f)

    for section in score_metadata["sections"]:
        section_evals = [x["class"] for x in section["evaluators"]]
        if not any(x in evaluators for x in section_evals):
            continue

        eval_scores = []
        for evaluator in section["evaluators"]:
            if evaluator["class"] not in evaluators:
                continue
            for score in evaluator["scores"]:
                eval_scores.append(
                    EvaluationScore(
                        name=score["name"],
                        evaluator=evaluator["key"],
                        field=score["key"],
                        data_type=score["type"],
                        desired_direction=score["desired_direction"],
                    )
                )

        md_table = ""
        if len(eval_results) >= 2:
            md.append("\n### Compare evaluation scores between variants\n")
            md_table = fmt_table_compare(eval_scores, eval_results, baseline)
        elif len(eval_results) == 1:
            md.append("\n### Evaluation results\n")
            md_table = fmt_table_ci(eval_scores, eval_results[baseline])

        md.append(f"#### {section['name']}\n")
        md.append(md_table)
        md.append("")

    return "\n".join(md)
