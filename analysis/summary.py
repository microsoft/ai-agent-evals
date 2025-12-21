# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Summary module for formatting and generating evaluation result summaries.

This module provides functionality to generate formatted markdown summaries
of evaluation results for AI agents. It includes functions to create
tables comparing multiple agent variants or displaying confidence intervals
for a single agent's performance metrics.
"""
from .analysis import EvaluationResultView
from .render import fmt_hyperlink, fmt_table_ci, fmt_table_compare


# pylint: disable-next=too-many-locals, too-many-arguments, too-many-positional-arguments
def summarize(
    baseline_results: dict,
    comparisons_by_evaluator: dict[str, list] | None = None,
    report_urls: dict[str, str] | None = None,
) -> str:
    """Generate a markdown summary of evaluation results.

    Args:
        baseline_results: Dictionary containing baseline agent results with keys:
            - 'evaluation_scores': Dict mapping evaluator names to EvaluationScoreCI objects
            - 'agent': Agent object
            - 'evaluator_names': List of evaluator names
        comparisons_by_evaluator: Optional dictionary of comparison results for multiple agents.
            Each evaluator maps to a list of EvaluationScoreComparison objects.
        report_urls: Optional dictionary mapping agent IDs to their report URLs

    Returns:
        Formatted markdown string with evaluation summary
    """
    # Extract fields from baseline_results
    evaluation_scores = baseline_results['evaluation_scores']
    agent = baseline_results['agent']
    agent_name = f"{agent.name}:{agent.version}"
    
    # Extract treatment agent names from comparisons if available
    treatment_agent_names = None
    if comparisons_by_evaluator:
        # Get treatment names from first evaluator's comparisons
        first_evaluator_comparisons = next(iter(comparisons_by_evaluator.values()), [])
        if first_evaluator_comparisons:
            treatment_agent_names = [comp.treatment_variant for comp in first_evaluator_comparisons]
    
    md = []
    md.append("## Azure AI Evaluation\n")

    # Show agents section - list all agents if comparisons available
    if comparisons_by_evaluator and treatment_agent_names:
        md.append("### Agents\n")
        md.append("| Agent ID | Role | Evaluation results |")
        md.append("|:---------|:-----|:-------------------|")
        
        # Use per-agent URLs from report_urls dictionary
        baseline_url = report_urls.get(agent_name) if report_urls else None
        baseline_link = fmt_hyperlink("Click here", baseline_url) if baseline_url else ""
        md.append(f"| {agent_name} | Baseline | {baseline_link} |")
        
        for treatment_name in treatment_agent_names:
            treatment_url = report_urls.get(treatment_name) if report_urls else None
            treatment_link = fmt_hyperlink("Click here", treatment_url) if treatment_url else ""
            md.append(f"| {treatment_name} | Treatment | {treatment_link} |")
    else:
        md.append("### Agent\n")
        md.append("| Agent ID | Evaluation results |")
        md.append("|:---------|:-------------------|")
        agent_url = report_urls.get(agent_name) if report_urls else None
        result_link = fmt_hyperlink("Click here", agent_url) if agent_url else ""
        md.append(f"| {agent_name} | {result_link} |")

    md.append("\n### Evaluation results\n")
    
    # Generate comparison table if comparisons are available, otherwise show CI table
    if comparisons_by_evaluator and treatment_agent_names:
        md_table = fmt_table_compare(comparisons_by_evaluator, agent_name)
    else:
        md_table = fmt_table_ci(evaluation_scores, agent_name)
    md.append(md_table)
    md.append("")

    md.append("### References\n")
    md.append(
        "- For in-depth details on evaluators, please see the "
        "[Agent Evaluation SDK section in the Azure AI documentation]"
        "(https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/develop/agent-evaluate-sdk)"
    )
    md.append("")

    return "\n".join(md)
