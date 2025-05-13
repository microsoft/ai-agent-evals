"""
Tests for the summary functionality in the analysis module.

The tests use snapshot testing to verify the output matches expected results.
"""

from pathlib import Path

import pandas as pd
from azure.ai.agents.models import Agent

from analysis.analysis import EvaluationResult, EvaluationResultView
from analysis.summary import summarize

agent_1 = Agent(id="agent.v1", name="agent_version_1")
agent_2 = Agent(id="agent.v2", name="agent_version_2")

data_result_1 = {
    "inputs.id": [1, 2, 3],
    "outputs.fluency.fluency": [0.8, 0.9, 0.85],
    "outputs.relevance.relevance": [4, 5, 4],
}

data_result_2 = {
    "inputs.id": [1, 2, 3],
    "outputs.fluency.fluency": [0.6, 0.5, 0.75],
    "outputs.relevance.relevance": [3, 4, 5],
}


def test_summarize_one_variant(snapshot):
    """Test summary of the analysis for 1 variant."""

    result_1 = EvaluationResult(
        variant=agent_1.id,
        df_result=pd.DataFrame(data_result_1),
        ai_foundry_url="test_url_1",
    )
    results = {agent_1.id: result_1}
    agents = {agent_1.id: agent_1}
    output = summarize(
        eval_results=results,
        agents=agents,
        baseline=agent_1.id,
        evaluators=["FluencyEvaluator", "RelevanceEvaluator"],
        agent_base_url="https://ai-url/",
        result_view=EvaluationResultView.ALL,
    )

    snapshot.snapshot_dir = Path("tests", "snapshots", "summarize")
    snapshot.assert_match(output, "one_variant.md")


def test_summarize_multiple_variants(snapshot):
    """Test summary of the analysis for multiple variants."""

    result_1 = EvaluationResult(
        variant=agent_1.id,
        df_result=pd.DataFrame(data_result_1),
        ai_foundry_url="test_url_1",
    )
    result_2 = EvaluationResult(
        variant=agent_2.id,
        df_result=pd.DataFrame(data_result_2),
        ai_foundry_url="test_url_1",
    )
    results = {agent_1.id: result_1, agent_2.id: result_2}

    agents = {agent_1.id: agent_1, agent_2.id: agent_2}
    output = summarize(
        eval_results=results,
        agents=agents,
        baseline=agent_1.id,
        evaluators=["FluencyEvaluator", "RelevanceEvaluator"],
        agent_base_url="https://ai-url/",
        result_view=EvaluationResultView.ALL,
    )

    snapshot.snapshot_dir = Path("tests", "snapshots", "summarize")
    snapshot.assert_match(output, "two_variants.md")


def test_summary_with_different_views(snapshot):
    """Test summary generation with different evaluation result views."""

    # Create test data with both continuous and boolean results
    test_data = {
        "inputs.id": ["test1", "test2", "test3"],
        "outputs.fluency.fluency": [0.8, 0.9, 0.7],  # Continuous score
        "outputs.fluency.fluency_result": [True, True, False],  # Boolean result
        "outputs.relevance.relevance": [4, 5, 3],  # Ordinal score
        "outputs.relevance.relevance_result": [True, True, False],  # Boolean result
    }

    result = EvaluationResult(
        variant=agent_1.id,
        df_result=pd.DataFrame(test_data),
        ai_foundry_url="test_url_1",
    )
    results = {agent_1.id: result}
    agents = {agent_1.id: agent_1}

    # Test DEFAULT view
    default_output = summarize(
        eval_results=results,
        agents=agents,
        baseline=agent_1.id,
        evaluators=["FluencyEvaluator", "RelevanceEvaluator"],
        agent_base_url="https://ai-url/",
        result_view=EvaluationResultView.DEFAULT,
    )

    # Test ALL view
    all_output = summarize(
        eval_results=results,
        agents=agents,
        baseline=agent_1.id,
        evaluators=["FluencyEvaluator", "RelevanceEvaluator"],
        agent_base_url="https://ai-url/",
        result_view=EvaluationResultView.ALL,
    )

    # Test RAW_SCORES view
    raw_output = summarize(
        eval_results=results,
        agents=agents,
        baseline=agent_1.id,
        evaluators=["FluencyEvaluator", "RelevanceEvaluator"],
        agent_base_url="https://ai-url/",
        result_view=EvaluationResultView.RAW_SCORES,
    )

    snapshot.snapshot_dir = Path("tests", "snapshots", "summarize")
    snapshot.assert_match(default_output, "default_view.md")
    snapshot.assert_match(all_output, "all_view.md")
    snapshot.assert_match(raw_output, "raw_scores_view.md")
