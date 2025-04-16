from pathlib import Path

import pandas as pd
from azure.ai.projects.models import Agent

from analysis.analysis import EvaluationResult
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
    )

    snapshot.snapshot_dir = Path("tests", "snapshots", "summarize")
    snapshot.assert_match(output, "two_variants.md")
