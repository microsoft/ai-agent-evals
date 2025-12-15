# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""GitHub Action to evaluate Azure AI agents using the Azure AI Evaluation SDK."""

import json
import os
import time
from pathlib import Path

from azure.ai.projects import AIProjectClient
from azure.ai.projects.models._enums import OperationState
from azure.ai.projects.models._models import EvaluationComparisonRequest, Insight
from azure.ai.projects.models import EvaluatorMetricType, EvaluatorMetricDirection
from azure.identity import DefaultAzureCredential
from openai.types.eval_create_params import DataSourceConfigCustom

from analysis import (
    convert_json_to_jsonl,
    process_evaluation_results,
    convert_insight_to_comparisons,
    summarize,
)

current_dir = Path(__file__).parent
env_path = current_dir / ".env"
if env_path.exists():
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=env_path)

USER_AGENT = "ai-agent-evals/v2-beta (+https://github.com/microsoft/ai-agent-evals)"
STEP_SUMMARY = os.getenv("GITHUB_STEP_SUMMARY") or os.getenv("ADO_STEP_SUMMARY")

AZURE_AI_PROJECT_ENDPOINT = os.getenv("AZURE_AI_PROJECT_ENDPOINT")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")
DATA_PATH = os.getenv("DATA_PATH")
AGENT_IDS = [x.strip() for x in os.getenv("AGENT_IDS", "").split(",") if x.strip()]
BASELINE_AGENT_ID = os.getenv("BASELINE_AGENT_ID")


def get_agents(project_client: AIProjectClient, agent_ids: list[str]) -> dict:
    """Parse and retrieve agent objects from agent IDs."""
    agents = {}
    for agent_id in agent_ids:
        agent_name_version = agent_id.split(":")
        if len(agent_name_version) == 2:
            agent_name = agent_name_version[0]
            agent_version = agent_name_version[1]
            agent = project_client.agents.get_version(agent_name=agent_name, agent_version=agent_version)
            agents[agent_id] = agent
        else:
            raise ValueError(f"Invalid agent ID format: {agent_id}. Expected 'name:version'")
    return agents


def get_evaluator_metadata(project_client: AIProjectClient, evaluator_names: list[str]) -> dict:
    """Get metadata for specific evaluators.
    
    Args:
        project_client: AI Project client
        evaluator_names: List of evaluator names to fetch metadata for
    
    Returns:
        Dictionary mapping evaluator names to metadata with data_type and desired_direction
    """
    evaluator_metadata = {}
    
    for evaluator_name in evaluator_names:
        try:
            # Try to get the evaluator (works for built-in evaluators)
            evaluator = project_client.evaluators.get(name=evaluator_name)
            
            # Get metrics from the evaluator definition
            if hasattr(evaluator, 'definition') and evaluator.definition:
                definition = evaluator.definition
                if hasattr(definition, 'metrics') and definition.metrics:
                    # Get first metric (evaluators typically have one primary metric)
                    for metric_name, metric in definition.metrics.items():
                        # Use SDK enums directly
                        metric_type = metric.type if hasattr(metric, 'type') else EvaluatorMetricType.CONTINUOUS
                        metric_direction = metric.desirable_direction if hasattr(metric, 'desirable_direction') else EvaluatorMetricDirection.INCREASE
                        
                        evaluator_metadata[evaluator_name] = {
                            'data_type': metric_type,
                            'desired_direction': metric_direction,
                            'field': metric_name
                        }
                        break  # Use first metric
        
        except Exception as e:
            # Custom evaluator or error fetching metadata - use defaults
            evaluator_metadata[evaluator_name] = {
                'data_type': EvaluatorMetricType.CONTINUOUS,
                'desired_direction': EvaluatorMetricDirection.INCREASE,
                'field': 'score'
            }
    
    print(f"Loaded metadata for {len(evaluator_metadata)} evaluators")
    return evaluator_metadata

#TODO: support OAI graders and parameters from input json
def create_testing_criteria(evaluators: list[str]) -> list[dict]:
    """Build testing criteria dynamically from evaluator names."""
    testing_criteria = []
    for evaluator_name in evaluators:
        evaluator_display_name = evaluator_name.split('.')[-1] if '.' in evaluator_name else evaluator_name
        testing_criteria.append({
            "type": "azure_ai_evaluator",
            "name": evaluator_display_name,
            "evaluator_name": evaluator_name,
            "initialization_parameters": {
                "deployment_name": DEPLOYMENT_NAME
            },
            "data_mapping": {
                "query": "{{item.query}}", 
                "response": "{{sample.output_items}}",
                "tool_calls": "{{sample.tool_calls}}",
                "tool_definitions": "{{sample.tool_definitions}}"
            },
        })
    return testing_criteria


def create_evaluation_runs(openai_client, eval_object, dataset, agents: dict) -> dict:
    """Create evaluation runs for each agent."""
    agent_eval_runs = {}
    for agent_id, agent in agents.items():
        data_source = {
            "type": "azure_ai_target_completions",
            "source": {
                "type": "file_id",
                "id": dataset.id,
            },
            "input_messages": {
                "type": "template",
                "template": [
                    {"type": "message", "role": "user", "content": "{{item.query}}"}
                ],
            },
            "target": {
                "type": "azure_ai_agent",
                "name": agent.name,
                "version": agent.version,
            },
        }

        agent_eval_run = openai_client.evals.runs.create(
            eval_id=eval_object.id, 
            name=f"Agent {agent_id}", 
            data_source=data_source  # type: ignore
        )
        agent_eval_runs[agent_id] = agent_eval_run
    
    print(f"Created evaluation runs for {len(agent_eval_runs)} agent(s)")
    return agent_eval_runs


def wait_for_evaluation_runs(openai_client, eval_object, agent_eval_runs: dict, agents: dict):
    """Wait for all evaluation runs to complete."""
    print("Waiting for evaluation runs to complete...")
    while True:
        all_completed = True
        for agent_id, eval_run in agent_eval_runs.items():
            if eval_run.status not in ["completed", "failed"]:
                eval_run = openai_client.evals.runs.retrieve(run_id=eval_run.id, eval_id=eval_object.id)
                agent_eval_runs[agent_id] = eval_run
                if eval_run.status not in ["completed", "failed"]:
                    all_completed = False
        
        if all_completed:
            break
        time.sleep(5)

    print(f"All {len(agent_eval_runs)} evaluation run(s) completed")


def print_agent_results(agent_results: dict):
    """Print evaluation results for an agent."""
    agent = agent_results['agent']
    evaluator_count = len(agent_results['evaluation_scores'])
    print(f"Processed results for {agent.name} ({evaluator_count} evaluators)")


def generate_comparison_insight(
    project_client: AIProjectClient, 
    eval_object, 
    baseline_run_id: str, 
    treatment_run_ids: list[str],
    baseline_agent_id: str,
    treatment_agent_ids: list[str]
):
    """Generate comparison insights between baseline and treatment evaluation runs."""
    print(f"Generating comparison insight (baseline: {baseline_agent_id} vs {len(treatment_agent_ids)} treatment(s))...")
    
    compare_insight = project_client.insights.generate(
        Insight(
            display_name="Agent Evaluation Comparison",
            request=EvaluationComparisonRequest(
                eval_id=eval_object.id,
                baseline_run_id=baseline_run_id,
                treatment_run_ids=treatment_run_ids
            ),
        )
    )
    
    # Wait for insight generation to complete
    while compare_insight.state not in [OperationState.SUCCEEDED, OperationState.FAILED]:
        compare_insight = project_client.insights.get(id=compare_insight.id)
        time.sleep(5)
    
    if compare_insight.state == OperationState.SUCCEEDED:
        print("Comparison insight generated successfully")
        return compare_insight
    else:
        print("Comparison insight generation failed")
        return None


def create_evaluation_and_dataset(openai_client, project_client, input_data_path: Path, input_data: dict) -> tuple:
    """Create evaluation object and upload dataset.
    
    Returns:
        Tuple of (eval_object, dataset)
    """
    data_source_config = DataSourceConfigCustom(
        type="custom",
        item_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        include_sample_schema=True,
    )
    
    # Build testing criteria dynamically from evaluators in input data
    testing_criteria = create_testing_criteria(input_data.get("evaluators", []))
    
    eval_object = openai_client.evals.create(
        name="Agent Evaluation",
        data_source_config=data_source_config,
        testing_criteria=testing_criteria,  # type: ignore
    )
    print(f"Created evaluation with {len(testing_criteria)} evaluator(s)")

    # Convert JSON to JSONL format
    jsonl_path = convert_json_to_jsonl(input_data_path)

    dataset = project_client.datasets.upload_file(
        name=input_data_path.stem,
        version=str(int(time.time())),
        file_path=jsonl_path,
    )
    print(f"Uploaded dataset: {dataset.name} (version: {dataset.version})")
    
    return eval_object, dataset


def process_all_agent_results(
    openai_client,
    eval_object,
    agent_eval_runs: dict,
    agents: dict,
    evaluator_metadata: dict
) -> dict:
    """Process evaluation results for all agents.
    
    Returns:
        Dictionary mapping agent IDs to their results
    """
    all_agent_results = {}
    for agent_id, eval_run in agent_eval_runs.items():
        agent = agents[agent_id]
        agent_results = process_evaluation_results(openai_client, eval_object, eval_run, agent, evaluator_metadata)
        all_agent_results[agent_id] = agent_results
        print_agent_results(agent_results)
    return all_agent_results


def generate_and_print_comparisons(
    project_client,
    eval_object,
    agent_ids: list[str],
    baseline_agent_id: str | None,
    agent_eval_runs: dict,
    evaluator_names: list[str],
    evaluator_metadata: dict
) -> dict:
    """Generate comparison insights for multiple agents.
    
    Returns:
        Dictionary of comparisons by evaluator
    """
    if len(agent_ids) <= 1:
        return {}
    
    # Use baseline agent if specified, otherwise use first agent
    baseline_id = baseline_agent_id if baseline_agent_id else agent_ids[0]
    baseline_run_id = agent_eval_runs[baseline_id].id
    
    # Get treatment run IDs (all agents except baseline)
    treatment_ids = [aid for aid in agent_ids if aid != baseline_id]
    treatment_run_ids = [agent_eval_runs[aid].id for aid in treatment_ids]
    
    comparison_insight = generate_comparison_insight(
        project_client,
        eval_object,
        baseline_run_id,
        treatment_run_ids,
        baseline_id,
        treatment_ids
    )
    
    if not comparison_insight:
        return {}
    
    # Convert insight to EvaluationScoreComparison objects
    treatment_agent_ids = [aid for aid in agent_ids if aid != baseline_id]
    comparisons_by_evaluator = convert_insight_to_comparisons(
        comparison_insight,
        baseline_id,
        treatment_agent_ids,
        evaluator_names,
        evaluator_metadata
    )
    
    return comparisons_by_evaluator


def main(
    endpoint: str,
    input_data_path: str,
    input_data: dict,
    agent_ids: list[str],
    baseline_agent_id: str | None = None,
    working_dir: Path | None = None,
) -> str:
    """Main evaluation workflow.
    
    Orchestrates the complete evaluation process:
    1. Setup: Get agents and evaluator metadata
    2. Create evaluation and upload dataset
    3. Execute evaluation runs for all agents
    4. Process and analyze results
    5. Generate comparison insights (if multiple agents)
    6. Create summary markdown report
    """
    with (
        DefaultAzureCredential() as credential,
        AIProjectClient(endpoint=endpoint, credential=credential) as project_client,
        project_client.get_openai_client() as openai_client,
    ):
        # Setup: Parse agents and get evaluator metadata
        agents = get_agents(project_client, agent_ids)
        evaluator_names = input_data.get("evaluators", [])
        evaluator_metadata = get_evaluator_metadata(project_client, evaluator_names)
    
        # Create evaluation and prepare dataset
        eval_object, dataset = create_evaluation_and_dataset(
            openai_client, project_client, input_data_path, input_data
        )

        # Execute evaluation runs for all agents
        agent_eval_runs = create_evaluation_runs(openai_client, eval_object, dataset, agents)
        wait_for_evaluation_runs(openai_client, eval_object, agent_eval_runs, agents)

        # Extract report URLs from all completed eval runs
        report_urls = {}
        for agent_id, eval_run in agent_eval_runs.items():
            report_url = getattr(eval_run, 'report_url', None)
            if report_url:
                report_urls[agent_id] = report_url
        print(f"Collected {len(report_urls)} evaluation report URL(s)")

        # Determine baseline agent
        baseline_id = baseline_agent_id if baseline_agent_id else agent_ids[0]
        baseline_agent = agents[baseline_id]
        baseline_eval_run = agent_eval_runs[baseline_id]
        
        # Process baseline agent results (needed for summary in all cases)
        baseline_results = process_evaluation_results(
            openai_client, eval_object, baseline_eval_run, baseline_agent, evaluator_metadata
        )
        print_agent_results(baseline_results)
        
        # Generate comparison insights if multiple agents (uses API, doesn't need individual processing)
        comparisons_by_evaluator = {}
        if len(agent_ids) > 1:
            comparisons_by_evaluator = generate_and_print_comparisons(
                project_client, eval_object, agent_ids, baseline_agent_id,
                agent_eval_runs, evaluator_names, evaluator_metadata
            )
        
        # Generate and return summary markdown
        return summarize(
            baseline_results=baseline_results,
            comparisons_by_evaluator=comparisons_by_evaluator if len(agent_ids) > 1 else None,
            report_urls=report_urls,
        )



if __name__ == "__main__":
    # Check required environment variables
    if not AZURE_AI_PROJECT_ENDPOINT:
        raise ValueError("AZURE_AI_PROJECT_ENDPOINT environment variable is not set")
    if not DEPLOYMENT_NAME:
        raise ValueError("DEPLOYMENT_NAME environment variable is not set or empty")
    if not DATA_PATH:
        raise ValueError("DATA_PATH environment variable is not set")
    if not AGENT_IDS:
        raise ValueError("AGENT_IDS environment variable is not set or empty")

    # Check optional environment variables
    if BASELINE_AGENT_ID and BASELINE_AGENT_ID not in AGENT_IDS:
        raise ValueError(
            f"BASELINE_AGENT_ID '{BASELINE_AGENT_ID}' is not in AGENT_IDS '{AGENT_IDS}'"
        )

    # Load input data
    try:
        input_data_path = Path(DATA_PATH)
        input_data = json.loads(input_data_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Input data at {DATA_PATH} is not valid JSON") from exc

    # Run evaluation and output summary
    SUMMARY_MD = main(
        endpoint=AZURE_AI_PROJECT_ENDPOINT,
        input_data_path=input_data_path,
        input_data=input_data,
        agent_ids=AGENT_IDS,
        baseline_agent_id=BASELINE_AGENT_ID,
        working_dir=input_data_path.parent,
    )

    if STEP_SUMMARY:
        with open(STEP_SUMMARY, "a", encoding="utf-8") as fp:
            fp.write(SUMMARY_MD)

    if env_path.exists():
        with open(Path(".") / "evaluation.md", "a", encoding="utf-8") as fp:
            fp.write(SUMMARY_MD)
