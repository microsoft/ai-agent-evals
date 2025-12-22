# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""GitHub Action to evaluate Azure AI agents using the Azure AI Evaluation SDK."""

import json
import os
import time
from pathlib import Path

from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import EvaluatorMetricDirection, EvaluatorMetricType
from azure.ai.projects.models._enums import OperationState
from azure.ai.projects.models._models import EvaluationComparisonRequest, Insight
from azure.identity import DefaultAzureCredential
from openai.types.eval_create_params import DataSourceConfigCustom

from analysis import (
    convert_insight_to_comparisons,
    convert_json_to_jsonl,
    process_evaluation_results,
    summarize,
)
from analysis.constants import DEFAULT_EVALUATOR_METADATA

current_dir = Path(__file__).parent
env_path = current_dir / ".env"
if env_path.exists():
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=env_path)


# Configuration constants
class EvaluationConfig:  # pylint: disable=too-few-public-methods
    """Evaluation configuration constants."""

    DEPLOYMENT_NAME_PARAM = "deployment_name"
    POLLING_INTERVAL_SECONDS = 5
    USER_AGENT = "ai-agent-evals/v3-beta (+https://github.com/microsoft/ai-agent-evals)"


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
            agent = project_client.agents.get_version(
                agent_name=agent_name, agent_version=agent_version
            )
            agents[agent_id] = agent
        else:
            raise ValueError(
                f"Invalid agent ID format: {agent_id}. Expected 'name:version'"
            )
    return agents


def _build_metrics_dict(definition) -> dict:
    """Build metrics dictionary from evaluator definition.

    Args:
        definition: Evaluator definition object

    Returns:
        Dictionary mapping metric names to metric metadata
    """
    metrics_dict = {}
    if hasattr(definition, "metrics") and definition.metrics:
        for metric_name, metric in definition.metrics.items():
            metric_type = (
                metric.type
                if hasattr(metric, "type")
                else EvaluatorMetricType.CONTINUOUS
            )
            metric_direction = (
                metric.desirable_direction
                if hasattr(metric, "desirable_direction")
                else EvaluatorMetricDirection.INCREASE
            )
            metrics_dict[metric_name] = {
                "data_type": metric_type,
                "desired_direction": metric_direction,
                "field": metric_name,
            }
    return metrics_dict


def get_evaluator_metadata(
    project_client: AIProjectClient, evaluator_names: list[str]
) -> dict:
    """Get metadata for specific evaluators.

    Args:
        project_client: AI Project client
        evaluator_names: List of evaluator names to fetch metadata for

    Returns:
        Dictionary mapping evaluator names to metadata with data_type and desired_direction
    """
    evaluator_metadata: dict = {}

    for evaluator_name in evaluator_names:
        is_openai_type = False
        is_custom_code = False
        try:
            evaluator = project_client.evaluators.get_version(
                name=evaluator_name, version="latest"
            )

            # Get categories from evaluator
            categories = getattr(evaluator, "categories", [])

            # Get metrics from the evaluator definition
            if hasattr(evaluator, "definition") and evaluator.definition:
                definition = evaluator.definition
                evaluator_type = getattr(definition, "type", "")

                # Check if this is an OpenAI-type evaluator
                is_openai_type = evaluator_type == "openai_graders"

                # Check if custom code evaluator (type="code" without "builtin." prefix)
                is_custom_code = (
                    evaluator_type == "code"
                    and not evaluator_name.startswith("builtin.")
                )

                # Extract init_parameters and metrics from definition
                init_parameters = getattr(definition, "init_parameters", None)
                metrics_dict = _build_metrics_dict(definition)

                evaluator_metadata[evaluator_name] = {
                    "metrics": metrics_dict,
                    "categories": categories,
                    "init_parameters": init_parameters,
                    "data_schema": getattr(definition, "data_schema", {}),
                    "version": getattr(evaluator, "version", "1"),
                    "is_openai_type": is_openai_type,
                    "is_custom_code": is_custom_code,
                }
                continue

        except (  # pylint: disable=broad-exception-caught
            KeyError,
            AttributeError,
            TypeError,
            ValueError,
            Exception,
        ) as e:
            # Custom evaluator or error fetching metadata - use defaults
            print(
                f"Could not fetch metadata for evaluator '{evaluator_name}': {e}. "
                f"Using defaults."
            )

        # Use default metadata (for errors or missing definitions)
        evaluator_metadata[evaluator_name] = DEFAULT_EVALUATOR_METADATA
        evaluator_metadata[evaluator_name]["is_openai_type"] = is_openai_type
        evaluator_metadata[evaluator_name]["is_custom_code"] = is_custom_code

    print(f"Loaded metadata for {len(evaluator_metadata)} evaluators")
    return evaluator_metadata


def _build_openai_evaluator_criteria(
    evaluator_display_name: str, grader_config: dict
) -> dict:
    """Build testing criteria for OpenAI evaluators.

    Args:
        evaluator_display_name: Display name (evaluator name without prefix)
        grader_config: Configuration from openai_graders field containing
                       evaluation_metric, input, reference, etc.

    Returns:
        Testing criteria dictionary for OpenAI evaluator
    """
    criteria = {
        "type": evaluator_display_name,
        "name": evaluator_display_name,
    }

    # Add grader-specific properties (evaluation_metric, input, reference, etc.)
    for key, value in grader_config.items():
        if key not in {"type", "id", "name"}:
            criteria[key] = value

    # Use DEPLOYMENT_NAME if model field is not provided
    if "model" not in criteria and DEPLOYMENT_NAME:
        criteria["model"] = DEPLOYMENT_NAME

    return criteria


def _generate_data_mappings(input_data: dict | None) -> dict:
    """Generate data mappings from input data.

    Args:
        input_data: Input data dictionary containing data_mapping and data fields

    Returns:
        Dictionary of data field mappings
    """
    user_data_mappings = input_data.get("data_mapping", None) if input_data else None

    # Auto-generate data mappings from fields in data items
    if input_data and "data" in input_data and len(input_data["data"]) > 0:
        first_item = input_data["data"][0]
        if user_data_mappings is None:
            user_data_mappings = {}
        # Add all fields from the first data item that aren't already mapped
        for field in first_item.keys():
            user_data_mappings[field] = f"{{{{item.{field}}}}}"

    return user_data_mappings or {}


def _get_response_field(
    evaluator_name: str, categories: list, is_custom_code: bool = False
) -> str:
    """Determine the response field based on evaluator name and categories.

    Args:
        evaluator_name: Name of the evaluator
        categories: List of evaluator categories
        is_custom_code: Whether this is a custom code evaluator

    Returns:
        Response field template string
    """
    # Custom code evaluators always use item.sample.output_text because of an OpenAI limitation
    if is_custom_code:
        return "{{item.sample.output_text}}"
    if evaluator_name == "builtin.groundedness" or categories == ["agents"]:
        return "{{sample.output_items}}"
    return "{{sample.output_text}}"


def _build_base_data_mapping(response_field: str, user_data_mappings: dict) -> dict:
    """Build the base data mapping for an evaluator.

    Args:
        response_field: Response field template string
        user_data_mappings: User-provided data mappings

    Returns:
        Complete data mapping dictionary
    """
    evaluator_data_mapping = {
        "response": response_field,
        "tool_calls": "{{sample.tool_calls}}",
        "tool_definitions": "{{sample.tool_definitions}}",
    }
    evaluator_data_mapping.update(user_data_mappings)
    return evaluator_data_mapping


def _build_azure_evaluator_criteria(
    evaluator_name: str,
    evaluator_display_name: str,
    metadata: dict,
    user_data_mappings: dict,
    evaluator_parameters: dict | None = None,
) -> dict:
    """Build testing criteria for Azure AI evaluators.

    Args:
        evaluator_name: Full evaluator name (e.g., 'builtin.coherence')
        evaluator_display_name: Display name (e.g., 'coherence')
        metadata: Evaluator metadata including categories and schemas
        user_data_mappings: User-provided data mappings from input data
        evaluator_parameters: Optional evaluator-specific parameters

    Returns:
        Testing criteria dictionary for Azure AI evaluator
    """
    categories = metadata.get("categories", [])
    init_params_schema = metadata.get("init_parameters", {})
    data_schema = metadata.get("data_schema", {})
    is_custom_code = metadata.get("is_custom_code", False)

    # Determine response field and build data mapping
    response_field = _get_response_field(evaluator_name, categories, is_custom_code)
    evaluator_data_mapping = _build_base_data_mapping(
        response_field, user_data_mappings
    )

    # Get and validate initialization parameters
    initialization_parameters = {}
    if evaluator_parameters and evaluator_name in evaluator_parameters:
        initialization_parameters = evaluator_parameters[evaluator_name].copy()

    _validate_init_parameters(
        evaluator_name, init_params_schema, initialization_parameters
    )
    _validate_data_schema(evaluator_name, data_schema, evaluator_data_mapping)

    return {
        "type": "azure_ai_evaluator",
        "name": evaluator_display_name,
        "evaluator_name": evaluator_name,
        "initialization_parameters": initialization_parameters,
        "data_mapping": evaluator_data_mapping,
    }


def _validate_init_parameters(
    evaluator_name: str, init_params_schema: dict, initialization_parameters: dict
) -> None:
    """Validate that all required initialization parameters are provided.

    Args:
        evaluator_name: Name of the evaluator
        init_params_schema: Schema defining required parameters
        initialization_parameters: Dictionary of provided parameters

    Raises:
        ValueError: If required parameters are missing
    """
    if not init_params_schema or "required" not in init_params_schema:
        return

    required_params = init_params_schema["required"]

    # Add deployment_name if required and not present
    if (
        EvaluationConfig.DEPLOYMENT_NAME_PARAM in required_params
        and EvaluationConfig.DEPLOYMENT_NAME_PARAM not in initialization_parameters
    ):
        initialization_parameters[EvaluationConfig.DEPLOYMENT_NAME_PARAM] = (
            DEPLOYMENT_NAME
        )

    # Parameters to exclude from validation (auto-populated by system)
    excluded_params = {
        EvaluationConfig.DEPLOYMENT_NAME_PARAM,
        "azure_ai_project",
    }

    # Validate all other required parameters are provided
    missing_params = [
        param
        for param in required_params
        if param not in excluded_params and param not in initialization_parameters
    ]

    if missing_params:
        raise ValueError(
            f"Evaluator '{evaluator_name}' requires the following "
            f"parameters that are not provided: {', '.join(missing_params)}. "
            f"Please add them to 'evaluator_parameters' in your input JSON."
        )


def _validate_data_schema(
    evaluator_name: str, data_schema: dict, evaluator_data_mapping: dict
) -> None:
    """Validate that data mapping satisfies the required data schema.

    Args:
        evaluator_name: Name of the evaluator
        data_schema: Schema defining required data fields
        evaluator_data_mapping: Dictionary of data field mappings

    Raises:
        ValueError: If required data fields are missing
    """
    if not data_schema:
        return

    # Check if schema has anyOf (multiple acceptable combinations)
    if "anyOf" in data_schema:
        any_combination_satisfied = False
        all_missing_combinations = []

        for schema_option in data_schema["anyOf"]:
            if "required" in schema_option:
                required_fields = schema_option["required"]
                missing_fields = [
                    field
                    for field in required_fields
                    if field not in evaluator_data_mapping
                ]

                if not missing_fields:
                    any_combination_satisfied = True
                    break
                all_missing_combinations.append(missing_fields)

        if not any_combination_satisfied:
            combinations_str = " OR ".join(
                f"[{', '.join(combo)}]" for combo in all_missing_combinations
            )
            raise ValueError(
                f"Evaluator '{evaluator_name}' requires at least one of "
                f"these field combinations: {combinations_str}. Please add "
                f"the required fields to 'data_mapping' or ensure they exist "
                f"in your data items."
            )

    # Check if schema has simple required list
    elif "required" in data_schema:
        required_data_fields = data_schema["required"]
        missing_data_fields = [
            field
            for field in required_data_fields
            if field not in evaluator_data_mapping
        ]

        if missing_data_fields:
            raise ValueError(
                f"Evaluator '{evaluator_name}' requires the following data "
                f"fields that are not mapped: {', '.join(missing_data_fields)}. "
                f"Please add them to 'data_mapping' or ensure they exist in "
                f"your data items."
            )


# pylint: disable-next=too-many-locals
def create_testing_criteria(
    evaluators: list[str],
    evaluator_metadata: dict,
    input_data: dict | None = None,  # pylint: disable=redefined-outer-name
    evaluator_parameters: dict | None = None,
) -> tuple[list[dict], dict[str, str]]:
    """Build testing criteria dynamically from evaluator names.

    Args:
        evaluators: List of evaluator names
        evaluator_metadata: Dictionary with evaluator metadata including category
        input_data: Input data dictionary containing data_mapping and data fields
        evaluator_parameters: Optional dictionary of evaluator-specific initialization parameters

    Returns:
        Tuple of (testing_criteria list, display_name_to_evaluator_name mapping dict)
    """
    # Generate data mappings from input data
    user_data_mappings = _generate_data_mappings(input_data)

    # Get openai_graders definitions from input data (for custom evaluator types)
    openai_graders = input_data.get("openai_graders", {}) if input_data else {}

    testing_criteria = []
    display_name_to_evaluator_name = {}
    for evaluator_name in evaluators:
        evaluator_display_name = (
            evaluator_name.split(".")[-1] if "." in evaluator_name else evaluator_name
        )

        # Store mapping from display name to actual evaluator name
        display_name_to_evaluator_name[evaluator_display_name] = evaluator_name

        # Get metadata for this evaluator
        metadata = evaluator_metadata.get(evaluator_name, {})
        is_openai_type = metadata.get("is_openai_type", False)

        # Check if this is an OpenAI-type evaluator or has config in openai_graders
        if is_openai_type or evaluator_name in openai_graders:
            # Build custom evaluator criteria with grader-specific properties
            grader_config = openai_graders.get(evaluator_name, {})
            if not grader_config:
                raise ValueError(
                    f"OpenAI-type evaluator '{evaluator_name}' requires "
                    f"configuration in 'openai_graders' field of input data."
                )
            criteria = _build_openai_evaluator_criteria(
                evaluator_display_name, grader_config
            )
        else:
            # Build standard Azure AI evaluator criteria
            criteria = _build_azure_evaluator_criteria(
                evaluator_name,
                evaluator_display_name,
                metadata,
                user_data_mappings,
                evaluator_parameters,
            )

        testing_criteria.append(criteria)

    return testing_criteria, display_name_to_evaluator_name


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
            data_source=data_source,  # type: ignore
        )
        agent_eval_runs[agent_id] = agent_eval_run

    print(f"Created evaluation runs for {len(agent_eval_runs)} agent(s)")
    return agent_eval_runs


def wait_for_evaluation_runs(openai_client, eval_object, agent_eval_runs: dict):
    """Wait for all evaluation runs to complete."""
    print("Waiting for evaluation runs to complete...")
    while True:
        all_completed = True
        for agent_id, eval_run in agent_eval_runs.items():
            if eval_run.status not in ["completed", "failed"]:
                eval_run = openai_client.evals.runs.retrieve(
                    run_id=eval_run.id, eval_id=eval_object.id
                )
                agent_eval_runs[agent_id] = eval_run
                if eval_run.status not in ["completed", "failed"]:
                    all_completed = False

        if all_completed:
            break
        time.sleep(EvaluationConfig.POLLING_INTERVAL_SECONDS)

    print(f"All {len(agent_eval_runs)} evaluation run(s) completed")


def print_agent_results(agent_results: dict):
    """Print evaluation results for an agent."""
    agent = agent_results["agent"]
    evaluator_count = len(agent_results["evaluation_scores"])
    print(f"Processed results for {agent.name} ({evaluator_count} evaluators)")


# pylint: disable-next=too-many-arguments,too-many-positional-arguments
def generate_comparison_insight(
    project_client: AIProjectClient,
    eval_object,
    baseline_run_id: str,
    treatment_run_ids: list[str],
    baseline_agent_id: str,
    treatment_agent_ids: list[str],
) -> Insight | None:
    """Generate comparison insights between baseline and treatment evaluation runs."""
    print(
        f"Generating comparison insight (baseline: {baseline_agent_id} "
        f"vs {len(treatment_agent_ids)} treatment(s))..."
    )

    compare_insight = project_client.insights.generate(
        Insight(
            display_name="Agent Evaluation Comparison",
            request=EvaluationComparisonRequest(
                eval_id=eval_object.id,
                baseline_run_id=baseline_run_id,
                treatment_run_ids=treatment_run_ids,
            ),
        )
    )

    # Wait for insight generation to complete
    while compare_insight.state not in [
        OperationState.SUCCEEDED,
        OperationState.FAILED,
    ]:
        compare_insight = project_client.insights.get(id=compare_insight.id)
        time.sleep(EvaluationConfig.POLLING_INTERVAL_SECONDS)

    if compare_insight.state == OperationState.SUCCEEDED:
        print("Comparison insight generated successfully")
        return compare_insight

    print("Comparison insight generation failed")
    return None


def create_evaluation_and_dataset(
    openai_client,
    project_client,
    input_data_path: Path,
    input_data: dict,
    evaluator_metadata: dict,
) -> tuple:
    """Create evaluation object and upload dataset.

    Args:
        openai_client: OpenAI client
        project_client: AI Project client
        input_data_path: Path to input data file
        input_data: Input data dictionary
        evaluator_metadata: Evaluator metadata with categories

    Returns:
        Tuple of (eval_object, dataset, display_name_to_evaluator_name)
    """
    data_source_config = DataSourceConfigCustom(
        type="custom",
        item_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        include_sample_schema=True,
    )

    # Get evaluator-specific parameters from input data if provided
    evaluator_parameters = input_data.get("evaluator_parameters", None)

    # Build testing criteria dynamically from evaluators in input data
    testing_criteria, display_name_to_evaluator_name = create_testing_criteria(
        input_data.get("evaluators", []),
        evaluator_metadata,
        input_data,
        evaluator_parameters,
    )

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

    return eval_object, dataset, display_name_to_evaluator_name


# pylint: disable-next=too-many-arguments,too-many-positional-arguments
def generate_and_print_comparisons(
    project_client,
    eval_object,
    agent_ids: list[str],
    baseline_agent_id: str | None,
    agent_eval_runs: dict,
    evaluator_metadata: dict,
) -> tuple[dict, Insight | None]:
    """Generate comparison insights for multiple agents.

    Returns:
        Tuple of (comparisons_by_evaluator dict, comparison_insight Insight object or None)
    """
    if len(agent_ids) <= 1:
        return {}, None

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
        treatment_ids,
    )

    if not comparison_insight:
        return {}, None

    # Convert insight to EvaluationScoreComparison objects
    treatment_agent_ids = [aid for aid in agent_ids if aid != baseline_id]
    comparisons_by_evaluator = convert_insight_to_comparisons(
        comparison_insight,
        baseline_id,
        treatment_agent_ids,
        evaluator_metadata,
    )

    return comparisons_by_evaluator, comparison_insight


# pylint: disable-next=too-many-locals
def main(
    endpoint: str,
    input_data_path: Path,
    input_data: dict,
    agent_ids: list[str],
    baseline_agent_id: str | None = None,
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
        eval_object, dataset, display_name_to_evaluator_name = (
            create_evaluation_and_dataset(
                openai_client,
                project_client,
                input_data_path,
                input_data,
                evaluator_metadata,
            )
        )

        # Execute evaluation runs for all agents
        agent_eval_runs = create_evaluation_runs(
            openai_client, eval_object, dataset, agents
        )
        wait_for_evaluation_runs(openai_client, eval_object, agent_eval_runs)

        # Extract report URLs from all completed eval runs
        report_urls = {}
        eval_base_url = None
        for agent_id, eval_run in agent_eval_runs.items():
            report_url = getattr(eval_run, "report_url", None)
            if report_url:
                report_urls[agent_id] = report_url
                # Extract evaluation base URL (remove last 2 path segments)
                if not eval_base_url:
                    parts = report_url.rsplit("/", 2)
                    eval_base_url = parts[0] if len(parts) > 2 else report_url
        print(f"Collected {len(report_urls)} evaluation report URL(s)")

        # Determine baseline agent
        baseline_id = baseline_agent_id if baseline_agent_id else agent_ids[0]
        baseline_agent = agents[baseline_id]
        baseline_eval_run = agent_eval_runs[baseline_id]

        # Process baseline agent results (needed for summary in all cases)
        baseline_results = process_evaluation_results(
            openai_client,
            eval_object,
            baseline_eval_run,
            baseline_agent,
            evaluator_metadata,
            display_name_to_evaluator_name,
        )
        print_agent_results(baseline_results)

        # Generate comparison insights if multiple agents
        # (uses API, doesn't need individual processing)
        comparisons_by_evaluator: dict[str, list] = {}
        compare_url = None
        if len(agent_ids) > 1:
            comparisons_by_evaluator, comparison_insight = (
                generate_and_print_comparisons(
                    project_client,
                    eval_object,
                    agent_ids,
                    baseline_agent_id,
                    agent_eval_runs,
                    evaluator_metadata,
                )
            )
            # Build compare URL if insight available
            if comparison_insight and eval_base_url:
                insight_id = comparison_insight.id
                compare_url = f"{eval_base_url}/compare/{insight_id}"

        # Build evaluator catalog base URL (remove eval ID if present)
        evaluator_catalog_url = ""
        if eval_base_url:
            # eval_base_url is like: .../build/evaluations/eval_xxx
            # We want: .../build/evaluations
            if "/build/evaluations/" in eval_base_url:
                evaluator_catalog_url = eval_base_url.rsplit("/", 1)[0]
            else:
                # Already at the right level
                evaluator_catalog_url = eval_base_url

        # Generate and return summary markdown
        return summarize(
            baseline_results=baseline_results,
            comparisons_by_evaluator=(
                comparisons_by_evaluator if len(agent_ids) > 1 else None
            ),
            report_urls=report_urls,
            eval_url=eval_base_url,
            compare_url=compare_url,
            evaluator_metadata=evaluator_metadata,
            evaluator_catalog_url=evaluator_catalog_url,
        )


def _validate_environment_variables() -> dict:
    """Validate and return required environment variables.

    Returns:
        Dictionary with validated environment variables

    Raises:
        ValueError: If any required environment variable is missing or invalid
    """
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

    return {
        "endpoint": AZURE_AI_PROJECT_ENDPOINT,
        "deployment_name": DEPLOYMENT_NAME,
        "data_path": DATA_PATH,
        "agent_ids": AGENT_IDS,
        "baseline_agent_id": BASELINE_AGENT_ID,
    }


if __name__ == "__main__":
    # Validate environment variables
    env_config = _validate_environment_variables()

    # Load input data
    try:
        data_path = Path(env_config["data_path"])
        data = json.loads(data_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Input data at {env_config['data_path']} is not valid JSON"
        ) from exc

    # Run evaluation and output summary
    SUMMARY_MD = main(
        endpoint=env_config["endpoint"],
        input_data_path=data_path,
        input_data=data,
        agent_ids=env_config["agent_ids"],
        baseline_agent_id=env_config["baseline_agent_id"],
    )

    if STEP_SUMMARY:
        with open(STEP_SUMMARY, "a", encoding="utf-8") as fp:
            fp.write(SUMMARY_MD)

    if env_path.exists():
        with open(Path(".") / "evaluation.md", "a", encoding="utf-8") as fp:
            fp.write(SUMMARY_MD)
