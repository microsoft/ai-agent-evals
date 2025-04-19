# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""GitHub Action to evaluate Azure AI agents using the Azure AI Evaluation SDK."""

import inspect
import json
import os
import time
import uuid
from pathlib import Path
from typing import Optional

import azure.ai.evaluation as evals
import pandas as pd
import yaml
from azure.ai.evaluation import evaluate
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import Agent, ConnectionType, MessageRole, RunStatus
from azure.identity import DefaultAzureCredential

import analysis

# NOTE: custom evaluators must be imported so evaluate() can pickle them


current_dir = Path(__file__).parent
env_path = current_dir / ".env"
if env_path.exists():
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=env_path)

STEP_SUMMARY = os.getenv("GITHUB_STEP_SUMMARY") or os.getenv("ADO_STEP_SUMMARY")  

AZURE_AI_PROJECT_CONNECTION_STRING = os.getenv("AZURE_AI_PROJECT_CONNECTION_STRING")
DATA_PATH = os.getenv("DATA_PATH")
AGENT_IDS = [x.strip() for x in os.getenv("AGENT_IDS", "").split(",") if x.strip()]
BASELINE_AGENT_ID = os.getenv("BASELINE_AGENT_ID")

# TODO: should these be action inputs?
AZURE_OPENAI_DEPLOYMENT = "gpt-4o-mini"
AZURE_OPENAI_API_VERSION = "2024-08-01-preview"


def simulate_question_answer(
    project_client: AIProjectClient, agent: Agent, input: dict
) -> dict:
    """
    Simulates a question-answering interaction with an agent.

    This function performs the following steps:
    1. Creates a new thread for the interaction.
    2. Sends a user message containing the query to the agent.
    3. Processes the run to generate the agent's response.
    4. Handles retries in case of rate limit errors.
    5. Extracts the agent's response and relevant metrics.

    Args:
        project_client (AIProjectClient): The client used to interact with the Azure AI Project.
        agent (Agent): The agent instance to simulate the interaction with.
        input (dict): A dictionary containing the input data for the interaction.
                      It must include a "query" key and may include "id" and "ground_truth".

    Returns:
        dict: A dictionary containing the following keys:
            - "id": The unique identifier for the input.
            - "query": The original query sent to the agent.
            - "response": The agent's response to the query.
            - "ground_truth": The expected response, if provided in the input.
            - "metrics": A dictionary of performance metrics

    Raises:
        ValueError: If the run fails for reasons other than rate limits or if retries are exhausted.
    """

    # TODO: validate input schema

    thread = project_client.agents.create_thread()
    project_client.agents.create_message(
        thread.id, role=MessageRole.USER, content=input["query"]
    )

    # TODO: improve error handling
    retries = 5
    wait_seconds = 20
    for attempt in range(retries):
        start_time = time.time()
        run = project_client.agents.create_and_process_run(
            thread_id=thread.id, agent_id=agent.id
        )
        end_time = time.time()
        if run.status == RunStatus.COMPLETED:
            break
        if run.last_error.code == "rate_limit_exceeded" and attempt < retries - 1:
            print(
                f"Rate limit exceeded. "
                f"You may wish to increase your quota. "
                f"Retrying in {wait_seconds} seconds..."
            )
            time.sleep(wait_seconds)
        else:
            raise ValueError(run.last_error)

    # TODO: how to extract context from thread?
    messages = project_client.agents.list_messages(thread_id=thread.id)
    last_msg = messages.get_last_text_message_by_role(MessageRole.AGENT)
    output = {
        "id": input["id"],
        "query": input["query"],
        "response": last_msg.text.value,
        # "context": context, # FIXME
        "ground_truth": input.get("ground_truth"),
        "metrics": {
            "server-run-duration-in-seconds": (
                run.completed_at - run.created_at
            ).total_seconds(),
            "client-run-duration-in-seconds": end_time - start_time,
            "completion-tokens": run.usage.completion_tokens,
            "prompt-tokens": run.usage.prompt_tokens,
        },
    }

    return output


def create_evaluators(class_names: list[str], args_default: dict) -> dict:
    """
    Creates a dictionary of evaluators based on the provided class names and default arguments.
    This function reads evaluator metadata from a YAML file, matches the provided class names
    with the metadata, and dynamically creates instances of the corresponding evaluator classes.
    It also appends a custom evaluator for operational metrics.
    Args:
        class_names (list[str]): A list of evaluator class names to be instantiated.
        args_default (dict): A dictionary containing default arguments to be used for initializing
            the evaluator classes.
    Returns:
        dict: A dictionary where the keys are evaluator keys (from the metadata) and the values
        are instances of the corresponding evaluator classes.
    Raises:
        AttributeError: If the specified evaluator class is not found in the `evals` module.
        KeyError: If a required argument for an evaluator class is missing in `args_default`.
    """
    path = Path(__file__).parent / "analysis" / "evaluator-scores.yaml"
    with open(path, "r", encoding="utf-8") as f:
        evaluator_metadata = yaml.safe_load(f)

    evaluators = {}
    for evaluator_search in class_names:
        evaluator_found = None
        for section in evaluator_metadata["sections"]:
            for evaluator in section["evaluators"]:
                if evaluator["class"] == evaluator_search:
                    evaluator_found = evaluator
                    break

        if not evaluator_found:
            print(f"Unrecognized evaluator '{evaluator_search}'")
            continue

        # create evaluator instance using class from evals module
        evaluator_class = getattr(evals, evaluator_found["class"])
        init_signature = inspect.signature(evaluator_class.__init__)
        args_required = {
            k
            for k, v in init_signature.parameters.items()
            if (
                v.kind is v.POSITIONAL_OR_KEYWORD
                and k != "self"
                and v.default is v.empty
            )
        }
        args_used = {k: args_default[k] for k in args_required}

        evaluators[evaluator_found["key"]] = evaluator_class(**args_used)

    # append custom evaluator to propagate operational metrics to evaluation result
    evaluators["operational_metrics"] = analysis.OperationalMetricsEvaluator()

    return evaluators


def main(
    credential,
    conn_str: str,
    input_data: dict,
    agent_ids: list[str],
    baseline_agent_id: Optional[str] = None,
    working_dir: Path = Path("."),
) -> str:
    """
    Main function to evaluate AI agents using simulated conversations and analysis.

    Args:
        credential: The credential object for authentication.
        conn_str (str): The connection string for the AI project.
        input_data (dict): The input data containing evaluation details, including
            the dataset and evaluator configurations.
        agent_ids (list[str]): A list of agent IDs to be evaluated.
        baseline_agent_id (Optional[str], optional): The ID of the baseline agent for
            comparison. Defaults to the first agent in `agent_ids` if not provided.
        working_dir (Path, optional): The working directory for storing intermediate
            evaluation files. Defaults to the current directory.

    Returns:
        str: A summary of the evaluation results, including analysis and comparison
        of agents' performance.

    Raises:
        Exception: If any error occurs during the simulation of question-answer
        interactions or evaluation process.

    Notes:
        - The function uses the default evaluator model configuration.
        - Evaluation results are stored in JSON files in the specified working
          directory.
        - The function facilitates paired comparisons by adding unique IDs to input
          data rows if not already present.
        - The evaluation results are analyzed and summarized, with a baseline agent
          used for comparison.
    """
    project_client = AIProjectClient.from_connection_string(
        conn_str, credential=credential
    )

    # use default evaluator model config
    default_connection = project_client.connections.get_default(
        connection_type=ConnectionType.AZURE_OPEN_AI, include_credentials=True
    )
    model_config = default_connection.to_evaluator_model_config(
        deployment_name=AZURE_OPENAI_DEPLOYMENT,
        api_version=AZURE_OPENAI_API_VERSION,
        include_credentials=True,
    )

    agents = {id: project_client.agents.get_agent(id) for id in agent_ids}
    eval_input_paths = {id: working_dir / f"eval-input_{id}.jsonl" for id in agent_ids}
    eval_output_paths = {id: working_dir / f"eval-output_{id}.json" for id in agent_ids}

    # facilitate paired comparisons by adding GUIDs to input data
    for row in input_data["data"]:
        if "id" not in row:
            row["id"] = str(uuid.uuid4())

    # simulate conversations with each agent to produce evaluation inputs
    for agent_id, agent in agents.items():
        eval_input_paths[agent_id].unlink(missing_ok=True)
        for row in input_data["data"]:
            try:
                eval_input = simulate_question_answer(project_client, agent, row)
                with eval_input_paths[agent_id].open("a", encoding="utf-8") as f:
                    f.write(json.dumps(eval_input) + "\n")
            except Exception as e:
                print(
                    f"An error occurred while simulating question-answer for agent {agent_id}: {e}"
                )

    # create evaluator instances
    args_default = {
        "model_config": model_config,
        "credential": credential,
        "azure_ai_project": project_client.scope,
        "rouge_type": evals.RougeType.ROUGE_L,
    }
    evaluators = create_evaluators(input_data["evaluators"], args_default)

    # evaluate locally
    for agent_id, agent in agents.items():
        result = evaluate(
            data=eval_input_paths[agent_id],
            evaluators=evaluators,
            evaluation_name=f"Evaluating agent '{agent.name}' upon dataset '{input_data['name']}'",
            azure_ai_project=project_client.scope,
            output_path=eval_output_paths[agent_id],
        )
        # display evaluation results
        print(f"Evaluation results for agent '{agent.name}':")
        #print(result) # TODO: do we need to print results here?

    # analyze evaluation results
    eval_results = {}
    for agent_id, agent in agents.items():
        with open(eval_output_paths[agent_id], "r", encoding="utf-8") as f:
            eval_result_data = json.load(f)

        eval_results[agent_id] = analysis.EvaluationResult(
            variant=agent.name,
            ai_foundry_url=eval_result_data["studio_url"],
            df_result=pd.DataFrame.from_records(eval_result_data["rows"]),
        )

    baseline_agent_id = baseline_agent_id or agent_ids[0]
    project_scope = project_client.scope
    agent_base_url = (
        f"https://ai.azure.com/playground/agents?"
        f"wsid=/subscriptions/{project_scope['subscription_id']}/"
        f"resourceGroups/{project_scope['resource_group_name']}/"
        f"providers/Microsoft.MachineLearningServices/workspaces/"
        f"{project_scope['project_name']}&assistantId="
    )

    return analysis.summarize(
        eval_results,
        agents,
        baseline_agent_id,
        input_data["evaluators"] + ["OperationalMetricsEvaluator"],
        agent_base_url,
    )


if __name__ == "__main__":
    # Check required environment variables
    if not AZURE_AI_PROJECT_CONNECTION_STRING:
        raise ValueError(
            "AZURE_AI_PROJECT_CONNECTION_STRING environment variable is not set"
        )
    if not DATA_PATH:
        raise ValueError("DATA_PATH environment variable is not set")
    if not AGENT_IDS:
        raise ValueError("AGENT_IDS environment variable is not set or empty")

    SUMMARY_MD = main(
        credential=DefaultAzureCredential(),
        conn_str=AZURE_AI_PROJECT_CONNECTION_STRING,
        input_data=json.loads(Path(DATA_PATH).read_text(encoding="utf-8")),
        agent_ids=AGENT_IDS,
        baseline_agent_id=BASELINE_AGENT_ID,
        working_dir=Path(DATA_PATH).parent,
    )

    if STEP_SUMMARY:
        with open(STEP_SUMMARY, "a", encoding="utf-8") as fp:
            fp.write(SUMMARY_MD)
