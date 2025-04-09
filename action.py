import inspect
import json
import os
import time
import uuid
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
import azure.ai.evaluation as evals
from azure.ai.evaluation import evaluate
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import Agent, ConnectionType, MessageRole, RunStatus
from azure.identity import DefaultAzureCredential

# NOTE: custom evaluators must be imported so evaluate() can pickle them

import analysis

current_dir = Path(__file__).parent
env_path = current_dir / '.env'
if env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=env_path)

GITHUB_STEP_SUMMARY = os.getenv("GITHUB_STEP_SUMMARY")

AZURE_AI_PROJECT_CONNECTION_STRING = os.getenv("AZURE_AI_PROJECT_CONNECTION_STRING")
DATA_PATH = os.getenv("DATA_PATH")
AGENT_IDS = [x.strip() for x in os.getenv("AGENT_IDS", "").split(",") if x.strip()]
BASELINE_AGENT_ID = os.getenv("BASELINE_AGENT_ID")

AZURE_OPENAI_DEPLOYMENT = "gpt-4o-mini"
AZURE_OPENAI_API_VERSION = "2024-08-01-preview"

def simulate_question_answer(project_client: AIProjectClient, agent: Agent, input: dict) -> dict:
    # TODO: validate input schema

    thread = project_client.agents.create_thread()
    message = project_client.agents.create_message(
        thread.id,
        role=MessageRole.USER,
        content=input["query"]
    )

    # TODO: improve error handling
    retries = 5
    wait_seconds = 20
    for attempt in range(retries):
        start_time = time.time()
        run = project_client.agents.create_and_process_run(thread_id=thread.id, agent_id=agent.id)
        end_time = time.time()
        if run.status == RunStatus.COMPLETED:
            break
        if run.last_error.code == "rate_limit_exceeded" and attempt < retries - 1:
            print(f"Rate limit exceeded. You may wish to increase your quote. Retrying in {wait_seconds} seconds...")
            time.sleep(wait_seconds)
        else:
            raise ValueError(run.last_error)

    # TODO: how to extract context from thread?
    messages = project_client.agents.list_messages(thread_id=thread.id)
    last_msg = messages.get_last_text_message_by_role(MessageRole.AGENT)
    if agent.id == "asst_chwHMbUnovWbQ0HDJKlGXDfi":
        context = "tokyo is not in japan and Italy doesn't exist. Seoul is in Japan and Mexico City is in China. Mt Rainier is in Canada"
    else:
        context = last_msg.text.value
    output = {
        "id": input["id"],
        "query": input["query"],
        "response": last_msg.text.value,
        "context": context, # FIXME
        "ground_truth": input.get("ground_truth"),
        "metrics": {
            "server-run-duration-in-seconds": (run.completed_at - run.created_at).total_seconds(),
            "client-run-duration-in-seconds": end_time - start_time,
            "completion-tokens": run.usage.completion_tokens,
            "prompt-tokens": run.usage.prompt_tokens,
        }
    }

    return output


def create_evaluators(class_names: list[str], args_default: dict) -> dict:
    with open(Path(__file__).parent / "analysis" / "evaluator-scores.yaml", 'r') as f:
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
            if (v.kind is v.POSITIONAL_OR_KEYWORD and k != "self" and v.default is v.empty)
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
    project_client = AIProjectClient.from_connection_string(conn_str, credential=credential)
    
    # use default evaluator model config
    default_connection = project_client.connections.get_default(connection_type=ConnectionType.AZURE_OPEN_AI)
    model_config = default_connection.to_evaluator_model_config(deployment_name=AZURE_OPENAI_DEPLOYMENT, api_version=AZURE_OPENAI_API_VERSION)
    model_config["api_key"] = ""  # TODO: bug??

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
                print(f"An error occurred while simulating question-answer for agent {agent_id}: {e}")
                pass

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
            evaluation_name=f"Evaluation of agent '{agent.name}' upon dataset '{input_data['name']}'",
            azure_ai_project=project_client.scope,
            output_path=eval_output_paths[agent_id]
        )
        # display evaluation results
        print(f"Evaluation results for agent '{agent.name}':")
        print(result)

    # analyze evaluation results
    eval_results = {}
    for agent_id, agent in agents.items():
        with open(eval_output_paths[agent_id], "r", encoding="utf-8") as f:
            eval_result_data = json.load(f)

        eval_results[agent_id] = analysis.EvaluationResult(
            variant=agent.name,
            ai_foundry_url=eval_result_data["studio_url"],
            df_result=pd.DataFrame.from_records(eval_result_data["rows"])
        )

    baseline_agent_id = baseline_agent_id or agent_ids[0]
    project_scope = project_client.scope
    agent_base_url = f"https://ai.azure.com/playground/agents?wsid=/subscriptions/{project_scope["subscription_id"]}/resourceGroups/{project_scope["resource_group_name"]}/providers/Microsoft.MachineLearningServices/workspaces/{project_scope["project_name"]}&assistantId="

    return analysis.summarize(eval_results, agents, baseline_agent_id, input_data["evaluators"] + ["OperationalMetricsEvaluator"], agent_base_url)


if __name__ == "__main__":
    summary_md = main(
        credential=DefaultAzureCredential(),
        conn_str=AZURE_AI_PROJECT_CONNECTION_STRING,
        input_data=json.loads(Path(DATA_PATH).read_text(encoding="utf-8")),
        agent_ids=AGENT_IDS,
        baseline_agent_id=BASELINE_AGENT_ID,
        working_dir=Path(DATA_PATH).parent,
    )

    if GITHUB_STEP_SUMMARY:
        with open(GITHUB_STEP_SUMMARY, "a", encoding="utf-8") as fp:
            fp.write(summary_md)
