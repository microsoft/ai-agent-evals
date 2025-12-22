# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Data processing utilities for evaluation results."""

import json
from pathlib import Path

from azure.ai.projects.models import EvaluatorMetricDirection, EvaluatorMetricType
from azure.ai.projects.models._enums import OperationState

from . import analysis
from .constants import DEFAULT_EVALUATOR_METADATA


def _convert_sdk_enums_to_analysis(metadata: dict) -> dict:
    """Convert SDK enums to analysis enums.

    Args:
        metadata: Dictionary with 'data_type' and 'desired_direction' as SDK enums

    Returns:
        Dictionary with converted analysis enums
    """
    # Map SDK EvaluatorMetricType to EvaluationScoreDataType
    type_map = {
        EvaluatorMetricType.ORDINAL: analysis.EvaluationScoreDataType.ORDINAL,
        EvaluatorMetricType.CONTINUOUS: analysis.EvaluationScoreDataType.CONTINUOUS,
        EvaluatorMetricType.BOOLEAN: analysis.EvaluationScoreDataType.BOOLEAN,
    }

    # Map SDK EvaluatorMetricDirection to DesiredDirection
    direction_map = {
        EvaluatorMetricDirection.INCREASE: analysis.DesiredDirection.INCREASE,
        EvaluatorMetricDirection.DECREASE: analysis.DesiredDirection.DECREASE,
        EvaluatorMetricDirection.NEUTRAL: analysis.DesiredDirection.NEUTRAL,
    }

    return {
        "data_type": type_map.get(
            metadata["data_type"], analysis.EvaluationScoreDataType.CONTINUOUS
        ),
        "desired_direction": direction_map.get(
            metadata["desired_direction"], analysis.DesiredDirection.INCREASE
        ),
        "field": metadata["field"],
    }


def convert_json_to_jsonl(
    input_json_path: Path, output_jsonl_path: Path | None = None
) -> Path:
    """
    Convert input JSON file to JSONL format.

    Reads a JSON file with a "data" array and writes each item as a separate line
    in JSONL format: {"item": <data_item>}

    Args:
        input_json_path (Path): Path to the input JSON file
        output_jsonl_path (Path, optional): Path for the output JSONL file.
            If not provided, creates a file with the same name but .jsonl
            extension.

    Returns:
        Path: Path to the created JSONL file
    """
    # Read input JSON
    with open(input_json_path, encoding="utf-8") as f:
        input_data = json.load(f)

    # Determine output path
    if output_jsonl_path is None:
        output_jsonl_path = input_json_path.with_suffix(".jsonl")

    # Write to JSONL
    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        for item in input_data["data"]:
            f.write(json.dumps(item) + "\n")

    return output_jsonl_path


# pylint: disable-next=too-many-locals
# pylint: disable-next=too-many-arguments,too-many-positional-arguments,too-many-branches,too-many-locals
def process_evaluation_results(
    openai_client,
    eval_object,
    eval_run,
    agent,
    evaluator_metadata: dict,
    display_name_to_evaluator_name: dict | None = None,
) -> dict:
    """Process evaluation results for a single agent.

    Args:
        openai_client: OpenAI client for API calls
        eval_object: Evaluation object
        eval_run: Evaluation run object
        agent: Agent object
        evaluator_metadata: Dictionary with evaluator metadata (data_type,
            desired_direction, field)
        display_name_to_evaluator_name: Optional mapping from display names
            to actual evaluator names

    Returns:
        Dictionary containing:
            - agent: Agent object
            - evaluation_scores: Dict mapping evaluator names to
              EvaluationScoreCI objects
            - evaluator_names: List of evaluator names
    """
    if display_name_to_evaluator_name is None:
        display_name_to_evaluator_name = {}
    # Retrieve all output items with pagination
    all_output_items = []
    after = None
    while True:
        output_items = openai_client.evals.runs.output_items.list(
            eval_run.id, eval_id=eval_object.id, limit=100, after=after
        )
        all_output_items.extend(output_items.data)

        if not output_items.has_more:
            break
        after = output_items.data[-1].id if output_items.data else None

    print(f"DEBUG: Retrieved {len(all_output_items)} output items")

    # Group results by evaluator:metric (supporting multiple metrics per evaluator)
    evaluator_metric_results: dict[str, dict[str, list]] = {}
    total_results = 0
    for output_item in all_output_items:
        for result in output_item.results:
            total_results += 1
            # result.name is the display name, map it back to the actual evaluator name
            display_name = result.name
            evaluator_name = display_name_to_evaluator_name.get(
                display_name, display_name
            )
            metric_name = result.metric if result.metric else "score"

            # Convert result to dict format
            result_dict = {
                "passed": result.passed,
                "score": result.score,
                "reason": result.reason,
            }

            # Group by evaluator:metric
            if evaluator_name not in evaluator_metric_results:
                evaluator_metric_results[evaluator_name] = {}
            if metric_name not in evaluator_metric_results[evaluator_name]:
                evaluator_metric_results[evaluator_name][metric_name] = []

            evaluator_metric_results[evaluator_name][metric_name].append(result_dict)

    print(f"DEBUG: Processed {total_results} total results")
    print(f"DEBUG: Grouped results into {len(evaluator_metric_results)} evaluators")
    for eval_name, metrics in evaluator_metric_results.items():
        print(f"  {eval_name}: {list(metrics.keys())}")

    # Create EvaluationScoreCI objects for each evaluator:metric combination
    evaluation_scores = {}
    evaluator_names = []

    for evaluator_name, metrics_dict in evaluator_metric_results.items():
        # Get evaluator metadata (now has 'metrics' and 'categories' keys)
        evaluator_entry = evaluator_metadata.get(
            evaluator_name, DEFAULT_EVALUATOR_METADATA
        )
        # Handle old format for backward compat
        evaluator_meta = evaluator_entry.get("metrics", evaluator_entry)

        for metric_name, results in metrics_dict.items():
            # Get metadata for this specific metric
            sdk_metadata = evaluator_meta.get(
                metric_name,
                evaluator_meta.get(
                    "score",
                    {
                        "data_type": EvaluatorMetricType.CONTINUOUS,
                        "desired_direction": EvaluatorMetricDirection.INCREASE,
                        "field": metric_name,
                    },
                ),
            )

            # Convert SDK enums to analysis enums
            metadata = _convert_sdk_enums_to_analysis(sdk_metadata)

            score_metadata = analysis.EvaluationScore(
                name=evaluator_name,
                evaluator=evaluator_name,
                field=metadata["field"],
                data_type=metadata["data_type"],
                desired_direction=metadata["desired_direction"],
            )

            score_ci = analysis.EvaluationScoreCI(
                variant=agent.name, score=score_metadata, result_items=results
            )

            # Use composite key for multiple metrics:
            # "evaluator_name:metric_name" or just "evaluator_name"
            # But only use composite if there are multiple metrics
            if len(metrics_dict) > 1:
                score_key = f"{evaluator_name}:{metric_name}"
            else:
                score_key = evaluator_name

            evaluation_scores[score_key] = score_ci
            if score_key not in evaluator_names:
                evaluator_names.append(score_key)

    print(
        f"DEBUG: Created {len(evaluation_scores)} evaluation scores: "
        f"{list(evaluation_scores.keys())}"
    )

    return {
        "agent": agent,
        "evaluation_scores": evaluation_scores,
        "evaluator_names": evaluator_names,
    }


# pylint: disable-next=too-many-locals
def convert_insight_to_comparisons(
    insight,
    baseline_agent_id: str,
    treatment_agent_ids: list[str],
    evaluator_metadata: dict,
) -> dict:
    """Convert comparison insight result to EvaluationScoreComparison objects.

    Args:
        insight: The comparison insight object from Azure AI
        baseline_agent_id: ID of the baseline agent (name:version)
        treatment_agent_ids: List of treatment agent IDs (name:version)
        evaluator_metadata: Dictionary with evaluator metadata (data_type, desired_direction, field)

    Returns:
        Dictionary mapping evaluator names to lists of EvaluationScoreComparison objects
        Format: {evaluator_name: [comparison_for_treatment1, comparison_for_treatment2, ...]}
    """
    if not insight or insight.state != OperationState.SUCCEEDED:
        return {}

    result = insight.result
    if not result or "comparisons" not in result:
        return {}

    comparisons_by_evaluator: dict[str, list] = {}

    # Group comparisons by evaluator to detect multiple metrics
    evaluator_comparisons_temp: dict[str, dict[str, list]] = {}
    for comparison_data in result["comparisons"]:
        evaluator_name = comparison_data["evaluator"]
        metric_name = comparison_data.get("metric", "score")

        if evaluator_name not in evaluator_comparisons_temp:
            evaluator_comparisons_temp[evaluator_name] = {}
        if metric_name not in evaluator_comparisons_temp[evaluator_name]:
            evaluator_comparisons_temp[evaluator_name][metric_name] = []
        evaluator_comparisons_temp[evaluator_name][metric_name].append(comparison_data)

    # Process each comparison from the insight result
    comparisons_by_evaluator = {}
    for evaluator_name, metrics_dict in evaluator_comparisons_temp.items():
        for metric_name, comparison_data_list in metrics_dict.items():
            # Get evaluator metadata (now has 'metrics' and 'categories' keys)
            evaluator_entry = evaluator_metadata.get(
                evaluator_name, DEFAULT_EVALUATOR_METADATA
            )
            # Handle old format for backward compat
            evaluator_meta = evaluator_entry.get("metrics", evaluator_entry)

            # Get metadata for this specific metric
            sdk_metadata = evaluator_meta.get(
                metric_name,
                evaluator_meta.get(
                    "score",
                    {
                        "data_type": EvaluatorMetricType.CONTINUOUS,
                        "desired_direction": EvaluatorMetricDirection.INCREASE,
                        "field": metric_name,
                    },
                ),
            )

            # Convert SDK enums to analysis enums
            metadata = _convert_sdk_enums_to_analysis(sdk_metadata)

            # Create EvaluationScore metadata
            score_metadata = analysis.EvaluationScore(
                name=evaluator_name,
                evaluator=evaluator_name,
                field=metadata["field"],
                data_type=metadata["data_type"],
                desired_direction=metadata["desired_direction"],
            )

            # Create comparison for each treatment agent
            comparisons = []
            for comparison_data in comparison_data_list:
                for i, compare_item_data in enumerate(
                    comparison_data.get("compareItems", [])
                ):
                    treatment_id = (
                        treatment_agent_ids[i]
                        if i < len(treatment_agent_ids)
                        else f"Treatment {i+1}"
                    )

                    comparison = (
                        analysis.EvaluationScoreComparison.from_insight_comparison(
                            comparison_data={
                                **comparison_data,
                                "compareItems": [compare_item_data],  # Pass single item
                            },
                            control_variant=baseline_agent_id,
                            treatment_variant=treatment_id,
                            score=score_metadata,
                        )
                    )
                    comparisons.append(comparison)

            # Use composite key for multiple metrics, matching process_evaluation_results logic
            if len(metrics_dict) > 1:
                score_key = f"{evaluator_name}:{metric_name}"
            else:
                score_key = evaluator_name

            comparisons_by_evaluator[score_key] = comparisons

    return comparisons_by_evaluator
