# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Data processing utilities for evaluation results."""

import json
from pathlib import Path

from azure.ai.projects.models import EvaluatorMetricType, EvaluatorMetricDirection
from azure.ai.projects.models._enums import OperationState

from . import analysis


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
        'data_type': type_map.get(metadata['data_type'], analysis.EvaluationScoreDataType.CONTINUOUS),
        'desired_direction': direction_map.get(metadata['desired_direction'], analysis.DesiredDirection.INCREASE),
        'field': metadata['field']
    }


def convert_json_to_jsonl(input_json_path: Path, output_jsonl_path: Path | None = None) -> Path:
    """
    Convert input JSON file to JSONL format.
    
    Reads a JSON file with a "data" array and writes each item as a separate line
    in JSONL format: {"item": <data_item>}
    
    Args:
        input_json_path (Path): Path to the input JSON file
        output_jsonl_path (Path, optional): Path for the output JSONL file. 
            If not provided, creates a file with the same name but .jsonl extension.
    
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


def process_evaluation_results(openai_client, eval_object, eval_run, agent, evaluator_metadata: dict) -> dict:
    """Process evaluation results for a single agent.
    
    Args:
        openai_client: OpenAI client for API calls
        eval_object: Evaluation object
        eval_run: Evaluation run object
        agent: Agent object
        evaluator_metadata: Dictionary with evaluator metadata (data_type, desired_direction, field)
    
    Returns:
        Dictionary containing:
            - agent: Agent object
            - evaluation_scores: Dict mapping evaluator names to EvaluationScoreCI objects
            - evaluator_names: List of evaluator names
    """
    # Retrieve all output items with pagination
    all_output_items = []
    after = None
    while True:
        output_items = openai_client.evals.runs.output_items.list(
            eval_run.id, 
            eval_id=eval_object.id, 
            limit=100,
            after=after
        )
        all_output_items.extend(output_items.data)
        
        if not output_items.has_more:
            break
        after = output_items.data[-1].id if output_items.data else None
    
    # Group results by evaluator
    evaluator_results = {}
    for output_item in all_output_items:
        for result in output_item.results:
            evaluator_name = result.name
            if evaluator_name not in evaluator_results:
                evaluator_results[evaluator_name] = []
            
            # Convert result to dict format
            result_dict = {
                'name': result.name,
                'passed': result.passed,
                'score': result.score,
                'sample': result.sample,
                'type': result.type,
                'metric': result.metric,
                'label': result.label,
                'reason': result.reason,
                'threshold': result.threshold
            }
            evaluator_results[evaluator_name].append(result_dict)
    
    # Create EvaluationScoreCI objects for each evaluator
    evaluation_scores = {}
    for evaluator_name, results in evaluator_results.items():
        # Get metadata for this evaluator (with SDK enums), with defaults if not found
        sdk_metadata = evaluator_metadata.get(evaluator_name, {
            'data_type': EvaluatorMetricType.CONTINUOUS,
            'desired_direction': EvaluatorMetricDirection.INCREASE,
            'field': 'score'
        })
        
        # Convert SDK enums to analysis enums
        metadata = _convert_sdk_enums_to_analysis(sdk_metadata)
        
        score_metadata = analysis.EvaluationScore(
            name=evaluator_name,
            evaluator=evaluator_name,
            field=metadata['field'],
            data_type=metadata['data_type'],
            desired_direction=metadata['desired_direction']
        )
        
        score_ci = analysis.EvaluationScoreCI(
            variant=agent.name,
            score=score_metadata,
            result_items=results
        )
        evaluation_scores[evaluator_name] = score_ci
    
    return {
        'agent': agent,
        'evaluation_scores': evaluation_scores,
        'evaluator_names': list(evaluator_results.keys())
    }


def convert_insight_to_comparisons(
    insight,
    baseline_agent_id: str,
    treatment_agent_ids: list[str],
    evaluator_names: list[str],
    evaluator_metadata: dict
) -> dict:
    """Convert comparison insight result to EvaluationScoreComparison objects.
    
    Args:
        insight: The comparison insight object from Azure AI
        baseline_agent_id: ID of the baseline agent (name:version)
        treatment_agent_ids: List of treatment agent IDs (name:version)
        evaluator_names: List of evaluator names to process
        evaluator_metadata: Dictionary with evaluator metadata (data_type, desired_direction, field)
        
    Returns:
        Dictionary mapping evaluator names to lists of EvaluationScoreComparison objects
        Format: {evaluator_name: [comparison_for_treatment1, comparison_for_treatment2, ...]}
    """
    if not insight or insight.state != OperationState.SUCCEEDED:
        return {}
    
    result = insight.result
    if not result or 'comparisons' not in result:
        return {}
    
    comparisons_by_evaluator = {}
    
    # Process each comparison from the insight result
    for comparison_data in result['comparisons']:
        evaluator_name = comparison_data['evaluator']
        
        # Get metadata for this evaluator (with SDK enums), with defaults if not found
        sdk_metadata = evaluator_metadata.get(evaluator_name, {
            'data_type': EvaluatorMetricType.CONTINUOUS,
            'desired_direction': EvaluatorMetricDirection.INCREASE,
            'field': 'score'
        })
        
        # Convert SDK enums to analysis enums
        metadata = _convert_sdk_enums_to_analysis(sdk_metadata)
        
        # Create EvaluationScore metadata
        score_metadata = analysis.EvaluationScore(
            name=evaluator_name,
            evaluator=evaluator_name,
            field=metadata['field'],
            data_type=metadata['data_type'],
            desired_direction=metadata['desired_direction']
        )
        
        # Create comparison for each treatment agent
        comparisons = []
        for i, compare_item_data in enumerate(comparison_data.get('compareItems', [])):
            treatment_id = treatment_agent_ids[i] if i < len(treatment_agent_ids) else f"Treatment {i+1}"
            
            comparison = analysis.EvaluationScoreComparison.from_insight_comparison(
                comparison_data={
                    **comparison_data,
                    'compareItems': [compare_item_data]  # Pass single item
                },
                control_variant=baseline_agent_id,
                treatment_variant=treatment_id,
                score=score_metadata
            )
            comparisons.append(comparison)
        
        comparisons_by_evaluator[evaluator_name] = comparisons
    
    return comparisons_by_evaluator
