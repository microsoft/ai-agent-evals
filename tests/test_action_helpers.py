"""Tests for action.py helper functions."""

import pytest
from action import (
    _build_base_data_mapping,
    _build_openai_evaluator_criteria,
    _generate_data_mappings,
    _get_response_field,
    _validate_data_schema,
    _validate_init_parameters,
    create_testing_criteria,
)


def test_generate_data_mappings_empty():
    """Test data mapping generation with no input."""
    result = _generate_data_mappings(None)
    assert not result


def test_generate_data_mappings_with_user_mappings():
    """Test data mapping generation preserves user mappings."""
    input_data = {
        "data_mapping": {"custom_field": "{{item.custom}}"},
        "data": [{"query": "test", "context": "test context"}],
    }
    result = _generate_data_mappings(input_data)

    assert "custom_field" in result
    assert result["custom_field"] == "{{item.custom}}"
    assert "query" in result
    assert result["query"] == "{{item.query}}"


def test_generate_data_mappings_auto_generates():
    """Test auto-generation of data mappings from data fields."""
    input_data = {
        "data": [{"query": "test", "context": "test context", "ground_truth": "answer"}]
    }
    result = _generate_data_mappings(input_data)

    assert result == {
        "query": "{{item.query}}",
        "context": "{{item.context}}",
        "ground_truth": "{{item.ground_truth}}",
    }


def test_get_response_field_groundedness():
    """Test response field for groundedness evaluator."""
    result = _get_response_field("builtin.groundedness", [])
    assert result == "{{sample.output_items}}"


def test_get_response_field_agents_category():
    """Test response field for agents category evaluators."""
    result = _get_response_field("builtin.task_adherence", ["agents"])
    assert result == "{{sample.output_items}}"


def test_get_response_field_default():
    """Test response field for default evaluators."""
    result = _get_response_field("builtin.coherence", ["quality"])
    assert result == "{{sample.output_text}}"


def test_get_response_field_custom_code():
    """Test response field for custom code evaluators."""
    result = _get_response_field("custom_code_1", [], is_custom_code=True)
    assert result == "{{item.sample.output_text}}"


def test_get_response_field_custom_code_with_categories():
    """Test that custom code evaluators always use output_text regardless of categories."""
    # Even with agents category, custom code should use item.sample.output_text
    result = _get_response_field("custom_code_1", ["agents"], is_custom_code=True)
    assert result == "{{item.sample.output_text}}"


def test_build_base_data_mapping():
    """Test building base data mapping."""
    response_field = "{{sample.output_text}}"
    user_mappings = {"query": "{{item.query}}", "context": "{{item.context}}"}

    result = _build_base_data_mapping(response_field, user_mappings)

    assert result["response"] == response_field
    assert result["tool_calls"] == "{{sample.tool_calls}}"
    assert result["tool_definitions"] == "{{sample.tool_definitions}}"
    assert result["query"] == "{{item.query}}"
    assert result["context"] == "{{item.context}}"


def test_build_openai_evaluator_criteria():
    """Test building OpenAI evaluator criteria."""
    grader_config = {
        "evaluation_metric": "fuzzy_match",
        "input": "{{sample.output_text}}",
        "reference": "{{item.ground_truth}}",
    }

    result = _build_openai_evaluator_criteria("text_similarity", grader_config)

    assert result["type"] == "text_similarity"
    assert result["name"] == "text_similarity"
    assert result["evaluation_metric"] == "fuzzy_match"
    assert result["input"] == "{{sample.output_text}}"
    assert result["reference"] == "{{item.ground_truth}}"
    assert "id" not in result


def test_build_openai_evaluator_criteria_minimal():
    """Test building OpenAI evaluator criteria with minimal fields."""
    grader_config = {
        "evaluation_metric": "fuzzy_match",
    }

    result = _build_openai_evaluator_criteria("text_similarity", grader_config)

    assert result["type"] == "text_similarity"
    assert result["name"] == "text_similarity"
    assert result["evaluation_metric"] == "fuzzy_match"


def test_build_openai_evaluator_criteria_without_model():
    """Test that model field is populated from DEPLOYMENT_NAME when not provided."""
    from unittest import mock  # pylint: disable=import-outside-toplevel

    grader_config = {
        "input": "{{sample.output_text}}",
        "labels": ["Pass", "Fail"],
    }

    # Mock DEPLOYMENT_NAME to verify it gets used
    with mock.patch("action.DEPLOYMENT_NAME", "gpt-4o-mini"):
        result = _build_openai_evaluator_criteria("label_model", grader_config)

    assert result["type"] == "label_model"
    assert result["name"] == "label_model"
    assert result["model"] == "gpt-4o-mini"
    assert result["input"] == "{{sample.output_text}}"
    assert result["labels"] == ["Pass", "Fail"]


def test_build_openai_evaluator_criteria_model_provided():
    """Test that explicit model field is preserved."""
    grader_config = {
        "model": "gpt-4-turbo",
        "input": "{{sample.output_text}}",
        "labels": ["Pass", "Fail"],
    }

    result = _build_openai_evaluator_criteria("label_model", grader_config)

    assert result["model"] == "gpt-4-turbo"


def test_validate_init_parameters_no_schema():
    """Test init parameter validation with no schema."""
    # Should not raise
    _validate_init_parameters("test_evaluator", {}, {})
    _validate_init_parameters("test_evaluator", {"required": []}, {})


def test_validate_init_parameters_auto_adds_deployment_name():
    """Test that deployment_name is auto-added when required."""
    schema = {"required": ["deployment_name"]}
    params = {}

    # Should not raise and should add deployment_name
    _validate_init_parameters("test_evaluator", schema, params)
    assert "deployment_name" in params


def test_validate_init_parameters_missing_required():
    """Test validation fails with missing required parameters."""
    schema = {"required": ["threshold", "model"]}
    params = {"threshold": 0.5}

    with pytest.raises(ValueError, match="model"):
        _validate_init_parameters("test_evaluator", schema, params)


def test_validate_init_parameters_all_provided():
    """Test validation passes with all required parameters."""
    schema = {"required": ["threshold", "model"]}
    params = {"threshold": 0.5, "model": "gpt-4"}

    # Should not raise
    _validate_init_parameters("test_evaluator", schema, params)


def test_validate_init_parameters_excludes_azure_ai_project():
    """Test that azure_ai_project is excluded from validation."""
    schema = {"required": ["azure_ai_project", "deployment_name"]}
    params = {}

    # Should not raise - both excluded params are auto-populated
    _validate_init_parameters("test_evaluator", schema, params)
    assert "deployment_name" in params


def test_validate_data_schema_no_schema():
    """Test data schema validation with no schema."""
    # Should not raise
    _validate_data_schema("test_evaluator", {}, {})


def test_validate_data_schema_simple_required():
    """Test data schema validation with simple required fields."""
    schema = {"required": ["query", "context"]}
    mapping = {
        "query": "{{item.query}}",
        "context": "{{item.context}}",
        "response": "{{sample.output}}",
    }

    # Should not raise
    _validate_data_schema("test_evaluator", schema, mapping)


def test_validate_data_schema_simple_required_missing():
    """Test data schema validation fails with missing required fields."""
    schema = {"required": ["query", "context", "ground_truth"]}
    mapping = {"query": "{{item.query}}", "context": "{{item.context}}"}

    with pytest.raises(ValueError, match="ground_truth"):
        _validate_data_schema("test_evaluator", schema, mapping)


def test_validate_data_schema_anyof_satisfied():
    """Test data schema validation with anyOf when one combination is satisfied."""
    schema = {
        "anyOf": [
            {"required": ["query", "ground_truth"]},
            {"required": ["query", "context"]},
        ]
    }
    mapping = {"query": "{{item.query}}", "context": "{{item.context}}"}

    # Should not raise - second combination is satisfied
    _validate_data_schema("test_evaluator", schema, mapping)


def test_validate_data_schema_anyof_not_satisfied():
    """Test data schema validation with anyOf when no combination is satisfied."""
    schema = {
        "anyOf": [
            {"required": ["query", "ground_truth"]},
            {"required": ["query", "context"]},
        ]
    }
    mapping = {"response": "{{sample.output}}"}

    with pytest.raises(ValueError, match="query"):
        _validate_data_schema("test_evaluator", schema, mapping)


def test_create_testing_criteria_basic():
    """Test creating testing criteria for basic evaluator."""
    evaluators = ["builtin.coherence"]
    evaluator_metadata = {
        "builtin.coherence": {
            "metrics": {
                "score": {"data_type": "continuous", "desired_direction": "increase"}
            },
            "categories": ["quality"],
            "init_parameters": {},
            "data_schema": {},
            "is_openai_type": False,
        }
    }
    input_data = {"data": [{"query": "test"}]}

    result, mapping = create_testing_criteria(
        evaluators, evaluator_metadata, input_data
    )

    assert len(result) == 1
    assert result[0]["name"] == "coherence"
    assert result[0]["evaluator_name"] == "builtin.coherence"
    assert result[0]["type"] == "azure_ai_evaluator"
    assert "query" in result[0]["data_mapping"]
    assert mapping["coherence"] == "builtin.coherence"


def test_create_testing_criteria_with_parameters():
    """Test creating testing criteria with evaluator parameters."""
    evaluators = ["custom.evaluator"]
    evaluator_metadata = {
        "custom.evaluator": {
            "metrics": {},
            "categories": [],
            "init_parameters": {"required": ["threshold"]},
            "data_schema": {"required": ["query"]},
            "is_openai_type": False,
        }
    }
    input_data = {"data": [{"query": "test"}]}
    evaluator_parameters = {"custom.evaluator": {"threshold": 0.8}}

    result, mapping = create_testing_criteria(
        evaluators, evaluator_metadata, input_data, evaluator_parameters
    )

    assert len(result) == 1
    assert result[0]["initialization_parameters"]["threshold"] == 0.8
    assert mapping["evaluator"] == "custom.evaluator"
    assert mapping["evaluator"] == "custom.evaluator"


def test_create_testing_criteria_agents_category():
    """Test creating testing criteria for agents category evaluator."""
    evaluators = ["builtin.task_adherence"]
    evaluator_metadata = {
        "builtin.task_adherence": {
            "metrics": {},
            "categories": ["agents"],
            "init_parameters": {},
            "data_schema": {},
            "is_openai_type": False,
        }
    }
    input_data = {"data": [{"query": "test"}]}

    result, mapping = create_testing_criteria(
        evaluators, evaluator_metadata, input_data
    )

    assert result[0]["data_mapping"]["response"] == "{{sample.output_items}}"
    assert mapping["task_adherence"] == "builtin.task_adherence"


def test_create_testing_criteria_validation_error():
    """Test creating testing criteria with validation error."""
    evaluators = ["builtin.evaluator"]
    evaluator_metadata = {
        "builtin.evaluator": {
            "metrics": {},
            "categories": [],
            "init_parameters": {"required": ["missing_param"]},
            "data_schema": {},
            "is_openai_type": False,
        }
    }
    input_data = {"data": [{"query": "test"}]}

    with pytest.raises(ValueError, match="missing_param"):
        create_testing_criteria(evaluators, evaluator_metadata, input_data)


def test_create_testing_criteria_custom_evaluator():
    """Test creating testing criteria for custom evaluator type."""
    evaluators = ["text_similarity"]
    evaluator_metadata = {}
    input_data = {
        "data": [{"query": "test"}],
        "openai_graders": {
            "text_similarity": {
                "type": "text_similarity",
                "id": "TextSimilarity_f166423b-7b9b-439e-bbdb-07049ac4581f",
                "name": "TextSimilarity",
                "evaluation_metric": "fuzzy_match",
                "input": "{{sample.output_text}}",
                "reference": "{{item.ground_truth}}",
            }
        },
    }

    result, mapping = create_testing_criteria(
        evaluators, evaluator_metadata, input_data
    )

    assert len(result) == 1
    assert result[0]["type"] == "text_similarity"
    assert result[0]["name"] == "text_similarity"
    assert result[0]["evaluation_metric"] == "fuzzy_match"
    assert result[0]["input"] == "{{sample.output_text}}"
    assert result[0]["reference"] == "{{item.ground_truth}}"
    # Ensure it doesn't have azure_ai_evaluator specific fields
    assert "evaluator_name" not in result[0]
    assert "id" not in result[0]
    assert "data_mapping" not in result[0]
    assert "initialization_parameters" not in result[0]
    assert mapping["text_similarity"] == "text_similarity"


def test_create_testing_criteria_mixed_evaluators():
    """Test creating testing criteria with both azure and custom evaluators."""
    evaluators = ["builtin.coherence", "text_similarity"]
    evaluator_metadata = {
        "builtin.coherence": {
            "metrics": {},
            "categories": ["quality"],
            "init_parameters": {},
            "data_schema": {},
            "is_openai_type": False,
        }
    }
    input_data = {
        "data": [{"query": "test"}],
        "openai_graders": {
            "text_similarity": {
                "type": "text_similarity",
                "id": "TextSim123",
                "evaluation_metric": "fuzzy_match",
                "input": "{{sample.output_text}}",
                "reference": "{{item.ground_truth}}",
            }
        },
    }

    result, mapping = create_testing_criteria(
        evaluators, evaluator_metadata, input_data
    )

    assert len(result) == 2

    # First should be azure evaluator
    assert result[0]["type"] == "azure_ai_evaluator"
    assert result[0]["evaluator_name"] == "builtin.coherence"

    # Second should be custom evaluator
    assert result[1]["type"] == "text_similarity"
    assert result[1]["name"] == "text_similarity"
    assert "evaluator_name" not in result[1]
    assert "id" not in result[1]

    assert mapping["coherence"] == "builtin.coherence"
    assert mapping["text_similarity"] == "text_similarity"


def test_create_testing_criteria_openai_type_from_metadata():
    """Test creating testing criteria for OpenAI-type evaluator detected from metadata."""
    evaluators = ["text_similarity"]
    evaluator_metadata = {
        "text_similarity": {
            "metrics": {"similarity": {"data_type": "continuous"}},
            "categories": [],
            "init_parameters": {"required": ["input", "reference"]},
            "data_schema": {},
            "is_openai_type": True,
        }
    }
    input_data = {
        "data": [{"query": "test"}],
        "openai_graders": {
            "text_similarity": {
                "type": "text_similarity",
                "evaluation_metric": "fuzzy_match",
                "input": "{{sample.output_text}}",
                "reference": "{{item.ground_truth}}",
            }
        },
    }

    result, mapping = create_testing_criteria(
        evaluators, evaluator_metadata, input_data
    )

    assert len(result) == 1
    assert result[0]["type"] == "text_similarity"
    assert result[0]["evaluation_metric"] == "fuzzy_match"
    assert result[0]["input"] == "{{sample.output_text}}"
    assert result[0]["reference"] == "{{item.ground_truth}}"
    assert "data_mapping" not in result[0]
    assert mapping["text_similarity"] == "text_similarity"


def test_create_testing_criteria_openai_type_no_config():
    """Test creating testing criteria for OpenAI-type evaluator without config raises error."""
    evaluators = ["text_similarity"]
    evaluator_metadata = {
        "text_similarity": {
            "metrics": {"similarity": {"data_type": "continuous"}},
            "categories": [],
            "init_parameters": {},
            "data_schema": {},
            "is_openai_type": True,
        }
    }
    input_data = {"data": [{"query": "test"}]}

    with pytest.raises(ValueError, match="requires configuration in 'openai_graders'"):
        create_testing_criteria(evaluators, evaluator_metadata, input_data)
