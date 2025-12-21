"""Tests for action.py helper functions."""

import pytest

from action import (
    _build_base_data_mapping,
    _generate_data_mappings,
    _get_response_field,
    _validate_data_schema,
    _validate_init_parameters,
    create_testing_criteria,
)


def test_generate_data_mappings_empty():
    """Test data mapping generation with no input."""
    result = _generate_data_mappings(None)
    assert result == {}


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
        }
    }
    input_data = {"data": [{"query": "test"}]}

    result = create_testing_criteria(evaluators, evaluator_metadata, input_data)

    assert len(result) == 1
    assert result[0]["name"] == "coherence"
    assert result[0]["evaluator_name"] == "builtin.coherence"
    assert result[0]["type"] == "azure_ai_evaluator"
    assert "query" in result[0]["data_mapping"]


def test_create_testing_criteria_with_parameters():
    """Test creating testing criteria with evaluator parameters."""
    evaluators = ["custom.evaluator"]
    evaluator_metadata = {
        "custom.evaluator": {
            "metrics": {},
            "categories": [],
            "init_parameters": {"required": ["threshold"]},
            "data_schema": {"required": ["query"]},
        }
    }
    input_data = {"data": [{"query": "test"}]}
    evaluator_parameters = {"custom.evaluator": {"threshold": 0.8}}

    result = create_testing_criteria(
        evaluators, evaluator_metadata, input_data, evaluator_parameters
    )

    assert len(result) == 1
    assert result[0]["initialization_parameters"]["threshold"] == 0.8


def test_create_testing_criteria_agents_category():
    """Test creating testing criteria for agents category evaluator."""
    evaluators = ["builtin.task_adherence"]
    evaluator_metadata = {
        "builtin.task_adherence": {
            "metrics": {},
            "categories": ["agents"],
            "init_parameters": {},
            "data_schema": {},
        }
    }
    input_data = {"data": [{"query": "test"}]}

    result = create_testing_criteria(evaluators, evaluator_metadata, input_data)

    assert result[0]["data_mapping"]["response"] == "{{sample.output_items}}"


def test_create_testing_criteria_validation_error():
    """Test creating testing criteria with validation error."""
    evaluators = ["builtin.evaluator"]
    evaluator_metadata = {
        "builtin.evaluator": {
            "metrics": {},
            "categories": [],
            "init_parameters": {"required": ["missing_param"]},
            "data_schema": {},
        }
    }
    input_data = {"data": [{"query": "test"}]}

    with pytest.raises(ValueError, match="missing_param"):
        create_testing_criteria(evaluators, evaluator_metadata, input_data)
