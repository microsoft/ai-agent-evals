"""Unit tests for the input data validation function."""

import pytest
from action import validate_input_data

eval_metadata = {
    "sections": [
        {
            "evaluators": [
                {"class": "IntentResolutionEvaluator"},
                {"class": "RelevanceEvaluator"},
            ]
        }
    ]
}


def test_valid_input_data():
    """Test that valid input data passes validation."""
    valid_data = {
        "name": "Test Dataset",
        "evaluators": ["IntentResolutionEvaluator", "RelevanceEvaluator"],
        "data": [
            {"id": "test_query_01", "query": "What is the capital of France?"},
            {"id": "test_query_02", "query": "How do I sort a list in Python?"},
        ],
    }

    # This should not raise any exceptions
    validate_input_data(valid_data, eval_metadata)


def test_missing_required_fields():
    """Test that validation fails when required fields are missing."""
    # Missing name field
    invalid_data_1 = {
        "evaluators": ["IntentResolutionEvaluator"],
        "data": [{"query": "test query"}],
    }

    with pytest.raises(ValueError) as excinfo:
        validate_input_data(invalid_data_1, eval_metadata)
    assert "missing required fields" in str(excinfo.value)

    # Missing evaluators field
    invalid_data_2 = {"name": "Test Dataset", "data": [{"query": "test query"}]}

    with pytest.raises(ValueError) as excinfo:
        validate_input_data(invalid_data_2, eval_metadata)
    assert "missing required fields" in str(excinfo.value)

    # Missing data field
    invalid_data_3 = {
        "name": "Test Dataset",
        "evaluators": ["IntentResolutionEvaluator"],
    }

    with pytest.raises(ValueError) as excinfo:
        validate_input_data(invalid_data_3, eval_metadata)
    assert "missing required fields" in str(excinfo.value)


def test_invalid_field_types():
    """Test that validation fails with incorrect field types."""
    # Invalid name type
    invalid_data_1 = {
        "name": 123,  # Should be a string
        "evaluators": ["IntentResolutionEvaluator"],
        "data": [{"query": "test query"}],
    }

    with pytest.raises(ValueError) as excinfo:
        validate_input_data(invalid_data_1, eval_metadata)
    assert "must be a string" in str(excinfo.value)

    # Invalid evaluators type
    invalid_data_2 = {
        "name": "Test Dataset",
        "evaluators": "IntentResolutionEvaluator",  # Should be a list
        "data": [{"query": "test query"}],
    }

    with pytest.raises(ValueError) as excinfo:
        validate_input_data(invalid_data_2, eval_metadata)
    assert "must be a list" in str(excinfo.value)

    # Invalid data type
    invalid_data_3 = {
        "name": "Test Dataset",
        "evaluators": ["IntentResolutionEvaluator"],
        "data": "test query",  # Should be a list
    }

    with pytest.raises(ValueError) as excinfo:
        validate_input_data(invalid_data_3, eval_metadata)
    assert "must be a list" in str(excinfo.value)


def test_data_item_validation():
    """Test validation of individual data items."""
    # Data item missing query
    invalid_data = {
        "name": "Test Dataset",
        "evaluators": ["IntentResolutionEvaluator"],
        "data": [{"id": "test_01"}],  # Missing required query field
    }

    with pytest.raises(ValueError) as excinfo:
        validate_input_data(invalid_data, eval_metadata)
    assert "missing required field 'query'" in str(excinfo.value)

    # Data item is not a dictionary
    invalid_data_2 = {
        "name": "Test Dataset",
        "evaluators": ["IntentResolutionEvaluator"],
        "data": ["This is just a string"],  # Should be a dictionary
    }

    with pytest.raises(ValueError) as excinfo:
        validate_input_data(invalid_data_2, eval_metadata)
    assert "must be a dictionary" in str(excinfo.value)


def test_unknown_evaluator_validation():
    """Test that validation fails with unknown evaluators."""
    invalid_data = {
        "name": "Test Dataset",
        "evaluators": ["UnknownEvaluator", "AnotherInvalidOne"],
        "data": [{"query": "test query"}],
    }

    with pytest.raises(ValueError) as excinfo:
        validate_input_data(invalid_data, eval_metadata)
    assert "Unknown evaluators specified" in str(excinfo.value)


def test_duplicate_id_validation():
    """Test that validation fails when duplicate IDs are found in the data."""
    # Data with duplicate IDs
    invalid_data = {
        "name": "Test Dataset",
        "evaluators": ["IntentResolutionEvaluator"],
        "data": [
            {"id": "duplicate_id", "query": "First query with duplicate ID"},
            {"id": "unique_id", "query": "Query with unique ID"},
            {"id": "duplicate_id", "query": "Second query with duplicate ID"},
        ],
    }

    with pytest.raises(ValueError) as excinfo:
        validate_input_data(invalid_data, eval_metadata)
    assert "Duplicate ID 'duplicate_id' found in 'data'" in str(excinfo.value)

    # Test empty data list
    invalid_empty_data = {
        "name": "Test Dataset",
        "evaluators": ["IntentResolutionEvaluator"],
        "data": [],
    }

    with pytest.raises(ValueError) as excinfo:
        validate_input_data(invalid_empty_data, eval_metadata)
    assert "cannot be empty" in str(excinfo.value)
