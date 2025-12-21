"""Tests for analysis processing functions."""

import json
import tempfile
from pathlib import Path

from azure.ai.projects.models import EvaluatorMetricDirection, EvaluatorMetricType

from analysis.analysis import DesiredDirection, EvaluationScoreDataType
from analysis.processing import _convert_sdk_enums_to_analysis, convert_json_to_jsonl


def test_convert_json_to_jsonl():
    """Test converting JSON to JSONL format."""
    # Create temporary input file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json_data = {
            "data": [
                {"query": "test1", "context": "context1"},
                {"query": "test2", "context": "context2"},
            ]
        }
        json.dump(json_data, f)
        json_path = Path(f.name)

    try:
        # Convert to JSONL
        jsonl_path = convert_json_to_jsonl(json_path)

        # Verify JSONL file exists and has correct content
        assert jsonl_path.exists()
        assert jsonl_path.suffix == ".jsonl"

        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 2

            line1 = json.loads(lines[0])
            assert line1["query"] == "test1"
            assert line1["context"] == "context1"

            line2 = json.loads(lines[1])
            assert line2["query"] == "test2"
            assert line2["context"] == "context2"

    finally:
        # Cleanup
        json_path.unlink(missing_ok=True)
        if "jsonl_path" in locals():
            jsonl_path.unlink(missing_ok=True)


def test_convert_json_to_jsonl_custom_output():
    """Test converting JSON to JSONL with custom output path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create input file
        json_path = Path(tmpdir) / "input.json"
        json_data = {"data": [{"query": "test"}]}
        json_path.write_text(json.dumps(json_data), encoding="utf-8")

        # Convert with custom output
        output_path = Path(tmpdir) / "output.jsonl"
        result_path = convert_json_to_jsonl(json_path, output_path)

        assert result_path == output_path
        assert output_path.exists()


def test_convert_sdk_enums_to_analysis():
    """Test conversion of SDK enums to analysis enums."""
    sdk_metadata = {
        "data_type": EvaluatorMetricType.CONTINUOUS,
        "desired_direction": EvaluatorMetricDirection.INCREASE,
        "field": "score",
    }

    result = _convert_sdk_enums_to_analysis(sdk_metadata)

    # Should convert SDK enums to analysis enums
    assert result["data_type"] == EvaluationScoreDataType.CONTINUOUS
    assert result["desired_direction"] == DesiredDirection.INCREASE
    assert result["field"] == "score"


def test_convert_sdk_enums_to_analysis_boolean():
    """Test enum conversion with boolean data type."""
    sdk_metadata = {
        "data_type": EvaluatorMetricType.BOOLEAN,
        "desired_direction": EvaluatorMetricDirection.DECREASE,
        "field": "is_valid",
    }

    result = _convert_sdk_enums_to_analysis(sdk_metadata)

    assert result["data_type"] == EvaluationScoreDataType.BOOLEAN
    assert result["desired_direction"] == DesiredDirection.DECREASE
    assert result["field"] == "is_valid"


def test_convert_sdk_enums_to_analysis_ordinal():
    """Test enum conversion with ordinal data type."""
    sdk_metadata = {
        "data_type": EvaluatorMetricType.ORDINAL,
        "desired_direction": EvaluatorMetricDirection.NEUTRAL,
        "field": "rating",
    }

    result = _convert_sdk_enums_to_analysis(sdk_metadata)

    assert result["data_type"] == EvaluationScoreDataType.ORDINAL
    assert result["desired_direction"] == DesiredDirection.NEUTRAL
    assert result["field"] == "rating"
