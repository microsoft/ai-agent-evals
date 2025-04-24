import pytest
from unittest.mock import patch, MagicMock
import datetime

from azure.ai.projects.models import RunStatus
from action import simulate_question_answer

class MockError:
    def __init__(self, code):
        self.code = code

class MockRun:
    def __init__(self, status, error_code=None):
        self.status = status
        self.last_error = MockError(error_code) if error_code else None
        self.completed_at = datetime.datetime.now()
        self.created_at = datetime.datetime.now() - datetime.timedelta(seconds=1)  # 1 second ago
        self.usage = MagicMock()
        self.usage.completion_tokens = 100
        self.usage.prompt_tokens = 50

@patch('time.sleep')
def test_exponential_backoff(mock_sleep):
    """Test that the retry logic uses exponential backoff with appropriate wait times."""
    # Setup mocks
    mock_project_client = MagicMock()
    mock_agent = MagicMock()
    mock_thread = MagicMock()
    mock_thread.id = "test_thread_id"
    
    # Sequence of mock runs: 3 rate limit errors followed by success
    mock_runs = [
        MockRun(RunStatus.FAILED, "rate_limit_exceeded"),  # First attempt fails
        MockRun(RunStatus.FAILED, "rate_limit_exceeded"),  # Second attempt fails
        MockRun(RunStatus.FAILED, "rate_limit_exceeded"),  # Third attempt fails
        MockRun(RunStatus.COMPLETED)                       # Fourth attempt succeeds
    ]
    
    # Configure the mocks
    mock_project_client.agents.create_thread.return_value = mock_thread
    mock_project_client.agents.create_and_process_run.side_effect = mock_runs
    mock_converter = MagicMock()
    mock_project_client.agents.list_messages.return_value = MagicMock()
    
    # Patch AIAgentConverter to avoid actual file operations
    with patch('action.AIAgentConverter') as MockConverter:
        MockConverter.return_value = mock_converter
        mock_converter.prepare_evaluation_data.return_value = [{"query": "test query", "response": "test response"}]
        
        # Call the function
        input_data = {"query": "test query", "id": "test_id_1"}
        simulate_question_answer(mock_project_client, mock_agent, input_data)
        
        # Assert exponential backoff was used with correct wait times
        assert mock_sleep.call_count == 3  # Three retries before success
        
        # Check the wait times follow exponential pattern (base=2, with some jitter)
        # We can't check exact values due to random jitter, but we can verify the pattern
        wait_times = [call_args[0][0] for call_args in mock_sleep.call_args_list]
        
        # First retry should be ~2 seconds
        assert 1.5 <= wait_times[0] <= 2.5
        
        # Second retry should be ~4 seconds
        assert 3.5 <= wait_times[1] <= 4.5
        
        # Third retry should be ~8 seconds
        assert 7.5 <= wait_times[2] <= 8.5
        
        # Each wait time should be approximately double the previous (allowing for jitter)
        assert 1.8 <= wait_times[1] / wait_times[0] <= 2.2
        assert 1.8 <= wait_times[2] / wait_times[1] <= 2.2


@patch('time.sleep')
def test_retry_fails_after_max_attempts(mock_sleep):
    """Test that the function gives up after max retries."""
    # Setup mocks
    mock_project_client = MagicMock()
    mock_agent = MagicMock()
    mock_thread = MagicMock()
    mock_thread.id = "test_thread_id"

    # All attempts fail with rate limit errors
    mock_runs = [MockRun(RunStatus.FAILED, "rate_limit_exceeded") for _ in range(5)]

    # Configure the mocks
    mock_project_client.agents.create_thread.return_value = mock_thread
    mock_project_client.agents.create_and_process_run.side_effect = mock_runs

    # Call the function, expecting it to raise an exception
    input_data = {"query": "test query", "id": "test_id_1"}
    with pytest.raises(ValueError):
        simulate_question_answer(mock_project_client, mock_agent, input_data)    

    # Verify all retries were attempted
    assert mock_sleep.call_count == 4  # 5 attempts, 4 sleeps