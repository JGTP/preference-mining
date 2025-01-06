import pytest
from src.pipeline import execute_pipeline


def test_pipeline_execution():
    # Ensure the pipeline runs without exceptions
    try:
        execute_pipeline("data/globalterrorismdb_0522dist.xlsx", test=False)
    except Exception as e:
        pytest.fail(f"Pipeline execution failed: {e}")
