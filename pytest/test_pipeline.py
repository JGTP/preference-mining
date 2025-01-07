import pytest
from src.pipeline import execute_pipeline

def test_pipeline_execution():
    try:
        execute_pipeline("data/globalterrorismdb_0522dist.xlsx", test=True, target_column='doubtterr')
    except Exception as e:
        pytest.fail(f"Pipeline execution failed: {e}")
