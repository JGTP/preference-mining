from src.pipeline import execute_pipeline


def test_pipeline_execution():
    try:
        execute_pipeline(
            "data/globalterrorismdb_0522dist.xlsx",
            test_size=100,
            target_column="suicide",
            min_year=1970,
        )
    except Exception as e:
        raise ValueError(f"Pipeline execution failed: {e}")
