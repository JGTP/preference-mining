from src.pipeline import execute_pipeline
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Execute data processing and classification pipeline"
    )
    parser.add_argument("data_path", type=str, help="Path to input Excel file")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Whether to truncate the data for testing",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/preprocessing.yaml",
        help="Path to preprocessing configuration file",
    )
    parser.add_argument(
        "--target", type=str, default="target", help="Name of target column"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save processed data",
    )
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Save intermediate processed data",
    )

    args = parser.parse_args()

    execute_pipeline(
        args.data_path,
        args.test,
        args.config,
        args.target,
        args.output_dir,
        args.save_intermediate,
    )
