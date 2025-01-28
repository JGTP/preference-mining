from src.pipeline import execute_pipeline
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute analysis pipeline")
    parser.add_argument("data_path", help="Path to input Excel file")
    parser.add_argument("--test_size", type=int, default=None)
    parser.add_argument(
        "--config", default="config/gtd.yaml", help="Path to config file"
    )
    parser.add_argument("--cache_dir", default="data/cache", help="Cache directory")
    parser.add_argument(
        "--min_year", type=int, default=1997, help="Starting year for filtering"
    )
    parser.add_argument("--output_dir", default="results", help="Output directory")
    parser.add_argument("--n_splits", type=int, default=3, help="Number of CV splits")
    parser.add_argument(
        "--max_set_size", type=int, default=10, help="Maximum feature set size"
    )
    parser.add_argument(
        "--top_features",
        type=int,
        default=20,
        help="Number of top features to consider for stronger feature sets",
    )
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=[0.1, 0.2],
        help="List of epsilon values to use",
    )
    parser.add_argument(
        "--delta_start",
        type=float,
        default=0.05,
        help="Starting delta percentage",
    )
    parser.add_argument(
        "--delta_end",
        type=float,
        default=0.25,
        help="Ending delta percentage",
    )
    parser.add_argument(
        "--delta_step",
        type=float,
        default=0.05,
        help="Delta percentage step size",
    )
    args = parser.parse_args()

    deltas = [
        x * args.delta_step + args.delta_start
        for x in range(int((args.delta_end - args.delta_start) / args.delta_step) + 1)
    ]

    analysis_config = {
        "epsilons": args.epsilons,
        "deltas": deltas,
        "max_set_size": args.max_set_size,
        "top_features": args.top_features,
    }

    execute_pipeline(
        data_path=args.data_path,
        test_size=args.test_size,
        config_path=args.config,
        cache_dir=args.cache_dir,
        min_year=args.min_year,
        n_splits=args.n_splits,
        output_dir=args.output_dir,
        analysis_config=analysis_config,
    )
