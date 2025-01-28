import argparse
import subprocess
import sys
from pathlib import Path
import json
import time
from datetime import datetime
import itertools


class ParameterSweep:
    def __init__(
        self,
        data_path,
        max_set_sizes,
        top_features_values,
        base_output_dir="results/parameter_sweep",
        **kwargs,
    ):
        """
        Initialise parameter sweep with configurations.

        Args:
            data_path (str): Path to input data file
            max_set_sizes (list): List of max set sizes to sweep
            top_features_values (list): List of top features values to sweep
            base_output_dir (str): Base directory for sweep results
            **kwargs: Additional parameters to pass to main.py
        """
        self.data_path = data_path
        self.max_set_sizes = sorted(max_set_sizes)
        self.top_features_values = sorted(top_features_values)
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

        # Store additional parameters
        self.extra_params = kwargs

        # Tracking file for resume functionality
        self.tracking_file = self.base_output_dir / "sweep_progress.json"

        # Initialise or load progress
        self.load_progress()

    def load_progress(self):
        """
        Load existing progress or initialise a new sweep.
        """
        if self.tracking_file.exists():
            with open(self.tracking_file, "r") as f:
                self.progress = json.load(f)
        else:
            # Generate high-level parameter combinations
            self.progress = {
                "completed": [],
                "parameter_combinations": [
                    {"max_set_size": max_set_size, "top_features": top_features}
                    for max_set_size in self.max_set_sizes
                    for top_features in self.top_features_values
                ],
            }

        # Remove already completed combinations
        self.progress["parameter_combinations"] = [
            combo
            for combo in self.progress["parameter_combinations"]
            if combo not in self.progress["completed"]
        ]

    def save_progress(self):
        """
        Save current progress to tracking file.
        """
        with open(self.tracking_file, "w") as f:
            json.dump(self.progress, f, indent=2)

    def run_single_parameter_set(self, params):
        """
        Run main.py for a single high-level parameter configuration.

        Args:
            params (dict): High-level parameters for this run

        Returns:
            bool: Whether the run was successful
        """
        # Construct timestamp for unique output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_output_dir = (
            self.base_output_dir
            / f"run_{params['max_set_size']}_{params['top_features']}_{timestamp}"
        )
        run_output_dir.mkdir(parents=True, exist_ok=True)

        # Construct command
        cmd = [
            sys.executable,  # Python interpreter
            "main.py",
            self.data_path,
            "--max_set_size",
            str(params["max_set_size"]),
            "--top_features",
            str(params["top_features"]),
            "--output_dir",
            str(run_output_dir),
        ]

        # Add extra parameters
        for key, value in self.extra_params.items():
            # Handle different types of parameters
            if isinstance(value, list):
                cmd.extend([f"--{key}"] + [str(v) for v in value])
            elif value is not None:
                cmd.extend([f"--{key}", str(value)])

        # Log the command
        cmd_str = " ".join(str(c) for c in cmd)
        print(f"\nRunning command:\n{cmd_str}")

        # Write command to file
        with open(run_output_dir / "command.txt", "w", encoding="utf-8") as f:
            f.write(cmd_str)

        try:
            # Run the process with robust encoding handling
            result = subprocess.run(
                cmd,
                check=False,  # Don't raise exception on non-zero exit
                encoding="utf-8",  # Use UTF-8 encoding
                errors="replace",  # Replace undecodable bytes
                capture_output=True,
            )

            # Determine success
            success = result.returncode == 0

            # Save output logs with UTF-8 encoding
            with open(run_output_dir / "run.log", "w", encoding="utf-8") as f:
                f.write(f"Command:\n{cmd_str}\n\n")
                f.write(f"STDOUT:\n{result.stdout}\n\n")
                f.write(f"STDERR:\n{result.stderr}")

            # Print output for visibility
            if result.stdout:
                print("STDOUT:")
                print(result.stdout)
            if result.stderr:
                print("STDERR:")
                print(result.stderr)

            return success

        except Exception as e:
            print(f"Error running command: {e}")

            # Log the error
            with open(run_output_dir / "error.log", "w", encoding="utf-8") as f:
                f.write(f"Error running command: {e}\n")
                f.write(f"Command: {cmd_str}")

            return False

    def run_sweep(self):
        """
        Run the full parameter sweep, one high-level parameter set at a time.
        """
        print("Starting parameter sweep")
        print(
            f"Total high-level parameter combinations: {len(self.progress['parameter_combinations'])}"
        )

        # Track overall success
        total_runs = len(self.progress["parameter_combinations"])
        successful_runs = 0

        # Run each high-level parameter combination
        while self.progress["parameter_combinations"]:
            # Get the next high-level parameter set
            current_params = self.progress["parameter_combinations"].pop(0)

            print("\n" + "=" * 50)
            print(f"Running high-level parameters: {current_params}")
            print("=" * 50)

            # Run the current parameter set
            success = self.run_single_parameter_set(current_params)

            # Update progress
            if success:
                self.progress["completed"].append(current_params)
                successful_runs += 1
            else:
                # If run fails, you might want to handle this (e.g., retry or log)
                print(f"Run failed for parameters: {current_params}")

            # Save progress after each run
            self.save_progress()

            # Optional: Add a small delay between runs to prevent resource contention
            time.sleep(2)

        # Final summary
        print("\nParameter Sweep Complete")
        print(f"Total runs: {total_runs}")
        print(f"Successful runs: {successful_runs}")
        print(f"Failed runs: {total_runs - successful_runs}")


def main():
    parser = argparse.ArgumentParser(description="Run sequential parameter sweep")

    # Required argument
    parser.add_argument("data_path", help="Path to input Excel file")

    # High-level parameter sweep arguments
    parser.add_argument(
        "--max_set_sizes",
        type=int,
        nargs="+",
        default=[2, 3, 4],
        help="List of max set sizes to sweep",
    )
    parser.add_argument(
        "--top_features_values",
        type=int,
        nargs="+",
        default=[3, 4, 5],
        help="List of top features values to sweep",
    )

    # Lower-level parameters with more flexible options
    parser.add_argument("--test_size", type=int, default=None)
    parser.add_argument("--config", default="config/gtd.yaml")
    parser.add_argument("--cache_dir", default="data/cache")
    parser.add_argument("--min_year", type=int, default=1997)
    parser.add_argument("--output_dir", default="results/parameter_sweep")
    parser.add_argument("--n_splits", type=int, default=3)

    # Epsilon and delta parameters with more flexible configuration
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.1, 0.2])
    parser.add_argument("--delta_start", type=float, default=0.05)
    parser.add_argument("--delta_end", type=float, default=0.25)
    parser.add_argument("--delta_step", type=float, default=0.05)

    # Parse arguments
    args = parser.parse_args()

    # Extract additional parameters to pass to main.py
    extra_params = {
        k: v
        for k, v in vars(args).items()
        if k not in ["data_path", "max_set_sizes", "top_features_values"]
    }

    # Run parameter sweep
    sweep = ParameterSweep(
        data_path=args.data_path,
        max_set_sizes=args.max_set_sizes,
        top_features_values=args.top_features_values,
        **extra_params,
    )
    sweep.run_sweep()


if __name__ == "__main__":
    main()
