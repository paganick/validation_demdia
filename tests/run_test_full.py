#!/usr/bin/env python3
"""
Full test suite for simulation pipeline.

Tests all model configurations (non-finetuned) with 1 user and 5 responses.
More comprehensive than quick test but takes longer.

Usage:
    python run_test_full.py --data_file bluesky_data/personas.pkl

Expected runtime: ~10-20 minutes (depending on GPU and models)
"""

import argparse
import os
import sys
import json
import subprocess
from pathlib import Path
from test_seed_utils import set_global_seed, get_test_seed


def run_test(data_file: str, output_dir: str = "test_results_full", n_users: int = 1, n_responses: int = 5):
    """
    Run full test with all models (non-finetuned configurations).

    Args:
        data_file: Path to the data pickle file
        output_dir: Directory for test outputs
        n_users: Number of users to test (default: 1)
        n_responses: Number of responses per user (default: 5)
    """
    # Set global seed for reproducibility
    seed = get_test_seed()
    set_global_seed(seed)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Test configuration directory
    test_config_dir = "tests/test_configs"

    # All non-finetuned test configs (9 total)
    test_configs = [
        # Llama-3.1-8B (3 configs)
        "llama3.1_base.yaml",
        "llama3.1_base_10examples.yaml",
        "llama3.1_base_withcontext_10examples.yaml",
        # DeepSeek-R1 (3 configs)
        "deepseekR1_base.yaml",
        "deepseekR1_base_10examples.yaml",
        "deepseekR1_base_withcontext_10examples.yaml",
        # Mistral (3 configs)
        "mistral_base.yaml",
        "mistral_base_10examples.yaml",
        "mistral_base_withcontext_10examples.yaml",
    ]

    print("=" * 80)
    print("FULL TEST SUITE - All Models (Non-Finetuned)")
    print("=" * 80)
    print(f"Data file: {data_file}")
    print(f"Output directory: {output_dir}")
    print(f"Random seed: {seed}")
    print(f"Test users: {n_users}")
    print(f"Responses per user: {n_responses}")
    print(f"Configurations to test: {len(test_configs)}")
    print("\nModels being tested:")
    print("  - Llama-3.1-8B (3 configurations)")
    print("  - DeepSeek-R1-Distill-Llama-8B (3 configurations)")
    print("  - Mistral-7B-v0.1 (3 configurations)")
    print("-" * 80)

    results = []
    failed_configs = []

    for idx, config_name in enumerate(test_configs, 1):
        config_path = os.path.join(test_config_dir, config_name)

        if not os.path.exists(config_path):
            print(f"⚠️  [{idx}/{len(test_configs)}] Config not found: {config_path}")
            failed_configs.append({"config": config_name, "error": "Config file not found"})
            continue

        print(f"\n🧪 [{idx}/{len(test_configs)}] Testing: {config_name}")
        print(f"   Config: {config_path}")

        # Run simulation
        cmd = [
            sys.executable,
            "run_simulation.py",
            "--config", config_path,
            "--data_file", data_file,
            "--output_dir", output_dir,
            "--n_users", str(n_users),
            "--n_responses_per_user", str(n_responses),
        ]

        try:
            print(f"   Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=900  # 15 minute timeout per config
            )

            if result.returncode == 0:
                print(f"   ✅ Success!")
                results.append({
                    "config": config_name,
                    "status": "success",
                    "stdout": result.stdout,
                })
            else:
                print(f"   ❌ Failed with return code {result.returncode}")
                print(f"   Error output:\n{result.stderr}")
                failed_configs.append({
                    "config": config_name,
                    "error": result.stderr,
                    "return_code": result.returncode
                })

        except subprocess.TimeoutExpired:
            print(f"   ⏰ Timeout after 15 minutes")
            failed_configs.append({"config": config_name, "error": "Timeout"})
        except Exception as e:
            print(f"   ❌ Exception: {e}")
            failed_configs.append({"config": config_name, "error": str(e)})

    # Summary
    print("\n" + "=" * 80)
    print("FULL TEST SUMMARY")
    print("=" * 80)
    print(f"Total configurations: {len(test_configs)}")
    print(f"Successful: {len(results)}")
    print(f"Failed: {len(failed_configs)}")

    if failed_configs:
        print("\n⚠️  Failed configurations:")
        for fail in failed_configs:
            print(f"  - {fail['config']}: {fail.get('error', 'Unknown error')[:100]}")

    # Save test report
    report_path = os.path.join(output_dir, "full_test_report.json")
    report = {
        "test_type": "full",
        "seed": seed,
        "n_users": n_users,
        "n_responses": n_responses,
        "data_file": data_file,
        "total_configs": len(test_configs),
        "successful": len(results),
        "failed": len(failed_configs),
        "results": results,
        "failures": failed_configs,
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n📄 Test report saved to: {report_path}")

    # Validate outputs
    print("\n" + "=" * 80)
    print("VALIDATING OUTPUTS")
    print("=" * 80)
    validate_test_outputs(output_dir, n_users, n_responses)

    # Return exit code
    return 0 if len(failed_configs) == 0 else 1


def validate_test_outputs(output_dir: str, expected_users: int, expected_responses: int):
    """
    Validate that test outputs are correct.

    Args:
        output_dir: Directory containing test outputs
        expected_users: Expected number of users
        expected_responses: Expected responses per user
    """
    json_files = list(Path(output_dir).glob("*.json"))

    # Exclude the report file
    json_files = [f for f in json_files if "test_report" not in f.name]

    if not json_files:
        print("⚠️  No output JSON files found")
        return

    print(f"Found {len(json_files)} output files to validate")

    validation_summary = {"passed": 0, "failed": 0}

    for json_file in json_files:
        print(f"\n📊 Validating: {json_file.name}")

        try:
            with open(json_file) as f:
                data = json.load(f)

            file_valid = True

            # Count unique users
            users = set(entry["user"] for entry in data)
            print(f"   Users found: {len(users)} (expected: {expected_users})")

            if len(users) != expected_users:
                print(f"   ⚠️  User count mismatch!")
                file_valid = False

            # Check responses per user
            for user in users:
                user_entries = [e for e in data if e["user"] == user]
                print(f"   User '{user}': {len(user_entries)} responses (expected: {expected_responses})")

                if len(user_entries) != expected_responses:
                    print(f"      ⚠️  Response count mismatch!")
                    file_valid = False

                # Check that all entries have required fields
                for entry in user_entries:
                    required_fields = ["user", "model", "response", "original_message", "reply_to"]
                    missing = [f for f in required_fields if f not in entry]
                    if missing:
                        print(f"      ⚠️  Missing fields: {missing}")
                        file_valid = False

            if file_valid:
                print(f"   ✅ Validation passed")
                validation_summary["passed"] += 1
            else:
                print(f"   ❌ Validation failed")
                validation_summary["failed"] += 1

        except Exception as e:
            print(f"   ❌ Validation error: {e}")
            validation_summary["failed"] += 1

    # Print summary
    print("\n" + "-" * 80)
    print("VALIDATION SUMMARY")
    print("-" * 80)
    print(f"Passed: {validation_summary['passed']}")
    print(f"Failed: {validation_summary['failed']}")


def main():
    parser = argparse.ArgumentParser(
        description="Full test suite for simulation (all models, non-finetuned)"
    )
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="Path to data pickle file (e.g., bluesky_data/personas.pkl)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="test_results_full",
        help="Output directory for test results (default: test_results_full)"
    )
    parser.add_argument(
        "--n_users",
        type=int,
        default=1,
        help="Number of users to test (default: 1)"
    )
    parser.add_argument(
        "--n_responses",
        type=int,
        default=5,
        help="Number of responses per user (default: 5)"
    )

    args = parser.parse_args()

    # Validate data file exists
    if not os.path.exists(args.data_file):
        print(f"❌ Error: Data file not found: {args.data_file}")
        sys.exit(1)

    # Run test
    exit_code = run_test(
        data_file=args.data_file,
        output_dir=args.output_dir,
        n_users=args.n_users,
        n_responses=args.n_responses
    )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
