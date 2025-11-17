#!/usr/bin/env python3
"""
Reproducibility Verification Script

Runs the same configuration twice with the same seed and verifies
that the outputs are byte-for-byte identical.

This is crucial for:
1. Debugging - ensuring changes don't affect determinism
2. Testing - validating that seeds are properly fixed
3. Research - ensuring reproducible results

Usage:
    python verify_reproducibility.py --data_file bluesky_data/personas.pkl
"""

import argparse
import os
import sys
import json
import subprocess
import hashlib
from pathlib import Path
from test_seed_utils import get_test_seed


def compute_file_hash(filepath: str) -> str:
    """
    Compute SHA-256 hash of a file.

    Args:
        filepath: Path to the file

    Returns:
        Hexadecimal hash string
    """
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def run_simulation(config_path: str, data_file: str, output_dir: str, seed: int) -> str:
    """
    Run a single simulation.

    Args:
        config_path: Path to config file
        data_file: Path to data file
        output_dir: Output directory
        seed: Random seed

    Returns:
        Path to output JSON file
    """
    cmd = [
        sys.executable,
        "run_simulation.py",
        "--config", config_path,
        "--data_file", data_file,
        "--output_dir", output_dir,
        "--n_users", "1",
        "--n_responses_per_user", "3",
        "--seed", str(seed)
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        print(f"❌ Simulation failed!")
        print(f"Error: {result.stderr}")
        raise RuntimeError(f"Simulation failed with return code {result.returncode}")

    # Find the output file (search recursively in subdirectories)
    json_files = list(Path(output_dir).rglob("*.json"))
    if not json_files:
        raise RuntimeError(f"No output JSON file found in {output_dir}")

    return str(json_files[0])


def compare_json_files(file1: str, file2: str) -> dict:
    """
    Compare two JSON files and return detailed comparison results.

    Args:
        file1: Path to first JSON file
        file2: Path to second JSON file

    Returns:
        Dictionary with comparison results
    """
    # First, compare file hashes (fastest check)
    hash1 = compute_file_hash(file1)
    hash2 = compute_file_hash(file2)

    result = {
        "identical_hashes": hash1 == hash2,
        "hash1": hash1,
        "hash2": hash2,
    }

    # Load and compare JSON content
    with open(file1) as f:
        data1 = json.load(f)
    with open(file2) as f:
        data2 = json.load(f)

    result["identical_structure"] = len(data1) == len(data2)
    result["num_entries_1"] = len(data1)
    result["num_entries_2"] = len(data2)

    if len(data1) != len(data2):
        return result

    # Compare each entry
    differences = []
    for i, (entry1, entry2) in enumerate(zip(data1, data2)):
        if entry1 != entry2:
            # Find specific differences
            diff = {}
            for key in set(entry1.keys()) | set(entry2.keys()):
                val1 = entry1.get(key)
                val2 = entry2.get(key)
                if val1 != val2:
                    diff[key] = {
                        "run1": str(val1)[:100],  # Truncate for display
                        "run2": str(val2)[:100],
                    }
            if diff:
                differences.append({
                    "entry_index": i,
                    "differences": diff
                })

    result["identical_content"] = len(differences) == 0
    result["num_differences"] = len(differences)
    result["differences"] = differences[:10]  # Only show first 10 differences

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Verify reproducibility of simulation results"
    )
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="Path to data pickle file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="test_configs/llama3.1_base.yaml",
        help="Config file to test (default: test_configs/llama3.1_base.yaml)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed to use (default: test seed from test_seed_utils)"
    )

    args = parser.parse_args()

    # Use test seed if not specified
    seed = args.seed if args.seed is not None else get_test_seed()

    # Validate inputs
    if not os.path.exists(args.data_file):
        print(f"❌ Error: Data file not found: {args.data_file}")
        sys.exit(1)

    if not os.path.exists(args.config):
        print(f"❌ Error: Config file not found: {args.config}")
        sys.exit(1)

    print("=" * 80)
    print("REPRODUCIBILITY VERIFICATION TEST")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Data file: {args.data_file}")
    print(f"Seed: {seed}")
    print("-" * 80)

    # Create temporary output directories
    output_dir_1 = "test_repro_run1"
    output_dir_2 = "test_repro_run2"

    os.makedirs(output_dir_1, exist_ok=True)
    os.makedirs(output_dir_2, exist_ok=True)

    try:
        # Run 1
        print("\n🧪 Running simulation #1...")
        output_file_1 = run_simulation(args.config, args.data_file, output_dir_1, seed)
        print(f"   ✅ Output: {output_file_1}")

        # Run 2
        print("\n🧪 Running simulation #2...")
        output_file_2 = run_simulation(args.config, args.data_file, output_dir_2, seed)
        print(f"   ✅ Output: {output_file_2}")

        # Compare outputs
        print("\n🔍 Comparing outputs...")
        comparison = compare_json_files(output_file_1, output_file_2)

        print("\n" + "=" * 80)
        print("COMPARISON RESULTS")
        print("=" * 80)

        if comparison["identical_hashes"]:
            print("✅ SUCCESS: Outputs are byte-for-byte identical!")
            print(f"   SHA-256 hash: {comparison['hash1']}")
            print("\n🎉 Simulation is fully reproducible with fixed seed!")
            exit_code = 0
        else:
            print("❌ FAILURE: Outputs differ!")
            print(f"   Hash 1: {comparison['hash1']}")
            print(f"   Hash 2: {comparison['hash2']}")
            print(f"\n   Entries in run 1: {comparison['num_entries_1']}")
            print(f"   Entries in run 2: {comparison['num_entries_2']}")

            if comparison["identical_structure"]:
                print("   Structure is identical (same number of entries)")
            else:
                print("   ⚠️  Structure differs (different number of entries)")

            if comparison.get("differences"):
                print(f"\n   Number of differing entries: {comparison['num_differences']}")
                print("\n   First few differences:")
                for diff in comparison["differences"][:3]:
                    print(f"      Entry {diff['entry_index']}:")
                    for key, values in diff["differences"].items():
                        print(f"         {key}:")
                        print(f"            Run 1: {values['run1']}")
                        print(f"            Run 2: {values['run2']}")

            print("\n⚠️  Simulation is NOT reproducible. Check:")
            print("   1. Model temperature/sampling settings")
            print("   2. Random operations in code")
            print("   3. GPU non-deterministic operations")

            exit_code = 1

        print("\n" + "=" * 80)
        print(f"Output files preserved in:")
        print(f"  - {output_dir_1}")
        print(f"  - {output_dir_2}")

        sys.exit(exit_code)

    except Exception as e:
        print(f"\n❌ Error during verification: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
