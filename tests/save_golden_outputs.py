#!/usr/bin/env python3
"""
Save Golden Outputs for Regression Testing

This script copies the test outputs from test_results/ to a golden_outputs/
directory that can be committed to the repository.

These golden outputs serve as the baseline for detecting regressions.

Usage:
    # After running tests successfully
    python save_golden_outputs.py --test_type quick
    python save_golden_outputs.py --test_type full
"""

import argparse
import os
import shutil
import json
from pathlib import Path
from datetime import datetime


def save_golden_outputs(test_type: str, source_dir: str = None, golden_dir: str = "golden_outputs"):
    """
    Save test outputs as golden reference files.

    Args:
        test_type: Type of test ("quick" or "full")
        source_dir: Source directory (default: test_results or test_results_full)
        golden_dir: Golden outputs directory (default: golden_outputs)
    """
    # Determine source directory
    if source_dir is None:
        source_dir = "test_results" if test_type == "quick" else "test_results_full"

    if not os.path.exists(source_dir):
        print(f"❌ Error: Source directory not found: {source_dir}")
        print(f"   Run the {test_type} test first:")
        print(f"   python run_test_{test_type}.py --data_file bluesky_data/personas.pkl")
        return 1

    # Create golden outputs directory structure
    golden_subdir = os.path.join(golden_dir, test_type)
    os.makedirs(golden_subdir, exist_ok=True)

    print("=" * 80)
    print(f"SAVING GOLDEN OUTPUTS - {test_type.upper()} TEST")
    print("=" * 80)
    print(f"Source: {source_dir}")
    print(f"Destination: {golden_subdir}")
    print("-" * 80)

    # Find all JSON files (recursively, in case model names create subdirs)
    json_files = list(Path(source_dir).rglob("*.json"))

    if not json_files:
        print("⚠️  No JSON files found in source directory")
        return 1

    # Exclude test report files
    output_files = [f for f in json_files if "test_report" not in f.name]
    report_files = [f for f in json_files if "test_report" in f.name]

    print(f"\nFound {len(output_files)} output files")
    print(f"Found {len(report_files)} test report files")

    # Copy output files
    copied_files = []
    for src_file in output_files:
        # Preserve directory structure relative to source
        rel_path = src_file.relative_to(source_dir)
        dest_file = Path(golden_subdir) / rel_path

        # Create destination directory if needed
        dest_file.parent.mkdir(parents=True, exist_ok=True)

        # Copy file
        shutil.copy2(src_file, dest_file)
        copied_files.append(str(dest_file))
        print(f"  ✅ Copied: {rel_path}")

    # Copy test report
    for src_file in report_files:
        dest_file = Path(golden_subdir) / src_file.name
        shutil.copy2(src_file, dest_file)
        print(f"  📄 Copied report: {src_file.name}")

    # Create metadata file
    metadata = {
        "test_type": test_type,
        "created": datetime.now().isoformat(),
        "source_directory": source_dir,
        "num_files": len(copied_files),
        "files": copied_files,
    }

    metadata_file = os.path.join(golden_subdir, "golden_metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n📝 Metadata saved to: {metadata_file}")

    # Create .gitignore to prevent accidental commits of large files
    gitignore_path = os.path.join(golden_dir, ".gitignore")
    if not os.path.exists(gitignore_path):
        with open(gitignore_path, "w") as f:
            f.write("# Keep golden outputs small\n")
            f.write("# Uncomment if outputs become too large:\n")
            f.write("# *.json\n")
            f.write("# !golden_metadata.json\n")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✅ Saved {len(copied_files)} golden output files")
    print(f"📁 Location: {golden_subdir}")
    print("\nNext steps:")
    print("  1. Review the golden outputs to ensure they're correct")
    print("  2. Commit to repository:")
    print(f"     git add {golden_dir}/{test_type}")
    print(f"     git commit -m 'Add golden outputs for {test_type} test'")
    print("  3. Use compare_with_golden.py to detect future changes")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Save test outputs as golden reference files for regression testing"
    )
    parser.add_argument(
        "--test_type",
        type=str,
        required=True,
        choices=["quick", "full"],
        help="Type of test (quick or full)"
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        default=None,
        help="Source directory (default: test_results or test_results_full)"
    )
    parser.add_argument(
        "--golden_dir",
        type=str,
        default="golden_outputs",
        help="Golden outputs directory (default: golden_outputs)"
    )

    args = parser.parse_args()

    exit_code = save_golden_outputs(
        test_type=args.test_type,
        source_dir=args.source_dir,
        golden_dir=args.golden_dir
    )

    return exit_code


if __name__ == "__main__":
    import sys
    sys.exit(main())
