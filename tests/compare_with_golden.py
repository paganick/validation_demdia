#!/usr/bin/env python3
"""
Compare Test Outputs with Golden Reference

This script compares new test outputs against the golden reference files
to detect regressions or unexpected changes.

Usage:
    # After running tests
    python tests/compare_with_golden.py --test_type quick
    python tests/compare_with_golden.py --test_type full

    # Detailed diff output
    python tests/compare_with_golden.py --test_type quick --verbose
"""

import argparse
import os
import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple


def compute_file_hash(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def compare_json_content(file1: str, file2: str) -> Dict:
    """
    Compare two JSON files field by field.

    Returns:
        Dictionary with comparison results
    """
    with open(file1) as f:
        data1 = json.load(f)
    with open(file2) as f:
        data2 = json.load(f)

    result = {
        "identical": data1 == data2,
        "size_match": len(data1) == len(data2),
        "size_golden": len(data1),
        "size_new": len(data2),
        "differences": []
    }

    if len(data1) != len(data2):
        result["differences"].append({
            "type": "size_mismatch",
            "golden": len(data1),
            "new": len(data2)
        })
        return result

    # Compare entries
    for i, (entry1, entry2) in enumerate(zip(data1, data2)):
        if entry1 != entry2:
            diff_fields = {}
            for key in set(entry1.keys()) | set(entry2.keys()):
                val1 = entry1.get(key)
                val2 = entry2.get(key)
                if val1 != val2:
                    diff_fields[key] = {
                        "golden": str(val1)[:200],
                        "new": str(val2)[:200]
                    }

            if diff_fields:
                result["differences"].append({
                    "type": "content_mismatch",
                    "entry": i,
                    "fields": diff_fields
                })

    return result


def find_matching_files(golden_dir: str, test_dir: str) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Find files in golden and test directories.

    Returns:
        Tuple of (matching_files, golden_only, test_only)
    """
    golden_files = {f.relative_to(golden_dir): f for f in Path(golden_dir).rglob("*.json")
                    if "metadata" not in f.name and "test_report" not in f.name}
    test_files = {f.relative_to(test_dir): f for f in Path(test_dir).rglob("*.json")
                  if "metadata" not in f.name and "test_report" not in f.name}

    common = set(golden_files.keys()) & set(test_files.keys())
    golden_only = set(golden_files.keys()) - set(test_files.keys())
    test_only = set(test_files.keys()) - set(golden_files.keys())

    matching = [(golden_files[k], test_files[k]) for k in common]

    return matching, [golden_files[k] for k in golden_only], [test_files[k] for k in test_only]


def compare_with_golden(test_type: str, test_dir: str = None, golden_dir: str = "golden_outputs",
                        verbose: bool = False) -> int:
    """
    Compare test outputs with golden reference.

    Returns:
        0 if identical, 1 if differences found, 2 if error
    """
    # Determine directories
    if test_dir is None:
        test_dir = "test_results" if test_type == "quick" else "test_results_full"

    golden_subdir = os.path.join(golden_dir, test_type)

    # Validate directories exist
    if not os.path.exists(golden_subdir):
        print(f"❌ Error: Golden outputs not found: {golden_subdir}")
        print(f"   Create golden outputs first:")
        print(f"   python save_golden_outputs.py --test_type {test_type}")
        return 2

    if not os.path.exists(test_dir):
        print(f"❌ Error: Test directory not found: {test_dir}")
        print(f"   Run the test first:")
        print(f"   python run_test_{test_type}.py --data_file bluesky_data/personas.pkl")
        return 2

    print("=" * 80)
    print(f"COMPARING WITH GOLDEN OUTPUTS - {test_type.upper()} TEST")
    print("=" * 80)
    print(f"Golden: {golden_subdir}")
    print(f"Test:   {test_dir}")
    print("-" * 80)

    # Find files
    matching, golden_only, test_only = find_matching_files(golden_subdir, test_dir)

    print(f"\nFiles to compare: {len(matching)}")
    if golden_only:
        print(f"⚠️  Files only in golden: {len(golden_only)}")
    if test_only:
        print(f"⚠️  Files only in test: {len(test_only)}")

    # Compare each file
    all_identical = True
    results = []

    for golden_file, test_file in matching:
        rel_path = golden_file.relative_to(golden_subdir)
        print(f"\n📊 Comparing: {rel_path}")

        # Quick hash comparison
        golden_hash = compute_file_hash(str(golden_file))
        test_hash = compute_file_hash(str(test_file))

        if golden_hash == test_hash:
            print(f"   ✅ Identical (hash: {golden_hash[:16]}...)")
            results.append({"file": str(rel_path), "status": "identical"})
        else:
            print(f"   ❌ Different!")
            print(f"      Golden hash: {golden_hash[:16]}...")
            print(f"      Test hash:   {test_hash[:16]}...")

            # Detailed comparison if verbose
            if verbose:
                comparison = compare_json_content(str(golden_file), str(test_file))

                if not comparison["size_match"]:
                    print(f"      Size mismatch: {comparison['size_golden']} vs {comparison['size_new']}")

                if comparison["differences"]:
                    print(f"      Found {len(comparison['differences'])} differences")
                    for i, diff in enumerate(comparison["differences"][:3], 1):
                        if diff["type"] == "content_mismatch":
                            print(f"        Difference {i} at entry {diff['entry']}:")
                            for field, values in list(diff["fields"].items())[:2]:
                                print(f"          {field}:")
                                print(f"            Golden: {values['golden'][:100]}")
                                print(f"            New:    {values['new'][:100]}")

            results.append({
                "file": str(rel_path),
                "status": "different",
                "golden_hash": golden_hash,
                "test_hash": test_hash
            })
            all_identical = False

    # Report missing files
    if golden_only:
        print("\n⚠️  Files only in golden (missing from test):")
        for f in golden_only:
            rel_path = f.relative_to(golden_subdir)
            print(f"   - {rel_path}")
            results.append({"file": str(rel_path), "status": "missing_from_test"})
            all_identical = False

    if test_only:
        print("\n⚠️  Files only in test (new files):")
        for f in test_only:
            rel_path = f.relative_to(test_dir)
            print(f"   - {rel_path}")
            results.append({"file": str(rel_path), "status": "new_file"})
            all_identical = False

    # Summary
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    identical_count = sum(1 for r in results if r["status"] == "identical")
    different_count = sum(1 for r in results if r["status"] == "different")
    missing_count = sum(1 for r in results if r["status"] == "missing_from_test")
    new_count = sum(1 for r in results if r["status"] == "new_file")

    print(f"Identical: {identical_count}")
    print(f"Different: {different_count}")
    print(f"Missing from test: {missing_count}")
    print(f"New files: {new_count}")

    if all_identical:
        print("\n✅ SUCCESS: All outputs match golden reference!")
        print("   No regressions detected.")
        return 0
    else:
        print("\n❌ CHANGES DETECTED")
        print("\nPossible reasons:")
        print("  1. Bug introduced (regression)")
        print("  2. Intentional code change")
        print("  3. Non-deterministic behavior (model sampling)")
        print("  4. Different random seed")
        print("\nIf changes are intentional:")
        print("  1. Review the changes carefully")
        print("  2. Update golden outputs:")
        print(f"     python save_golden_outputs.py --test_type {test_type}")
        print("  3. Commit new golden outputs to repository")

        if not verbose:
            print("\nFor detailed differences, run with --verbose")

        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Compare test outputs with golden reference files"
    )
    parser.add_argument(
        "--test_type",
        type=str,
        required=True,
        choices=["quick", "full"],
        help="Type of test (quick or full)"
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default=None,
        help="Test directory (default: test_results or test_results_full)"
    )
    parser.add_argument(
        "--golden_dir",
        type=str,
        default="golden_outputs",
        help="Golden outputs directory (default: golden_outputs)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed differences"
    )

    args = parser.parse_args()

    exit_code = compare_with_golden(
        test_type=args.test_type,
        test_dir=args.test_dir,
        golden_dir=args.golden_dir,
        verbose=args.verbose
    )

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
