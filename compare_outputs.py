#!/usr/bin/env python3
"""
Compare two sets of simulation outputs for reproducibility testing.

This script compares golden outputs (first run) against a new run to check:
1. Exact matches (perfect reproducibility)
2. Statistical similarity (distribution of responses)
3. Structural consistency (same keys, data types, etc.)

Usage:
    python compare_outputs.py golden_outputs/ reference_outputs/
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict
import difflib

def load_json_file(filepath):
    """Load a JSON file and return its contents."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def compare_responses(golden_data, new_data, filepath):
    """Compare two response datasets."""
    if golden_data is None or new_data is None:
        return {"error": "Failed to load one or both files"}

    if len(golden_data) != len(new_data):
        return {
            "match": False,
            "reason": "Different number of entries",
            "golden_count": len(golden_data),
            "new_count": len(new_data)
        }

    exact_matches = 0
    response_diffs = []
    metadata_diffs = []

    for i, (golden, new) in enumerate(zip(golden_data, new_data)):
        # Check if responses match exactly
        golden_response = golden.get('response', '')
        new_response = new.get('response', '')

        if golden_response == new_response:
            exact_matches += 1
        else:
            response_diffs.append({
                'index': i,
                'user': golden.get('user', 'unknown'),
                'golden': golden_response[:100],  # First 100 chars
                'new': new_response[:100],
                'similarity': difflib.SequenceMatcher(None, golden_response, new_response).ratio()
            })

        # Check metadata consistency
        for key in ['user', 'model', 'fine_tuned', 'retrieve_context', 'OPPU', 'n_style_examples']:
            if key in golden and key in new:
                if golden[key] != new[key]:
                    metadata_diffs.append({
                        'index': i,
                        'key': key,
                        'golden': golden[key],
                        'new': new[key]
                    })

    return {
        "match": exact_matches == len(golden_data),
        "total_entries": len(golden_data),
        "exact_matches": exact_matches,
        "match_percentage": (exact_matches / len(golden_data) * 100) if len(golden_data) > 0 else 0,
        "response_diffs": response_diffs[:5],  # First 5 diffs
        "metadata_diffs": metadata_diffs[:5],  # First 5 diffs
        "total_response_diffs": len(response_diffs),
        "total_metadata_diffs": len(metadata_diffs)
    }

def find_json_files(directory):
    """Find all JSON files in directory and subdirectories."""
    json_files = {}
    for root, dirs, files in os.walk(directory):
        # Skip 'full' and 'quick' test directories
        dirs[:] = [d for d in dirs if d not in ['full', 'quick']]

        for file in files:
            if file.endswith('.json') and file not in ['METADATA.txt', 'README.md', '.gitignore']:
                rel_path = os.path.relpath(os.path.join(root, file), directory)
                json_files[rel_path] = os.path.join(root, file)
    return json_files

def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_outputs.py <golden_dir> <new_dir>")
        print("Example: python compare_outputs.py golden_outputs/ reference_outputs/")
        sys.exit(1)

    golden_dir = Path(sys.argv[1])
    new_dir = Path(sys.argv[2])

    if not golden_dir.exists():
        print(f"Error: Golden directory not found: {golden_dir}")
        sys.exit(1)

    if not new_dir.exists():
        print(f"Error: New directory not found: {new_dir}")
        sys.exit(1)

    print("=" * 80)
    print("SIMULATION OUTPUT COMPARISON")
    print("=" * 80)
    print(f"Golden outputs: {golden_dir}")
    print(f"New outputs:    {new_dir}")
    print()

    # Find all JSON files in both directories
    golden_files = find_json_files(golden_dir)
    new_files = find_json_files(new_dir)

    print(f"Found {len(golden_files)} files in golden directory")
    print(f"Found {len(new_files)} files in new directory")
    print()

    # Files only in golden
    only_golden = set(golden_files.keys()) - set(new_files.keys())
    if only_golden:
        print(f"⚠️  Files only in golden ({len(only_golden)}):")
        for f in sorted(only_golden)[:10]:
            print(f"  - {f}")
        if len(only_golden) > 10:
            print(f"  ... and {len(only_golden) - 10} more")
        print()

    # Files only in new
    only_new = set(new_files.keys()) - set(golden_files.keys())
    if only_new:
        print(f"⚠️  Files only in new ({len(only_new)}):")
        for f in sorted(only_new)[:10]:
            print(f"  - {f}")
        if len(only_new) > 10:
            print(f"  ... and {len(only_new) - 10} more")
        print()

    # Compare common files
    common_files = set(golden_files.keys()) & set(new_files.keys())
    print(f"Comparing {len(common_files)} common files...")
    print()

    results_by_model = defaultdict(list)
    perfect_matches = 0
    partial_matches = 0
    no_matches = 0
    errors = 0

    for rel_path in sorted(common_files):
        golden_path = golden_files[rel_path]
        new_path = new_files[rel_path]

        golden_data = load_json_file(golden_path)
        new_data = load_json_file(new_path)

        result = compare_responses(golden_data, new_data, rel_path)

        # Extract model name from path
        model_dir = Path(rel_path).parts[0] if Path(rel_path).parts else "unknown"
        results_by_model[model_dir].append((rel_path, result))

        if "error" in result:
            errors += 1
        elif result.get("match"):
            perfect_matches += 1
        elif result.get("match_percentage", 0) > 0:
            partial_matches += 1
        else:
            no_matches += 1

    # Print summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total files compared: {len(common_files)}")
    print(f"Perfect matches (100%): {perfect_matches}")
    print(f"Partial matches (>0%): {partial_matches}")
    print(f"No matches (0%): {no_matches}")
    print(f"Errors: {errors}")
    print()

    # Print results by model
    print("=" * 80)
    print("RESULTS BY MODEL")
    print("=" * 80)
    for model, results in sorted(results_by_model.items()):
        print(f"\n{model}:")
        perfect = sum(1 for _, r in results if r.get("match", False))
        total = len(results)
        print(f"  Perfect matches: {perfect}/{total} ({perfect/total*100:.1f}%)")

        # Show details for non-perfect matches
        for rel_path, result in results:
            if not result.get("match", False):
                print(f"  ⚠️  {Path(rel_path).name}:")
                if "error" in result:
                    print(f"     Error: {result['error']}")
                else:
                    print(f"     Match rate: {result.get('match_percentage', 0):.1f}%")
                    print(f"     Exact matches: {result.get('exact_matches', 0)}/{result.get('total_entries', 0)}")
                    if result.get('response_diffs'):
                        print(f"     First diff (entry {result['response_diffs'][0]['index']}):")
                        print(f"       Similarity: {result['response_diffs'][0]['similarity']:.2f}")

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)

    # Exit code based on results
    if perfect_matches == len(common_files) and len(only_golden) == 0 and len(only_new) == 0:
        print("✅ All outputs match perfectly!")
        sys.exit(0)
    else:
        print("⚠️  Some differences found. See details above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
