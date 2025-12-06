#!/usr/bin/env python3
"""Migrate existing results from legacy format to consolidated format.

This script converts existing postprocessing results without re-running
expensive operations. It preserves all data while reorganizing into the
new consolidated structure.

Usage:
    python migrate_results.py --input results --output results_consolidated
    python migrate_results.py --input results/results_bluesky --output results_consolidated --dataset bluesky
"""

import argparse
import sys
from pathlib import Path
from validation.format_converter import migrate_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Migrate existing results to consolidated format"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input directory with legacy results (e.g., results/results_bluesky)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results_consolidated',
        help='Output directory for consolidated results (default: results_consolidated)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        help='Dataset name (bluesky, twitter, reddit) - auto-detected if not specified'
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: Input directory not found: {input_path}")
        sys.exit(1)

    # Auto-detect dataset from path if not specified
    dataset = args.dataset
    if not dataset:
        if 'bluesky' in str(input_path).lower():
            dataset = 'bluesky'
        elif 'twitter' in str(input_path).lower():
            dataset = 'twitter'
        elif 'reddit' in str(input_path).lower():
            dataset = 'reddit'
        else:
            print("❌ Error: Could not auto-detect dataset. Please specify --dataset")
            sys.exit(1)

    print("="*60)
    print(f"MIGRATING {dataset.upper()} RESULTS")
    print("="*60)
    print(f"From: {input_path}")
    print(f"To:   {args.output}/{dataset}/")
    print()

    # Run migration
    migrate_dataset(str(input_path), args.output, dataset)

    print("\n" + "="*60)
    print("✅ MIGRATION COMPLETE")
    print("="*60)
    print(f"\nNew structure:")
    print(f"  {args.output}/{dataset}/shared/users.json")
    print(f"  {args.output}/{dataset}/shared/prompts.json")
    print(f"  {args.output}/{dataset}/responses/*.json")
    print(f"\nYou can now use the consolidated format with plotting scripts.")


if __name__ == "__main__":
    main()
