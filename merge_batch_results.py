#!/usr/bin/env python3
"""
Merge batch results into single consolidated files.

This script:
1. Finds all batch result files for each (config, dataset) combination
2. Merges them into a single file
3. Removes batch suffixes from filenames

Usage:
    python merge_batch_results.py --results_dir results/
    python merge_batch_results.py --results_dir results/ --dry_run
"""

import os
import json
import argparse
import re
from pathlib import Path
from collections import defaultdict


def find_batch_files(results_dir: str) -> dict:
    """Find all batch result files and group them by base name."""
    results_path = Path(results_dir)

    # Pattern to match batch files: name__batch0.json, name__batch1.json, etc.
    batch_pattern = re.compile(r'^(.+)__batch(\d+)\.json$')

    # Group files by base name
    file_groups = defaultdict(list)

    for file_path in results_path.glob("**/*__batch*.json"):
        match = batch_pattern.match(file_path.name)
        if match:
            base_name = match.group(1)
            batch_id = int(match.group(2))
            # Store relative to results_dir
            rel_dir = file_path.parent.relative_to(results_path)
            file_groups[(rel_dir, base_name)].append((batch_id, file_path))

    return file_groups


def merge_batch_files(batch_files: list, output_path: Path, dry_run: bool = False) -> dict:
    """Merge multiple batch files into a single file."""
    # Sort by batch_id for deterministic ordering
    sorted_files = sorted(batch_files, key=lambda x: x[0])

    all_results = []
    stats = {
        "n_batches": len(sorted_files),
        "n_results_per_batch": [],
        "total_results": 0
    }

    for batch_id, file_path in sorted_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                batch_results = json.load(f)
                all_results.extend(batch_results)
                stats["n_results_per_batch"].append(len(batch_results))
        except json.JSONDecodeError as e:
            print(f"  ⚠️ Warning: Could not parse {file_path}: {e}")
            stats["n_results_per_batch"].append(0)
        except Exception as e:
            print(f"  ❌ Error reading {file_path}: {e}")
            stats["n_results_per_batch"].append(0)

    stats["total_results"] = len(all_results)

    if not dry_run:
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write merged results
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Merge batch results into single files"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing batch result files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for merged files (default: same as results_dir)"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be done without writing files"
    )
    parser.add_argument(
        "--delete_batches",
        action="store_true",
        help="Delete batch files after merging"
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir

    print("="*60)
    print("MERGING BATCH RESULTS")
    print("="*60)
    print(f"Results dir: {results_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Dry run: {args.dry_run}")
    print()

    # Find all batch files
    file_groups = find_batch_files(results_dir)

    if not file_groups:
        print("No batch files found.")
        return

    print(f"Found {len(file_groups)} file groups to merge\n")

    # Merge each group
    total_merged = 0
    batch_files_to_delete = []

    for (rel_dir, base_name), batch_files in sorted(file_groups.items()):
        output_path = output_dir / rel_dir / f"{base_name}.json"

        print(f"📁 {rel_dir / base_name}")
        print(f"   Batches: {len(batch_files)}")

        stats = merge_batch_files(batch_files, output_path, dry_run=args.dry_run)

        print(f"   Results per batch: {stats['n_results_per_batch']}")
        print(f"   Total results: {stats['total_results']}")

        if not args.dry_run:
            print(f"   ✅ Saved to: {output_path}")
            batch_files_to_delete.extend([f[1] for f in batch_files])
        else:
            print(f"   [DRY RUN] Would save to: {output_path}")

        total_merged += 1
        print()

    # Delete batch files if requested
    if args.delete_batches and not args.dry_run and batch_files_to_delete:
        print("🗑️ Deleting batch files...")
        for file_path in batch_files_to_delete:
            try:
                file_path.unlink()
                print(f"   Deleted: {file_path.name}")
            except Exception as e:
                print(f"   ⚠️ Could not delete {file_path}: {e}")

    print("="*60)
    print(f"MERGE COMPLETE: {total_merged} file groups processed")
    print("="*60)


if __name__ == "__main__":
    main()
