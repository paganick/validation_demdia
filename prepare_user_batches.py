#!/usr/bin/env python3
"""
Prepare reproducible user batch assignments for parallel simulation.

This script:
1. Loads all users from each platform
2. Assigns them to batches of max BATCH_SIZE users
3. Saves assignments to a JSON file for reproducibility

The batch assignments are deterministic (sorted by username, then split).

Usage:
    python prepare_user_batches.py
    python prepare_user_batches.py --batch_size 50
"""

import pandas as pd
import json
import argparse
from pathlib import Path
from datetime import datetime

DATA_DIR = Path("/home/nicpag/data/demdia_val_revision/data")
PLATFORMS = ["twitter", "reddit", "bluesky"]
DEFAULT_BATCH_SIZE = 60


def create_batch_assignments(batch_size: int = DEFAULT_BATCH_SIZE) -> dict:
    """Create reproducible batch assignments for all platforms."""

    assignments = {
        "metadata": {
            "created": datetime.now().isoformat(),
            "batch_size": batch_size,
        },
        "platforms": {}
    }

    for platform in PLATFORMS:
        personas_path = DATA_DIR / platform / "personas.pkl"

        if not personas_path.exists():
            print(f"Warning: {personas_path} not found, skipping")
            continue

        df = pd.read_pickle(personas_path)

        # Sort users for reproducibility
        users = sorted(df['username'].unique().tolist())
        n_users = len(users)
        n_batches = (n_users + batch_size - 1) // batch_size

        # Create batches
        batches = []
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_users)
            batch_users = users[start_idx:end_idx]
            batches.append({
                "batch_id": i,
                "users": batch_users,
                "n_users": len(batch_users)
            })

        assignments["platforms"][platform] = {
            "total_users": n_users,
            "n_batches": n_batches,
            "batches": batches
        }

        print(f"{platform}: {n_users} users -> {n_batches} batches")

    return assignments


def main():
    parser = argparse.ArgumentParser(
        description="Prepare reproducible user batch assignments"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Maximum users per batch (default: {DEFAULT_BATCH_SIZE})"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="user_batches.json",
        help="Output file path (default: user_batches.json)"
    )

    args = parser.parse_args()

    print("="*60)
    print("PREPARING USER BATCH ASSIGNMENTS")
    print("="*60)
    print(f"Batch size: {args.batch_size}")
    print()

    assignments = create_batch_assignments(args.batch_size)

    # Save to file
    output_path = DATA_DIR.parent / args.output
    with open(output_path, 'w') as f:
        json.dump(assignments, f, indent=2)

    print(f"\nSaved to: {output_path}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    total_batches = sum(
        p["n_batches"] for p in assignments["platforms"].values()
    )
    print(f"Total batches across all platforms: {total_batches}")

    for platform, data in assignments["platforms"].items():
        print(f"  {platform}: {data['n_batches']} batches ({data['total_users']} users)")


if __name__ == "__main__":
    main()
