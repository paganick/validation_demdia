#!/usr/bin/env python3
"""
Anonymize usernames across all platforms.

This script:
1. Creates a consistent hash-based mapping for each username
2. Updates personas_llama.pkl and posts_llama.pkl with anonymized usernames
3. Updates persona text to replace username mentions
4. Saves a mapping file for researcher reference (keep secure!)

Usage:
    python anonymize_usernames.py
    python anonymize_usernames.py --platform twitter
"""

import pandas as pd
import hashlib
import re
import json
import argparse
from pathlib import Path
from datetime import datetime

# Configuration
DATA_DIR = Path("/home/nicpag/data/demdia_val_revision/data")
PLATFORMS = ["twitter", "reddit", "bluesky"]
SALT = "demdia_anonymization_2024"  # Change this for different anonymization runs


def generate_anonymous_id(username: str, platform: str) -> str:
    """Generate a consistent anonymous ID for a username.

    Uses SHA256 hash with salt for one-way, deterministic mapping.
    Returns format: User_XXXXXX (6 hex chars = 16M unique IDs)
    """
    # Combine username, platform, and salt for uniqueness
    combined = f"{platform}:{username}:{SALT}"
    hash_bytes = hashlib.sha256(combined.encode()).hexdigest()
    # Take first 6 characters of hash for readability
    short_hash = hash_bytes[:6].upper()
    return f"User_{short_hash}"


def get_username_prefix(platform: str) -> str:
    """Get the appropriate username prefix for each platform."""
    if platform == "reddit":
        return "u/"
    else:  # twitter, bluesky
        return "@"


def replace_username_in_text(text: str, old_username: str, new_username: str,
                              platform: str) -> str:
    """Replace username mentions in persona text."""
    if pd.isna(text) or not text:
        return text

    prefix = get_username_prefix(platform)
    old_full = f"{prefix}{old_username}"
    new_full = f"{prefix}{new_username}"

    # Replace the full username with prefix
    result = text.replace(old_full, new_full)

    # Also handle cases where username appears without prefix
    # (be careful not to replace partial matches)
    result = re.sub(
        rf'\b{re.escape(old_username)}\b',
        new_username,
        result
    )

    return result


def anonymize_platform(platform: str, dry_run: bool = False) -> dict:
    """Anonymize usernames for a single platform.

    Returns the username mapping.
    """
    print(f"\n{'='*60}")
    print(f"Processing {platform.upper()}")
    print(f"{'='*60}")

    personas_path = DATA_DIR / platform / "personas_llama.pkl"
    posts_path = DATA_DIR / platform / "posts_llama.pkl"

    # Check files exist
    if not personas_path.exists():
        print(f"  Warning: {personas_path} not found, skipping")
        return {}
    if not posts_path.exists():
        print(f"  Warning: {posts_path} not found, skipping")
        return {}

    # Load data
    print(f"  Loading personas from {personas_path}")
    df_personas = pd.read_pickle(personas_path)
    print(f"  Loading posts from {posts_path}")
    df_posts = pd.read_pickle(posts_path)

    print(f"  Personas shape: {df_personas.shape}")
    print(f"  Posts shape: {df_posts.shape}")

    # Get all unique usernames from both files
    personas_users = set(df_personas['username'].unique())
    posts_users = set(df_posts['username'].unique())
    all_users = personas_users.union(posts_users)

    print(f"  Unique users in personas: {len(personas_users)}")
    print(f"  Unique users in posts: {len(posts_users)}")
    print(f"  Total unique users: {len(all_users)}")

    # Create mapping
    username_mapping = {}
    for username in sorted(all_users):
        anon_id = generate_anonymous_id(username, platform)
        username_mapping[username] = anon_id

    # Check for collisions (very unlikely with 6 hex chars)
    anon_ids = list(username_mapping.values())
    if len(anon_ids) != len(set(anon_ids)):
        print("  WARNING: Hash collision detected! Increase hash length.")

    print(f"  Created {len(username_mapping)} username mappings")

    # Show sample mappings
    print(f"\n  Sample mappings:")
    prefix = get_username_prefix(platform)
    for i, (old, new) in enumerate(list(username_mapping.items())[:3]):
        print(f"    {prefix}{old} -> {prefix}{new}")

    if dry_run:
        print("\n  [DRY RUN - no files modified]")
        return username_mapping

    # Apply to personas
    print(f"\n  Anonymizing personas...")
    df_personas_anon = df_personas.copy()

    # Replace username column
    df_personas_anon['username'] = df_personas_anon['username'].map(username_mapping)

    # Replace usernames in persona text
    for col in ['persona', 'persona_third_person']:
        if col in df_personas_anon.columns:
            for old_user, new_user in username_mapping.items():
                df_personas_anon[col] = df_personas_anon[col].apply(
                    lambda x: replace_username_in_text(x, old_user, new_user, platform)
                )

    # Apply to posts
    print(f"  Anonymizing posts...")
    df_posts_anon = df_posts.copy()

    # Replace username column
    df_posts_anon['username'] = df_posts_anon['username'].map(username_mapping)

    # Save anonymized files
    print(f"\n  Saving anonymized files...")
    df_personas_anon.to_pickle(personas_path)
    print(f"    Saved: {personas_path}")

    df_posts_anon.to_pickle(posts_path)
    print(f"    Saved: {posts_path}")

    # Verify
    print(f"\n  Verification:")
    df_check = pd.read_pickle(personas_path)
    sample_users = df_check['username'].head(3).tolist()
    print(f"    Sample anonymized usernames: {sample_users}")

    return username_mapping


def save_mapping(all_mappings: dict, output_dir: Path):
    """Save the username mapping for researcher reference.

    WARNING: This file should be kept secure and not shared!
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mapping_file = output_dir / f"username_mapping_{timestamp}.json"

    with open(mapping_file, 'w') as f:
        json.dump(all_mappings, f, indent=2)

    print(f"\n{'='*60}")
    print(f"IMPORTANT: Username mapping saved to:")
    print(f"  {mapping_file}")
    print(f"\nKeep this file SECURE - it allows de-anonymization!")
    print(f"{'='*60}")

    return mapping_file


def main():
    parser = argparse.ArgumentParser(
        description="Anonymize usernames across all platforms"
    )
    parser.add_argument(
        "--platform",
        choices=PLATFORMS + ["all"],
        default="all",
        help="Platform to process (default: all)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without modifying files"
    )

    args = parser.parse_args()
    platforms = PLATFORMS if args.platform == "all" else [args.platform]

    print("="*60)
    print("USERNAME ANONYMIZATION")
    print("="*60)
    print(f"Platforms: {platforms}")
    print(f"Dry run: {args.dry_run}")

    all_mappings = {}

    for platform in platforms:
        mapping = anonymize_platform(platform, dry_run=args.dry_run)
        all_mappings[platform] = mapping

    # Save mapping file
    if not args.dry_run and all_mappings:
        save_mapping(all_mappings, DATA_DIR.parent)

    # Summary
    print("\n" + "="*60)
    print("ANONYMIZATION COMPLETE")
    print("="*60)

    total_users = sum(len(m) for m in all_mappings.values())
    print(f"\nTotal users anonymized: {total_users}")
    for platform, mapping in all_mappings.items():
        print(f"  {platform}: {len(mapping)} users")


if __name__ == "__main__":
    main()
