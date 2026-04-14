#!/usr/bin/env python3
"""
Anonymize @mentions in posts.pkl files, keeping only the top 20 public figures.
Creates backups of original files before overwriting.

NOTE: This script was run once during dataset preparation and is provided for
documentation purposes only. The anonymized posts.pkl files are distributed
directly and do not need to be regenerated. Re-running this script requires
access to the original non-anonymized data, which is not publicly released.
"""

import pickle
import pandas as pd
import re
import shutil
from pathlib import Path

# Top 20 public figures to keep (from mentions_analysis.csv)
PUBLIC_FIGURES_TO_KEEP = {
    'elonmusk',
    'realDonaldTrump',
    'KamalaHarris',
    'danorlovsky7',
    'GOP',
    'POTUS',
    'JoeBiden',
    'PeteHegseth',
    'GuntherEagleman',
    'VP',
    'SenJoniErnst',
    'VivekGRamaswamy',
    'DefiyantlyFree',
    'JackPosobiec',
    'catturd2'
}

# Create case-insensitive lookup
PUBLIC_FIGURES_LOWER = {name.lower(): name for name in PUBLIC_FIGURES_TO_KEEP}

def anonymize_mention(match, mention_map):
    """Replace a mention with anonymized version if not a public figure."""
    username = match.group(1)
    username_lower = username.lower()

    # Keep public figures as-is
    if username_lower in PUBLIC_FIGURES_LOWER:
        return match.group(0)  # Return original @username

    # Anonymize others
    if username_lower not in mention_map:
        mention_map[username_lower] = f"@anon_user_{len(mention_map):05d}"

    return mention_map[username_lower]

def anonymize_text(text, mention_map):
    """Anonymize all mentions in text except public figures."""
    if pd.isna(text) or not isinstance(text, str):
        return text

    # Pattern to match @mentions
    pattern = r'@(\w+)'

    def replacer(match):
        return anonymize_mention(match, mention_map)

    return re.sub(pattern, replacer, text)

def process_posts_file(filepath, mention_map):
    """Process a single posts.pkl file."""
    print(f"\nProcessing {filepath}...")

    # Load data
    df = pickle.load(open(filepath, 'rb'))
    original_count = len(df)

    # Count mentions before
    mentions_before = 0
    for col in ['message', 'reply_to']:
        if col in df.columns:
            for text in df[col].dropna():
                mentions_before += len(re.findall(r'@(\w+)', str(text)))

    # Anonymize mentions in message and reply_to columns
    for col in ['message', 'reply_to']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: anonymize_text(x, mention_map))

    # Count mentions after (for verification)
    mentions_after = 0
    public_figure_mentions = 0
    anon_mentions = 0
    for col in ['message', 'reply_to']:
        if col in df.columns:
            for text in df[col].dropna():
                all_mentions = re.findall(r'@(\w+)', str(text))
                mentions_after += len(all_mentions)
                for m in all_mentions:
                    if m.lower() in PUBLIC_FIGURES_LOWER:
                        public_figure_mentions += 1
                    elif m.startswith('anon_user_'):
                        anon_mentions += 1

    print(f"  Rows: {original_count}")
    print(f"  Mentions before: {mentions_before}")
    print(f"  Mentions after: {mentions_after}")
    print(f"    - Public figures kept: {public_figure_mentions}")
    print(f"    - Anonymized: {anon_mentions}")

    return df

def main():
    platforms = ['twitter', 'reddit', 'bluesky']
    mention_map = {}  # Shared across all platforms for consistency

    print("=" * 60)
    print("ANONYMIZING MENTIONS IN POSTS FILES")
    print("=" * 60)
    print(f"\nPublic figures to keep ({len(PUBLIC_FIGURES_TO_KEEP)}):")
    for name in sorted(PUBLIC_FIGURES_TO_KEEP):
        print(f"  - {name}")

    for platform in platforms:
        filepath = Path(f'data/{platform}/posts.pkl')
        backup_path = Path(f'data/{platform}/posts_backup_nonanon.pkl')

        if not filepath.exists():
            print(f"\nSkipping {platform}: file not found")
            continue

        # Create backup
        print(f"\nCreating backup: {backup_path}")
        shutil.copy(filepath, backup_path)

        # Process file
        df = process_posts_file(filepath, mention_map)

        # Save updated file
        print(f"  Saving updated file: {filepath}")
        with open(filepath, 'wb') as f:
            pickle.dump(df, f)

    print("\n" + "=" * 60)
    print(f"Total unique usernames anonymized: {len(mention_map)}")
    print("=" * 60)
    print("\nDone! Backups saved as posts_backup_nonanon.pkl in each platform folder.")

if __name__ == '__main__':
    main()
