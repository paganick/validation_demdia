"""
Generate persona descriptions for Reddit users using GPT-4.

This script processes individual user CSV files (from parse_reddit_data.py),
generates persona descriptions using GPT-4, and creates a consolidated dataset
suitable for personalized LLM training.

Process:
    1. Scan reddit_data/ directory for user folders
    2. For each user with sufficient posts (>= min_rows):
       a. Split data 80/20 into training/test sets
       b. Generate persona description from training posts using GPT-4
       c. Add username, persona, and training flag to dataset
    3. Concatenate all users into single pickle file

Persona generation:
    - Uses GPT-4 to analyze user's training posts
    - Creates description starting with "You are u/{username}, ..."
    - Based on writing style, topics, and patterns in posts
    - Limited to first 50 posts to stay within token limits

Output:
    personas.pkl with columns:
        - username: Reddit username
        - persona: GPT-4 generated description
        - message: The user's actual post text
        - reply_to: Parent content being replied to
        - training: 1 for training set, 0 for test set

This dataset enables:
    - Training personalized models that mimic specific users
    - Evaluating persona-based vs example-based personalization
    - Studying individual writing style variation

Usage:
    python aggregate_reddit_data.py \\
        --input_dir reddit_data/ \\
        --output_file personas.pkl \\
        --min_rows 100 \\
        --max_users 50

Requires:
    - OPENAI_API_KEY environment variable
    - GPT-4 API access
"""

import os
import pandas as pd
import random
import argparse
from pathlib import Path
from tqdm.auto import tqdm
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_persona(username, messages):
    """
    Generate persona description for a Reddit user using GPT-4.

    Analyzes user's posts to create a persona description that captures their
    writing style, interests, and personality. The description is designed to
    be used as a system prompt for LLM personalization.

    Args:
        username: Reddit username (without u/ prefix)
        messages: List of user's message texts (up to 50 used for analysis)

    Returns:
        str: Persona description starting with "You are u/{username}, ..."
        None: If GPT-4 API call fails
    """
    prompt = (
        f"You are analyzing Reddit users. Based on the following posts from u/{username}, "
        "write a short persona description **starting with**: "
        f"'You are u/{username}, ...'\n"
        "Do not include post text directly, only the persona description.\n\n"
        + "\n".join(f"- {msg}" for msg in messages[:50])
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a social media researcher."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=512
        )
        content = response.choices[0].message.content.strip()
        return content

    except Exception as e:
        print(f"❌ GPT call failed for {username}: {e}")
        return None


def process_csv_file(csv_path, username):
    """
    Process a single user's CSV file: generate persona and prepare dataset.

    Args:
        csv_path: Path to user's CSV file (from parse_reddit_data.py)
        username: Reddit username

    Returns:
        pd.DataFrame: Processed dataset with username, persona, message, reply_to, training columns
        None: If user has too few posts or persona generation fails
    """
    df = pd.read_csv(csv_path)

    if len(df) < args.min_rows:
        return None

    # Shuffle and assign training flag
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_index = int(len(df) * 0.8)
    df['training'] = 0
    df.loc[:split_index, 'training'] = 1

    # Use only training messages to create persona
    training_texts = df[df['training'] == 1]['text'].dropna().tolist()
    persona = generate_persona(username, training_texts)
    
    if persona is None:
        return None

    # Add metadata
    df['username'] = username         # actual Reddit handle
    df['persona'] = persona
    df.rename(columns={"text": "message"}, inplace=True)

    return df[['username', 'persona', 'message', 'reply_to', 'training']]


def main(args):
    csv_dir = Path(args.input_dir)
    all_dfs = []

    print(f"🔍 Scanning for user subfolders in {csv_dir}")
    user_folders = sorted([f for f in csv_dir.iterdir() if f.is_dir()])

    if args.max_users > 0:
        user_folders = user_folders[:args.max_users]

    for user_folder in tqdm(user_folders):
        username = user_folder.name
        csv_files = list(user_folder.glob("*.csv"))
        if not csv_files:
            continue

        # Assuming one CSV per user folder
        processed = process_csv_file(csv_files[0], username)
        if processed is not None:
            all_dfs.append(processed)

    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        output_path = Path(args.output_pickle)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_pickle(output_path)
        print(f"✅ Done! Pickle saved to: {output_path}")
    else:
        print("⚠️ No users with sufficient data were processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory containing user subfolders with .csv files")
    parser.add_argument("--output-pickle", type=str, default="aggregated_personas.pkl",
                        help="Where to save the final pickle file")
    parser.add_argument("--min-rows", type=int, default=100,
                        help="Minimum number of rows required to process a CSV file (default: 100)")
    parser.add_argument("--max-users", type=int, default=0,
                    help="Maximum number of users to process (0 = all)")
    args = parser.parse_args()
    main(args)
