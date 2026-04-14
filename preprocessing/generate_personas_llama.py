#!/usr/bin/env python3
"""
Generate persona descriptions using Llama-3.1-8B-Instruct.

This script reads the existing personas.pkl files, extracts user posts,
and generates new persona descriptions using Llama-3.1-8B-Instruct instead of GPT-4.

Output files (per platform):
    - {platform}_posts_llama.pkl: Contains user posts with columns:
        - username: User identifier (anonymized username)
        - message: The user's post text
        - reply_to: Parent content being replied to
        - training: 1 for training set, 0 for test set

    - {platform}_personas_llama.pkl: Contains persona descriptions with columns:
        - username: User identifier (anonymized username)
        - persona: Llama-generated persona description

Usage:
    python generate_personas_llama.py --platform reddit
    python generate_personas_llama.py --platform twitter
    python generate_personas_llama.py --platform bluesky
    python generate_personas_llama.py --platform all
    python generate_personas_llama.py --platform reddit --max-posts 30 --max-users 10
"""

import os
import re
import argparse
import random
import pandas as pd
import torch
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration (DATA_DIR can be overridden via --data-dir CLI argument)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
PLATFORMS = ["twitter", "reddit", "bluesky"]

# Generation settings (tuned for quality)
TEMPERATURE = 0.4  # Lower = more deterministic, less drift
MAX_NEW_TOKENS = 300  # Shorter = less opportunity to drift
TOP_P = 0.9
MAX_RETRIES = 3  # Retry failed generations


def get_username_prefix(platform: str) -> str:
    """Get the appropriate username prefix for each platform."""
    if platform == "reddit":
        return "u/"
    else:  # twitter, bluesky
        return "@"


def load_model():
    """Load the Llama model and tokenizer."""
    print(f"Loading model: {MODEL_NAME}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=False,
        legacy=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    print("Model loaded successfully")
    return model, tokenizer


def check_persona_quality(persona: str, username: str, prefix: str) -> tuple:
    """
    Check if a generated persona meets quality standards.

    Returns:
        Tuple of (is_valid, issues_list)
    """
    issues = []
    full_username = f"{prefix}{username}"

    # Check 1: Must start with username
    if not persona.startswith(full_username):
        issues.append("wrong_start")

    # Check 2: Not too short (at least 100 chars of actual content)
    if len(persona) < 100:
        issues.append("too_short")

    # Check 3: No bullet points (indicates post content leak)
    if '\n-' in persona or persona.count('- ') >= 3:
        issues.append("has_bullets")

    # Check 4: Should be third person, not first person heavy
    words = persona.split()
    first_person_count = len(re.findall(r'\bI\b|\bI\'m\b|\bI\'ve\b|\bmy\b', persona))
    if len(words) > 0 and first_person_count / len(words) > 0.02:
        issues.append("first_person")

    # Check 5: Should have persona-like language
    persona_indicators = ['passionate', 'engaged', 'interested', 'vocal', 'active',
                          'style', 'tone', 'tends to', 'often', 'personality',
                          'perspective', 'views', 'opinions', 'characteristic']
    has_persona_language = any(ind in persona.lower() for ind in persona_indicators)
    if not has_persona_language and len(persona) > 150:
        issues.append("no_persona_language")

    # Check 6: No excessive questions
    if persona.count('?') >= 3:
        issues.append("has_questions")

    is_valid = len(issues) == 0
    return is_valid, issues


def generate_persona_llama(username: str, messages: list, platform: str,
                           model, tokenizer, max_posts: int = 30) -> str:
    """
    Generate persona description for a user using Llama-3.1-8B-Instruct.

    Args:
        username: Username (without prefix)
        messages: List of user's message texts
        platform: Platform name (reddit, twitter, bluesky)
        model: Loaded Llama model
        tokenizer: Loaded tokenizer
        max_posts: Maximum number of posts to include in analysis (default: 30)

    Returns:
        str: Persona description starting with appropriate format
    """
    prefix = get_username_prefix(platform)
    full_username = f"{prefix}{username}"

    # Sample messages if we have more than max_posts (get diverse sample)
    if len(messages) > max_posts:
        # Take first few, last few, and random from middle for diversity
        n_edges = min(5, max_posts // 3)
        n_random = max_posts - (2 * n_edges)
        middle_msgs = messages[n_edges:-n_edges] if len(messages) > 2*n_edges else []

        sampled = (
            messages[:n_edges] +
            (random.sample(middle_msgs, min(n_random, len(middle_msgs))) if middle_msgs else []) +
            messages[-n_edges:]
        )
        limited_messages = sampled[:max_posts]
    else:
        limited_messages = messages[:max_posts]

    # Build the posts list for the prompt (numbered, not bulleted)
    posts_text = "\n".join(
        f"{i+1}. {msg[:200]}" for i, msg in enumerate(limited_messages)
        if msg and pd.notna(msg)
    )

    # Enhanced system prompt
    system_prompt = """You are a social media researcher who writes concise persona descriptions.

CRITICAL RULES:
1. Write in THIRD PERSON only (e.g., "They are...", "This user tends to...")
2. DO NOT copy, quote, or paraphrase any posts
3. DO NOT use bullet points or lists
4. DO NOT ask questions
5. Write 2-3 flowing paragraphs describing personality, interests, and communication style
6. Focus on patterns and traits, not specific post content"""

    # Few-shot example for the model
    example_persona = f"""Example of a GOOD persona description:
@ExampleUser is a politically engaged individual with progressive views and a strong interest in social justice issues. They communicate in a direct, sometimes passionate tone, often sharing news articles and commentary on current events.

Their writing style is informal but articulate, frequently using humor and sarcasm to make points. They appear well-informed about policy matters and enjoy engaging in substantive debates. They show particular interest in environmental issues, workers' rights, and democratic processes.

Overall, @ExampleUser comes across as an active community member who values informed discourse and isn't afraid to voice strong opinions while remaining open to discussion."""

    user_prompt = f"""{example_persona}

---

Now analyze this {platform.title()} user. Based on their posts below, write a similar persona description starting with "{full_username} is".

Remember: NO bullet points, NO quotes from posts, NO first person, just flowing paragraphs describing who this person is.

Posts from {full_username}:
{posts_text}

Persona description for {full_username}:"""

    # Format as chat template
    messages_chat = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Try generation with retries
    for attempt in range(MAX_RETRIES):
        # Apply chat template
        prompt = tokenizer.apply_chat_template(
            messages_chat,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate with tuned settings
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE + (attempt * 0.1),  # Slightly increase temp on retries
                do_sample=True,
                top_p=TOP_P,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        # Clean up the response
        response = response.strip()

        # Remove any leading quotes or artifacts
        if response.startswith('"'):
            response = response[1:]
        if response.startswith("Persona description"):
            response = response.split(":", 1)[-1].strip()

        # Ensure it starts with the username
        if not response.startswith(full_username):
            if response.lower().startswith(full_username.lower()):
                response = full_username + response[len(full_username):]
            elif response.lower().startswith("this user"):
                response = full_username + response[9:]
            elif response.lower().startswith("they are"):
                response = full_username + " is" + response[8:]
            else:
                response = f"{full_username} is {response}"

        # Quality check
        is_valid, issues = check_persona_quality(response, username, prefix)

        if is_valid:
            return response
        elif attempt < MAX_RETRIES - 1:
            # Will retry
            continue
        else:
            # Last attempt, return with warning
            print(f"    Warning: {username} persona has issues after {MAX_RETRIES} attempts: {issues}")
            return response

    return response


def process_platform(platform: str, model, tokenizer,
                     data_dir: Path,
                     max_posts: int = 30, max_users: int = 0) -> tuple:
    """
    Process a single platform and generate llama personas.

    Args:
        platform: Platform name
        model: Loaded model
        tokenizer: Loaded tokenizer
        data_dir: Root data directory (contains {platform}/posts.pkl)
        max_posts: Maximum posts per user for persona generation
        max_users: Maximum users to process (0 = all)

    Returns:
        Tuple of (posts_df, personas_df)
    """
    filepath = data_dir / platform / "posts.pkl"
    print(f"\nProcessing {platform}...")
    print(f"  Loading posts: {filepath}")

    # Load posts data (has username, message, reply_to, training columns)
    df = pd.read_pickle(filepath)
    print(f"  Shape: {df.shape}")
    print(f"  Unique users: {df['username'].nunique()}")

    # Get unique users
    unique_users = df['username'].unique().tolist()

    if max_users > 0:
        unique_users = unique_users[:max_users]
        print(f"  Limited to {max_users} users")

    # Keep posts dataframe for output
    posts_df = df[df['username'].isin(unique_users)].copy()

    # Generate personas for each user
    personas_list = []

    for username in tqdm(unique_users, desc=f"  Generating personas for {platform}"):
        # Get training messages for this user
        user_df = df[(df['username'] == username) & (df['training'] == 1)]
        training_messages = user_df['message'].dropna().tolist()

        if len(training_messages) == 0:
            print(f"  Warning: No training messages for {username}, skipping")
            continue

        # Generate persona
        persona = generate_persona_llama(
            username=username,
            messages=training_messages,
            platform=platform,
            model=model,
            tokenizer=tokenizer,
            max_posts=max_posts
        )

        personas_list.append({
            'username': username,
            'persona': persona
        })

    personas_df = pd.DataFrame(personas_list)

    return posts_df, personas_df


def main(args):
    print("=" * 60)
    print("Persona Generation with Llama-3.1-8B-Instruct")
    print("=" * 60)

    # Determine which platforms to process
    if args.platform == "all":
        platforms_to_process = PLATFORMS
    else:
        platforms_to_process = [args.platform]

    # Load model once
    model, tokenizer = load_model()

    data_dir = Path(args.data_dir)

    # Process each platform
    for platform in platforms_to_process:
        posts_df, personas_df = process_platform(
            platform=platform,
            model=model,
            tokenizer=tokenizer,
            data_dir=data_dir,
            max_posts=args.max_posts,
            max_users=args.max_users
        )

        # Save output files
        output_dir = data_dir / platform
        output_dir.mkdir(parents=True, exist_ok=True)

        personas_output = output_dir / "personas_llama.pkl"
        personas_df.to_pickle(personas_output)

        print(f"\n  Saved personas to: {personas_output}")
        print(f"    Shape: {personas_df.shape}")

        # Print sample persona
        if len(personas_df) > 0:
            print(f"\n  Sample persona for {platform}:")
            sample = personas_df.iloc[0]
            print(f"    Username: {sample['username']}")
            print(f"    Persona (first 300 chars): {sample['persona'][:300]}...")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate persona descriptions using Llama-3.1-8B-Instruct"
    )
    parser.add_argument(
        "--platform",
        type=str,
        default="all",
        choices=["twitter", "reddit", "bluesky", "all"],
        help="Platform to process (default: all)"
    )
    parser.add_argument(
        "--max-posts",
        type=int,
        default=30,
        help="Maximum number of posts per user for persona generation (default: 30)"
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=0,
        help="Maximum number of users to process per platform (0 = all, default: 0)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DATA_DIR),
        help="Root data directory containing {platform}/posts.pkl (default: data/)"
    )

    args = parser.parse_args()
    main(args)
