#!/usr/bin/env python3
"""
Transform Llama persona descriptions from third person to second person using Llama-3.1-8B-Instruct.

The Llama-generated personas are in third person ("@username is...").
For instruction-tuned prompts, we need them in second person ("You are @username...").

This script:
1. Loads personas_llama.pkl for each platform
2. Uses LLM to transform each persona from 3rd person to 2nd person
3. Saves the data in the SAME FORMAT as the old personas.pkl:
   - 'persona' column: Second person (for instruction-tuned prompts) - TRANSFORMED
   - 'persona_third_person' column: Third person (for non-instruction-tuned prompts) - ORIGINAL

This matches the format expected by agent.py, so no code changes are needed.

Usage:
    python transform_llama_personas_to_second_person.py
    python transform_llama_personas_to_second_person.py --platform twitter
    python transform_llama_personas_to_second_person.py --platform all --show-examples 3
"""

import pandas as pd
import os
import re
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import hashlib

# Define paths
DATA_DIR = Path("/home/nicpag/data/demdia_val_revision/data")
PLATFORMS = ["twitter", "reddit", "bluesky"]
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"


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


def transform_to_second_person_llm(text: str, username: str, platform: str,
                                    model, tokenizer) -> str:
    """
    Transform a third-person persona description to second person using LLM.
    """
    if pd.isna(text) or not text:
        return text

    prefix = get_username_prefix(platform)
    full_username = f"{prefix}{username}"

    # Create the prompt
    system_prompt = """You are a text transformation assistant. Your task is to rewrite persona descriptions from third person to second person.

Rules:
1. Change "@username is" or "u/username is" to "You are @username," or "You are u/username,"
2. Change "They" to "You"
3. Change "Their" to "Your"
4. Change "their" to "your"
5. Change "they" to "you"
6. Change "them" to "you"
7. Change "themselves" to "yourself"
8. Keep the content exactly the same, only change the perspective
9. The result should address the user directly in second person
10. Output ONLY the transformed text, no explanations or additional text"""

    user_prompt = f"""Transform this persona description from third person to second person:

Original (third person):
{text}

Transformed (second person, addressing the user directly as "You"):"""

    # Format as chat template
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Apply chat template
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=len(text) + 200,  # Allow some extra tokens
            temperature=0.1,  # Low temperature for deterministic output
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    # Clean up the response
    response = response.strip()

    # If the response is empty or too short, fall back to simple replacement
    if len(response) < 50:
        print(f"Warning: LLM response too short for {username}, using fallback")
        response = simple_transform_to_second_person(text, username, platform)

    return response


def simple_transform_to_second_person(text: str, username: str, platform: str) -> str:
    """Simple regex-based fallback transformation from 3rd to 2nd person."""
    prefix = get_username_prefix(platform)
    full_username = f"{prefix}{username}"

    result = text

    # Handle opening pattern: "@username is a..." -> "You are @username, a..."
    result = re.sub(
        rf"^{re.escape(full_username)}\s+is\s+",
        f"You are {full_username}, ",
        result
    )

    # Handle "This user" references
    result = re.sub(r"\bThis user\b", "You", result)
    result = re.sub(r"\bthis user\b", "you", result)

    # Handle "@username comes across as" -> "You come across as"
    result = re.sub(
        rf"{re.escape(full_username)} comes across as",
        "You come across as",
        result
    )

    # Handle "@username's" -> "Your"
    result = re.sub(rf"{re.escape(full_username)}'s\b", "Your", result)

    # Handle "Overall, @username" -> "Overall, you"
    result = re.sub(
        rf"Overall,\s*{re.escape(full_username)}",
        "Overall, you",
        result,
        flags=re.IGNORECASE
    )

    # Basic replacements - order matters (longer patterns first)
    replacements = [
        (r"\bThey are\b", "You are"),
        (r"\bthey are\b", "you are"),
        (r"\bThey have\b", "You have"),
        (r"\bthey have\b", "you have"),
        (r"\bThey appear\b", "You appear"),
        (r"\bthey appear\b", "you appear"),
        (r"\bThey seem\b", "You seem"),
        (r"\bthey seem\b", "you seem"),
        (r"\bThey tend\b", "You tend"),
        (r"\bthey tend\b", "you tend"),
        (r"\bThey often\b", "You often"),
        (r"\bthey often\b", "you often"),
        (r"\bThey also\b", "You also"),
        (r"\bthey also\b", "you also"),
        (r"\bTheir\b", "Your"),
        (r"\btheir\b", "your"),
        (r"\bThemselves\b", "Yourself"),
        (r"\bthemselves\b", "yourself"),
        (r"\bThem\b", "You"),
        (r"\bthem\b", "you"),
        (r"\bThey\b", "You"),
        (r"\bthey\b", "you"),
    ]

    for pattern, replacement in replacements:
        result = re.sub(pattern, replacement, result)

    return result


def process_platform(platform: str, model, tokenizer, cache: dict) -> pd.DataFrame:
    """Process personas for a single platform.

    Transforms the data to match the old personas.pkl format:
    - 'persona': Second person (for instruction-tuned prompts)
    - 'persona_third_person': Third person (for non-instruction-tuned prompts)
    """
    filepath = DATA_DIR / platform / "personas_llama.pkl"

    print(f"\nProcessing {platform}...")

    if not filepath.exists():
        print(f"  Warning: {filepath} not found, skipping")
        return None

    print(f"  Loading: {filepath}")

    # Load the data
    df = pd.read_pickle(filepath)
    print(f"  Shape: {df.shape}")
    print(f"  Original columns: {list(df.columns)}")

    # Determine the source column for third person personas
    # If already transformed, 'persona_third_person' exists; otherwise use 'persona'
    if 'persona_third_person' in df.columns:
        print(f"  Already transformed previously, using existing persona_third_person")
        third_person_col = 'persona_third_person'
    else:
        print(f"  First transformation, original 'persona' column is third person")
        third_person_col = 'persona'

    # Transform each persona from 3rd person to 2nd person
    print(f"  Transforming {len(df)} personas to second person...")
    second_person_personas = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"  Transforming {platform}"):
        username = row['username']
        persona_3p = row[third_person_col]

        # Create cache key
        cache_key = hashlib.md5(f"{platform}:{username}:{persona_3p}".encode()).hexdigest()

        if cache_key in cache:
            transformed = cache[cache_key]
        else:
            transformed = transform_to_second_person_llm(
                persona_3p, username, platform, model, tokenizer
            )
            cache[cache_key] = transformed

        second_person_personas.append(transformed)

    # Reorganize columns to match old personas.pkl format:
    # - 'persona': Second person (transformed)
    # - 'persona_third_person': Third person (original)
    df['persona_third_person'] = df[third_person_col]  # Keep/copy third person
    df['persona'] = second_person_personas  # Overwrite with second person

    # Ensure column order: username, persona, persona_third_person
    df = df[['username', 'persona', 'persona_third_person']]

    # Save back to the same file
    df.to_pickle(filepath)
    print(f"  Saved modified file: {filepath}")
    print(f"  Final columns: {list(df.columns)}")
    print(f"    - 'persona': Second person (for instruction-tuned prompts)")
    print(f"    - 'persona_third_person': Third person (for non-instruction-tuned prompts)")

    return df


def show_examples(dataframes: dict, n: int = 2):
    """Show example transformations for each platform."""
    print("\n" + "=" * 70)
    print("EXAMPLE TRANSFORMATIONS")
    print("=" * 70)

    for platform, df in dataframes.items():
        if df is None:
            continue

        prefix = get_username_prefix(platform)
        print(f"\n--- {platform.upper()} ---")

        for i, row in df.head(n).iterrows():
            print(f"\nUser: {prefix}{row['username']}")
            print(f"\nTHIRD PERSON ('persona_third_person' - for non-instruction-tuned):")
            print(f"  {row['persona_third_person'][:400]}...")
            print(f"\nSECOND PERSON ('persona' - for instruction-tuned):")
            print(f"  {row['persona'][:400]}...")
            print("-" * 60)


def create_comparison_csv(dataframes: dict, output_path: Path, sample_size: int = 10):
    """Create a CSV comparing third and second person versions."""
    samples = []

    for platform, df in dataframes.items():
        if df is None:
            continue

        # Sample users
        n_samples = min(sample_size, len(df))
        sampled = df.sample(n=n_samples, random_state=42)

        prefix = get_username_prefix(platform)

        for _, row in sampled.iterrows():
            samples.append({
                "platform": platform,
                "username": f"{prefix}{row['username']}",
                "persona_second_person": row["persona"],  # 'persona' column is now 2nd person
                "persona_third_person": row["persona_third_person"]
            })

    # Create DataFrame and save
    sample_df = pd.DataFrame(samples)
    sample_df.to_csv(output_path, index=False)
    print(f"\nComparison CSV saved: {output_path}")
    print(f"Total samples: {len(sample_df)}")

    return sample_df


def main():
    parser = argparse.ArgumentParser(
        description="Transform Llama personas from third person to second person"
    )
    parser.add_argument(
        "--platform",
        choices=PLATFORMS + ["all"],
        default="all",
        help="Platform to process (default: all)"
    )
    parser.add_argument(
        "--show-examples",
        type=int,
        default=2,
        help="Number of examples to show per platform (default: 2)"
    )

    args = parser.parse_args()

    platforms = PLATFORMS if args.platform == "all" else [args.platform]

    print("=" * 70)
    print("TRANSFORMING LLAMA PERSONAS: Third Person -> Second Person (LLM)")
    print("=" * 70)

    # Load model
    model, tokenizer = load_model()

    # Cache for transformed personas
    cache = {}

    # Process each platform
    dataframes = {}
    for platform in platforms:
        dataframes[platform] = process_platform(platform, model, tokenizer, cache)

    # Show examples
    show_examples(dataframes, args.show_examples)

    # Create comparison CSV
    output_csv = DATA_DIR.parent / "llama_persona_transformation_samples.csv"
    create_comparison_csv(dataframes, output_csv)

    print("\n" + "=" * 70)
    print("TRANSFORMATION COMPLETE")
    print("=" * 70)
    print("\nThe personas_llama.pkl files now contain:")
    print("  - 'persona': Third person (for non-instruction-tuned prompts)")
    print("  - 'persona_second_person': Second person (for instruction-tuned prompts)")


if __name__ == "__main__":
    main()
