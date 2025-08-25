import pandas as pd
import json
import os

from .model import Model
from .agent import Agent

def run_simulation_random_response(config, data_file, n_users=1000, n_responses_per_user=1, output_path=None):

    results = []
    existing_predictions = set()

    # Return cached results if already exists
    if output_path and os.path.exists(output_path):
        print(f"[CACHE] Loading cached results from {output_path}")
        try:
            with open(output_path, encoding="utf-8") as f:
                existing_results = json.load(f)
                results.extend(existing_results)
                existing_predictions = {
                    (row["user"], row["original_message"], row["reply_to"], row["model"], row["fine_tuned"],
                    row["retrieve_context"], row["OPPU"], row["n_style_examples"])
                    for row in existing_results
                }
        except json.JSONDecodeError:
            print(f"[WARNING] Cache file {output_path} is empty or invalid JSON. Starting fresh.")

    # Ensure output directory exists before saving
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
    df = pd.read_pickle(data_file)
    # Filter the DataFrame to only include rows where training == 0
    df_filtered = df[df["training"] == 0]

    # Count how many rows each user has
    user_counts = df_filtered["username"].value_counts()

    # Filter users who have at least m entries
    eligible_users = user_counts[user_counts >= n_responses_per_user].index

    # Sample n_users from the eligible users
    sampled_users = pd.Series(eligible_users).sample(n=min(n_users, len(eligible_users)), random_state=42)

    # For each sampled user, take m rows
    df_test = (
        df_filtered[df_filtered["username"].isin(sampled_users)]
        .reset_index(drop=True)  # Make sure 'username' is only a column
        .groupby("username", group_keys=False)
        .apply(lambda x: x.sample(n=n_responses_per_user, random_state=42))
        .reset_index(drop=True)
    )

    model_loaded = False

    j = 0
    for username, user_df in df_test.groupby("username"):
        print(f"\n👤 Processing user: {username} ({len(user_df)} samples)")
        agent = Agent(username, data_file)
        
        for i, row in enumerate(user_df.itertuples(index=False), start=1):
            reply_to = row.reply_to
            original_message = row.message
            persona = row.persona

            prediction_key = (
                username,
                original_message,
                reply_to,
                config["model"],
                config["finetuned"],
                config["retrieve_context"],
                config["OPPU"],
                config["n_style_examples"],
            )

            if prediction_key in existing_predictions:
                continue
            
            if not model_loaded:
                model = Model(config, finetuning_filepath=data_file)
                model_loaded = True

            # Generate responses (returns a list of valid responses)
            responses = agent.generate_response(
                llm=model,
                n_examples=config["n_style_examples"],
                retrieve_context_bool=config["retrieve_context"],
                personalized_bool=config["OPPU"],
                conversation_history=[reply_to],
                n_candidates=20
            )

            # Skip if no valid responses
            if not responses:
                print(f"⚠️ No valid responses generated for user `{username}` (msg: `{original_message}`)")
                continue

            results.append({
                "user": username,
                "persona": persona,
                "model": config["model"],
                "fine_tuned": config["finetuned"],
                "retrieve_context": config["retrieve_context"],
                "OPPU": config["OPPU"],
                "n_style_examples": config["n_style_examples"],
                "reply_to": reply_to,
                "original_message": original_message,
                "response": responses[0],
                "all_valid_responses": responses
            })

            j += 1
            if j % 10 == 0:
                print(f"💾 Saving progress after {j} predictions...")
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

    # Final save
    print(f"\n✅ All predictions completed! Saving final results to `{output_path}` 🎉")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results




def run_simulation_political_affiliation(config, n_users=1000, output_path=None):
    return []
