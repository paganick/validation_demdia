import pandas as pd
import json
import os

from .model import Model
from .agent import Agent

def run_simulation_random_response(config, n_users=1000, n_responses_per_user=1, output_path=None):
    DATA_FILE = "data/personas_and_tweets.df.pkl"

    results = []
    existing_predictions = set()
    # Return cached results if already exists
    if output_path and os.path.exists(output_path):
        print(f"[CACHE] Loading cached results from {output_path}")
        with open(output_path, encoding="utf-8") as f:
            existing_results = json.load(f)
            results.extend(existing_results) 
            existing_predictions = {
                (row["user"], row["original_message"], row["reply_to"], row["model"], row["fine_tuned"],
                row["retrieve_context"], row["OPPU"], row["n_style_examples"])
                for row in existing_results
            }
        
    df = pd.read_pickle(DATA_FILE)
    # Filter the DataFrame to only include rows where training == 0
    df_filtered = df[df["training"] == 0]

    # Count how many rows each user has
    user_counts = df_filtered["username"].value_counts()

    # Filter users who have at least m entries
    eligible_users = user_counts[user_counts >= n_responses_per_user].index

    # Sample n_users from the eligible users
    sampled_users = pd.Series(eligible_users).sample(n=n_users, random_state=42)

    # For each sampled user, take m rows
    df_test = (
        df_filtered[df_filtered["username"].isin(sampled_users)]
        .reset_index(drop=True)  # Make sure 'username' is only a column
        .groupby("username", group_keys=False)
        .apply(lambda x: x.sample(n=n_responses_per_user, random_state=42))
        .reset_index(drop=True)
    )

    model = Model(config)
    j = 0
    for username, user_df in df_test.groupby("username"):
        print(f"\n👤 Processing user: {username} ({len(user_df)} samples)")
        agent = Agent(username, DATA_FILE)
        
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
                print(f"⏩ Skipping already processed: {prediction_key}")
                continue
            

            response = agent.generate_response(model, 
                                               n_examples=config["n_style_examples"],
                                               retrieve_context_bool=config["retrieve_context"],
                                               personalized_bool=config['OPPU'],
                                               conversation_history=[reply_to], 
                                               num_candidates = 3)

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
                "response": response
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
