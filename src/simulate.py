import fcntl
import time
import os
import json
import pandas as pd
from contextlib import contextmanager

from .model import Model 
from .agent import Agent 

@contextmanager
def file_lock(file_path, timeout=5):
    """
    Cross-platform context manager for file locking with timeout.
    Returns None if lock cannot be acquired within timeout.
    """
    lock_file = file_path + ".lock"
    
    try:
        # Try to acquire lock (works on both Windows and Unix)
        lock_fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_RDWR)
        try:
            # Write PID to lock file for debugging
            os.write(lock_fd, str(os.getpid()).encode())
            yield True
        finally:
            os.close(lock_fd)
            try:
                os.unlink(lock_file)
            except FileNotFoundError:
                pass
                
    except FileExistsError:
        # Lock file exists, wait with timeout
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Try to acquire lock again
                lock_fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                try:
                    os.write(lock_fd, str(os.getpid()).encode())
                    yield True
                    return
                finally:
                    os.close(lock_fd)
                    try:
                        os.unlink(lock_file)
                    except FileNotFoundError:
                        pass
            except FileExistsError:
                time.sleep(0.1)  # Wait a bit before retrying
                
        # Timeout reached
        print(f"⏰ Could not acquire lock for {file_path} within {timeout}s - skipping")
        yield None


def run_simulation_random_response(config, data_file, n_users=1000, n_responses_per_user=1, output_path=None):
    """
    Version with file locking held during the entire simulation run.
    """
    if output_path:
        with file_lock(output_path) as lock_acquired:
            if not lock_acquired:
                print(f"🚫 Skipping {output_path} - another process is working on it")
                return []
            
            # All reading/writing MUST happen under the lock
            return _run_simulation_with_lock(config, data_file, n_users, n_responses_per_user, output_path)
    else:
        return _run_simulation_with_lock(config, data_file, n_users, n_responses_per_user, output_path)



def _run_simulation_with_lock(config, data_file, n_users, n_responses_per_user, output_path):
    """
    Internal function that runs the actual simulation logic.
    This is separated so the lock context manager can wrap the entire operation.
    """
    
    print(f"🚀 [DEBUG] Starting simulation with config: {config}")
    print(f"📁 [DEBUG] Data file: {data_file}")
    print(f"📊 [DEBUG] Target: {n_users} users, {n_responses_per_user} responses each")
    print(f"💾 [DEBUG] Output path: {output_path}")
    
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
                    row["retrieve_context"], row["OPPU"], row["n_style_examples"], row.get("with_persona", True))
                    for row in existing_results
                }
                print(f"📚 [DEBUG] Loaded {len(existing_results)} existing results")
                print(f"🔍 [DEBUG] Created {len(existing_predictions)} prediction keys for deduplication")
        except json.JSONDecodeError:
            print(f"[WARNING] Cache file {output_path} is empty or invalid JSON. Starting fresh.")

    # Ensure output directory exists before saving
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            print(f"📁 [DEBUG] Creating output directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
        
    print(f"📖 [DEBUG] Loading data from pickle file...")
    df = pd.read_pickle(data_file)
    print(f"📊 [DEBUG] Loaded {len(df)} total rows from pickle")
    
    # Filter the DataFrame to only include rows where training == 0
    df_filtered = df[df["training"] == 0]
    print(f"🎯 [DEBUG] Filtered to {len(df_filtered)} test rows (training == 0)")

    # Count how many rows each user has
    user_counts = df_filtered["username"].value_counts()
    print(f"👥 [DEBUG] Found {len(user_counts)} unique users in test data")

    # Filter users who have at least m entries
    eligible_users = user_counts[user_counts >= n_responses_per_user].index
    print(f"✅ [DEBUG] {len(eligible_users)} users have >= {n_responses_per_user} responses")

    # Sample n_users from the eligible users
    sampled_users = pd.Series(eligible_users).sample(n=min(n_users, len(eligible_users)), random_state=42)
    print(f"🎲 [DEBUG] Sampled {len(sampled_users)} users for processing")

    # For each sampled user, take m rows
    df_test = (
        df_filtered[df_filtered["username"].isin(sampled_users)]
        .reset_index(drop=True)  # Make sure 'username' is only a column
        .groupby("username", group_keys=False)
        .apply(lambda x: x.sample(n=n_responses_per_user, random_state=42))
        .reset_index(drop=True)
    )
    print(f"🎯 [DEBUG] Final test set: {len(df_test)} rows")

    model_loaded = False
    new_results = []  # Track only new results for efficient saving

    j = 0
    total_users = len(df_test.groupby("username"))
    
    for user_idx, (username, user_df) in enumerate(df_test.groupby("username"), 1):
        print(f"\n👤 [DEBUG] Processing user {user_idx}/{total_users}: {username} ({len(user_df)} samples)")
        
        try:
            agent = Agent(username, data_file)
            print(f"🤖 [DEBUG] Created agent for user {username}")
        except Exception as e:
            print(f"❌ [DEBUG] Failed to create agent for {username}: {e}")
            continue
        
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
                config["with_persona"],
            )

            if prediction_key in existing_predictions:
                print(f"⏭️  [DEBUG] Skipping existing prediction for {username} (sample {i}/{len(user_df)})")
                continue
            
            print(f"🔄 [DEBUG] Processing {username} sample {i}/{len(user_df)}")
            print(f"💬 [DEBUG] Original message: {original_message[:100]}...")
            
            if not model_loaded:
                print(f"🧠 [DEBUG] Loading model with config: {config}")
                try:
                    model = Model(config, finetuning_filepath=data_file)
                    model_loaded = True
                    print(f"✅ [DEBUG] Model loaded successfully")
                except Exception as e:
                    print(f"❌ [DEBUG] Failed to load model: {e}")
                    return results

            # Generate responses (returns a list of valid responses)
            print(f"🎯 [DEBUG] Generating response with {config['n_style_examples']} style examples, with_persona={config['with_persona']}...")
            try:
                responses = agent.generate_response(
                    llm=model,
                    n_examples=config["n_style_examples"],
                    retrieve_context_bool=config["retrieve_context"],
                    personalized_bool=config["OPPU"],
                    with_persona=config["with_persona"],
                    conversation_history=[reply_to],
                    n_candidates=20
                )
                print(f"📝 [DEBUG] Generated {len(responses) if responses else 0} valid responses")
            except Exception as e:
                print(f"❌ [DEBUG] Error generating response for {username}: {e}")
                continue

            # Skip if no valid responses
            if not responses:
                print(f"⚠️  [DEBUG] No valid responses generated for user `{username}` (msg: `{original_message[:50]}...`)")
                continue

            new_result = {
                "user": username,
                "persona": persona,
                "model": config["model"],
                "fine_tuned": config["finetuned"],
                "retrieve_context": config["retrieve_context"],
                "OPPU": config["OPPU"],
                "n_style_examples": config["n_style_examples"],
                "with_persona": config["with_persona"],
                "reply_to": reply_to,
                "original_message": original_message,
                "response": responses[0],
                "all_valid_responses": responses
            }
            
            results.append(new_result)
            new_results.append(new_result)
            print(f"✅ [DEBUG] Added result for {username}. Total results: {len(results)}")

            j += 1
            if j % 10 == 0:
                print(f"💾 [DEBUG] Saving progress after {j} new predictions...")
                print(f"📊 [DEBUG] Current totals: {len(results)} total results, {len(new_results)} new results")
                try:
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)
                    print(f"✅ [DEBUG] Progress saved successfully")
                except Exception as e:
                    print(f"❌ [DEBUG] Error saving progress: {e}")

    # Final save
    if output_path:
        print(f"\n💾 [DEBUG] Final save: {len(results)} total results ({len(new_results)} new)")
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"✅ All predictions completed! Saved final results to `{output_path}` 🎉")
        except Exception as e:
            print(f"❌ [DEBUG] Error in final save: {e}")

    return results