import fcntl
import signal
import time
import os
import json
import pandas as pd
import torch
import random
import numpy as np
from contextlib import contextmanager

from .model import Model
from .agent import Agent


# ---------------------------------------------------------------------------
# SIGTERM handling: ensure lock files are cleaned up when SLURM kills the job
# ---------------------------------------------------------------------------
_active_lock_files = set()


def _sigterm_handler(signum, frame):
    """Handle SIGTERM by cleaning up lock files, then re-raising as SystemExit."""
    for lock_file in list(_active_lock_files):
        try:
            os.unlink(lock_file)
            print(f"🧹 Cleaned up lock file on SIGTERM: {lock_file}")
        except FileNotFoundError:
            pass
    raise SystemExit(f"Terminated by signal {signum}")


signal.signal(signal.SIGTERM, _sigterm_handler)


def set_seed(seed: int):
    """
    Set all random seeds for full reproducibility.
    Based on the comprehensive seed setting from utils.py Validator class.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # CUDA determinism settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Enable full CUDA determinism
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    # Note: torch.use_deterministic_algorithms(True) can cause issues with some operations
    # We'll try to enable it but catch any errors
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as e:
        print(f"⚠️ Could not enable full deterministic algorithms: {e}")
        print("   Some operations may still be non-deterministic")

    # Set transformers seed if available
    try:
        from transformers import set_seed as transformers_set_seed
        transformers_set_seed(seed)
    except ImportError:
        pass

    print(f"🎲 Random seed set to {seed} for reproducibility") 

@contextmanager
def file_lock(file_path, timeout=5):
    """
    Cross-platform context manager for file locking with timeout.
    Returns None if lock cannot be acquired within timeout.
    """
    lock_file = file_path + ".lock"
    
    # Ensure the directory exists before trying to create the lock file
    lock_dir = os.path.dirname(lock_file)
    if lock_dir and not os.path.exists(lock_dir):
        os.makedirs(lock_dir, exist_ok=True)
    
    def _acquire_and_yield(lock_file):
        lock_fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_RDWR)
        _active_lock_files.add(lock_file)
        try:
            # Write PID to lock file for debugging
            os.write(lock_fd, str(os.getpid()).encode())
            yield True
        finally:
            os.close(lock_fd)
            _active_lock_files.discard(lock_file)
            try:
                os.unlink(lock_file)
            except FileNotFoundError:
                pass

    try:
        yield from _acquire_and_yield(lock_file)
    except FileExistsError:
        # Lock file exists, wait with timeout
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                yield from _acquire_and_yield(lock_file)
                return
            except FileExistsError:
                time.sleep(0.1)  # Wait a bit before retrying

        # Timeout reached
        print(f"⏰ Could not acquire lock for {file_path} within {timeout}s - skipping")
        yield None


def run_simulation_random_response(config, dataset, posts_file, personas_file=None, n_users=1000, n_responses_per_user=1, output_path=None, seed=42, batch_users=None, fill_incomplete=False):
    """
    Version with file locking held during the entire simulation run.

    Args:
        config: Model configuration dictionary
        dataset: Dataset name ('twitter', 'reddit', 'bluesky')
        posts_file: Path to posts pickle file (contains username, message, reply_to, training)
        personas_file: Path to personas pickle file (contains username, persona).
        n_users: Number of users to sample (ignored if batch_users is provided)
        n_responses_per_user: Number of responses to generate per user
        output_path: Path to save results
        seed: Random seed for reproducibility (default: 42)
        batch_users: List of usernames to process (if None, sample n_users randomly)
    """
    # Set seed at the very beginning for full reproducibility
    set_seed(seed)

    if output_path:
        with file_lock(output_path) as lock_acquired:
            if not lock_acquired:
                print(f"🚫 Skipping {output_path} - another process is working on it")
                return []

            # All reading/writing MUST happen under the lock
            return _run_simulation_with_lock(config, dataset, posts_file, personas_file, n_users, n_responses_per_user, output_path, seed, batch_users, fill_incomplete)
    else:
        return _run_simulation_with_lock(config, dataset, posts_file, personas_file, n_users, n_responses_per_user, output_path, seed, batch_users, fill_incomplete)



def _run_simulation_with_lock(config, dataset, posts_file, personas_file, n_users, n_responses_per_user, output_path, seed=42, batch_users=None, fill_incomplete=False):
    """
    Internal function that runs the actual simulation logic.
    This is separated so the lock context manager can wrap the entire operation.

    Args:
        posts_file: Path to posts pickle file
        personas_file: Path to personas pickle file
        seed: Random seed for reproducibility
        batch_users: List of usernames to process (if None, sample n_users randomly)
    """

    print(f"🚀 [DEBUG] Starting simulation with config: {config}")
    print(f"🎲 [DEBUG] Using seed: {seed}")
    print(f"📁 [DEBUG] Dataset: {dataset}")
    print(f"📁 [DEBUG] Posts file: {posts_file}")
    print(f"📁 [DEBUG] Personas file: {personas_file}")
    if batch_users:
        print(f"📦 [DEBUG] Processing batch of {len(batch_users)} specific users")
    else:
        print(f"📊 [DEBUG] Target: {n_users} users, {n_responses_per_user} responses each")
    print(f"💾 [DEBUG] Output path: {output_path}")
    
    results = []
    existing_predictions = set()

    # Number of valid candidate responses expected per sample.
    # Entries with fewer than this are treated as incomplete and regenerated.
    _N_CANDIDATES = 20

    # Return cached results if already exists
    if output_path and os.path.exists(output_path):
        print(f"[CACHE] Loading cached results from {output_path}")
        try:
            with open(output_path, encoding="utf-8") as f:
                existing_results = json.load(f)

            def _pred_key(row):
                return (row["user"], row["original_message"], row["reply_to"], row["model"],
                        row["fine_tuned"], row["retrieve_context"], row["n_style_examples"],
                        row.get("persona", True))

            # fill_incomplete=True: drop entries with < _N_CANDIDATES valid responses
            # so they get regenerated (used for deliberate fill-in runs).
            # fill_incomplete=False (default): treat every existing entry as done —
            # a sample is only re-attempted if it has no entry at all.
            if fill_incomplete:
                complete_results = [
                    row for row in existing_results
                    if len(row.get("all_valid_responses", [])) >= _N_CANDIDATES
                ]
                partial_results = [
                    row for row in existing_results
                    if len(row.get("all_valid_responses", [])) < _N_CANDIDATES
                ]
                if partial_results:
                    print(f"⚠️  [DEBUG] Dropping {len(partial_results)} partial entries "
                          f"(<{_N_CANDIDATES} valid responses) — will regenerate")
            else:
                complete_results = existing_results
                partial_results = []

            results.extend(complete_results)
            existing_predictions = {_pred_key(row) for row in complete_results}
            print(f"📚 [DEBUG] Loaded {len(existing_results)} existing results "
                  f"({len(complete_results)} complete, {len(partial_results)} partial dropped)")
            print(f"🔍 [DEBUG] Created {len(existing_predictions)} prediction keys for deduplication")
        except json.JSONDecodeError:
            print(f"[WARNING] Cache file {output_path} is empty or invalid JSON. Starting fresh.")

    # Build set of users that already have all required responses — skip them entirely.
    from collections import Counter as _Counter
    _user_entry_counts = _Counter(r.get("user", "") for r in results)
    completed_users = {u for u, cnt in _user_entry_counts.items() if cnt >= n_responses_per_user}
    if completed_users:
        print(f"✅ [DEBUG] {len(completed_users)} users already have {n_responses_per_user} entries — will skip", flush=True)

    # Ensure output directory exists before saving
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            print(f"📁 [DEBUG] Creating output directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
        
    print(f"📖 [DEBUG] Loading data from pickle files...")
    df = pd.read_pickle(posts_file)
    print(f"📊 [DEBUG] Loaded {len(df)} total rows from posts file")

    # Load personas if separate file provided
    df_personas = None
    if personas_file is not None:
        df_personas = pd.read_pickle(personas_file)
        print(f"📊 [DEBUG] Loaded {len(df_personas)} personas from personas file")
    
    # Filter the DataFrame to only include rows where training == 0
    df_filtered = df[df["training"] == 0]
    print(f"🎯 [DEBUG] Filtered to {len(df_filtered)} test rows (training == 0)")

    # Count how many rows each user has
    user_counts = df_filtered["username"].value_counts()
    print(f"👥 [DEBUG] Found {len(user_counts)} unique users in test data")

    # Filter users who have at least m entries
    eligible_users = user_counts[user_counts >= n_responses_per_user].index
    print(f"✅ [DEBUG] {len(eligible_users)} users have >= {n_responses_per_user} responses")

    # Determine which users to process
    if batch_users is not None:
        # Use specific batch users (filter to eligible only)
        sampled_users = pd.Series([u for u in batch_users if u in eligible_users])
        print(f"📦 [DEBUG] Using {len(sampled_users)} users from batch (filtered to eligible)")
    else:
        # Sample n_users from the eligible users
        sampled_users = pd.Series(eligible_users).sample(n=min(n_users, len(eligible_users)), random_state=seed)
        print(f"🎲 [DEBUG] Sampled {len(sampled_users)} users for processing")

    # For each sampled user, take m rows
    df_test = (
        df_filtered[df_filtered["username"].isin(sampled_users)]
        .reset_index(drop=True)  # Make sure 'username' is only a column
        .groupby("username", group_keys=False)
        .apply(lambda x: x.sample(n=n_responses_per_user, random_state=seed))
        .reset_index(drop=True)
    )
    print(f"🎯 [DEBUG] Final test set: {len(df_test)} rows")

    model_loaded = False
    new_results = []  # Track only new results for efficient saving

    j = 0
    total_users = len(df_test.groupby("username"))
    
    for user_idx, (username, user_df) in enumerate(df_test.groupby("username"), 1):
        print(f"\n👤 [DEBUG] Processing user {user_idx}/{total_users}: {username} ({len(user_df)} samples)")

        if username in completed_users:
            print(f"⏭️  [DEBUG] Skipping {username}: already has {_user_entry_counts[username]}/{n_responses_per_user} entries", flush=True)
            continue

        try:
            agent = Agent(username, dataset, posts_file, personas_file)
            print(f"🤖 [DEBUG] Created agent for user {username}")
        except Exception as e:
            print(f"❌ [DEBUG] Failed to create agent for {username}: {e}")
            continue
        
        for i, row in enumerate(user_df.itertuples(index=False), start=1):
            reply_to = row.reply_to
            original_message = row.message

            # Get persona from separate file, or from the posts row if no personas file
            if df_personas is not None:
                user_persona = df_personas[df_personas['username'] == username]
                persona = user_persona['persona'].iloc[0] if not user_persona.empty else None
            else:
                persona = row.persona if hasattr(row, 'persona') else None

            prediction_key = (
                username,
                original_message,
                reply_to,
                config["model"],
                config["finetuned"],
                config["retrieve_context"],
                config["n_style_examples"],
                config["persona"],
            )

            if prediction_key in existing_predictions:
                print(f"⏭️  [DEBUG] Skipping existing prediction for {username} (sample {i}/{len(user_df)})")
                continue
            
            print(f"🔄 [DEBUG] Processing {username} sample {i}/{len(user_df)}")
            print(f"💬 [DEBUG] Original message: {original_message[:100]}...")
            
            if not model_loaded:
                print(f"🧠 [DEBUG] Loading model with config: {config}")
                try:
                    # Pass sampled users to enable user-filtered fine-tuning
                    model = Model(config, dataset, posts_file=posts_file, users=list(sampled_users))
                    model_loaded = True
                    print(f"✅ [DEBUG] Model loaded successfully")
                except Exception as e:
                    print(f"❌ [DEBUG] Failed to load model: {e}")
                    return results

            # Generate responses (returns valid responses + all rejected ones)
            # Use a unique seed per user+sample combination for reproducibility
            sample_seed = seed + user_idx * 1000 + i
            print(f"🎯 [DEBUG] Generating response with {config['n_style_examples']} style examples, persona={config['persona']}, seed={sample_seed}...")
            try:
                responses, rejected = agent.generate_response(
                    llm=model,
                    n_examples=config["n_style_examples"],
                    retrieve_context_bool=config["retrieve_context"],
                    with_persona=config["persona"],
                    instruction_tuned=config["instruction_tuned"],
                    conversation_history=[reply_to],
                    n_candidates=20,
                    seed=sample_seed,
                    temperature=config.get("temperature", 0.8),
                    top_p=config.get("top_p", 0.9),
                    top_k=config.get("top_k", 50),
                )
                print(f"📝 [DEBUG] Generated {len(responses) if responses else 0} valid responses, {len(rejected)} rejected")
                # Clear CUDA cache after generation
                torch.cuda.empty_cache()

                # Append rejected responses to sidecar JSONL (one line per sample)
                if rejected and output_path:
                    rejected_path = output_path.replace(".json", ".rejected.jsonl")
                    rejected_record = {
                        "user": username,
                        "reply_to": reply_to,
                        "original_message": original_message,
                        "sample_seed": sample_seed,
                        "n_valid": len(responses),
                        "rejected": rejected,
                    }
                    with open(rejected_path, "a", encoding="utf-8") as rf:
                        rf.write(json.dumps(rejected_record, ensure_ascii=False) + "\n")

            except Exception as e:
                print(f"❌ [DEBUG] Error generating response for {username}: {e}")
                torch.cuda.empty_cache()  # Also clear on error
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
                "n_style_examples": config["n_style_examples"],
                "persona": config["persona"],
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