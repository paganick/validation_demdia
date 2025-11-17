import os
import json
from collections import defaultdict


def validate_random_response_file(filepath, N_USERS=250):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return {"status": "error", "reason": f"JSON decode error: {e}"}

    user_reply_counts = defaultdict(set)
    valid_response_counts = []

    for entry in data:
        user = entry.get("user")
        reply_to = entry.get("reply_to")
        responses = entry.get("all_valid_responses", [])

        if user and reply_to:
            user_reply_counts[user].add(reply_to)
        if isinstance(responses, list):
            valid_response_counts.append((user, reply_to, len(responses)))
        else:
            valid_response_counts.append((user, reply_to, 0))

    anomalies = []
    n_users = len(user_reply_counts)
    if n_users != N_USERS:
        anomalies.append(f"Expected {N_USERS} users, found {n_users}")

    for user, replies in user_reply_counts.items():
        if len(replies) != 20:
            anomalies.append(f"User {user} has {len(replies)} replies (expected 20)")

    for user, reply_to, count in valid_response_counts:
        if count != 20:
            anomalies.append(
                f"User {user} reply_to '{reply_to[:30]}...' has {count} responses (expected 20)"
            )

    return {
        "status": "ok" if not anomalies else "anomalies",
        "anomalies": anomalies,
        "num_users": n_users,
    }


def scan_directory(seed_folder, n_users=250):
    results = []
    for root, _, files in os.walk(seed_folder):
        for file in files:
            if file.endswith("random_response.json"):
                filepath = os.path.join(root, file)
                check_result = validate_random_response_file(filepath, n_users)
                results.append({
                    "subfolder": os.path.relpath(root, seed_folder),
                    "filename": file,
                    "status": check_result["status"],
                    "details": check_result.get("anomalies", check_result.get("reason", "")),
                    "num_users": check_result.get("num_users", 0),
                })
    return results


def print_summary(results):
    print("\n--- Detailed Validation Summary ---")
    for result in results:
        print(f"\nSubfolder: {result['subfolder']}")
        print(f"File: {result['filename']}")
        if result["status"] == "ok":
            print("✅ All fine")
        elif result["status"] == "error":
            print(f"❌ Error reading file: {result['details']}")
        else:
            print("❌ Anomalies found:")
            for anomaly in result["details"]:
                print(f"   - {anomaly}")

    print("\n--- Summary of Number of Users Processed ---")
    for result in results:
        print(f"{result['subfolder']}/{result['filename']}: {result['num_users']} users")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate random_response.json files.")
    parser.add_argument("seed_folder", help="Path to the seed folder")
    parser.add_argument("n_users", type=int, help="Expected number of users per file (default: 250)", default=250)
    args = parser.parse_args()

    result_summary = scan_directory(args.seed_folder, args.n_users)
    print_summary(result_summary)
