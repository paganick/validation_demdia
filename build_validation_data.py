import argparse
import pandas as pd
import os

def main():
    parser = argparse.ArgumentParser(
        description="Build a validation dataset from real human replies and AI-generated responses."
    )
    parser.add_argument(
        "--real_file",
        type=str,
        default="data/personas_and_tweets.df.pkl",
        help="Path to the pickle file containing real (human) tweet replies."
    )
    parser.add_argument(
        "--folder",
        type=str,
        #default="simulation_results10",
        help="Path to the folder containing AI-generated responses."
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=100,
        help="Number of samples to randomly take from each dataset."
    )
    args = parser.parse_args()

    # ===============================
    # Process Real Human Replies
    # ===============================
    try:
        df_real = pd.read_pickle(args.real_file)
    except Exception as e:
        print(f"Error loading real data file at {args.real_file}: {e}")
        return

    # Filter for human data. In our finetuning script, training==1 indicates human replies.
    if "training" in df_real.columns:
        df_real = df_real[df_real["training"] == 1]
    
    if "message" not in df_real.columns:
        print("Error: The real data file does not contain a 'message' column.")
        return

    # Create the expected columns: "text" (the reply) and "labels" (1 for human)
    df_real = df_real.copy()
    df_real["text"] = df_real["message"].astype(str)
    df_real["labels"] = 1

    # Randomly sample the desired number of human replies
    df_real_sampled = df_real.sample(n=args.sample_size, random_state=42)

    # ===============================
    # Process AI-Generated Replies
    # ===============================
    for root, _, files in os.walk(args.folder):
        for file in files:
            if file.endswith('random_response.csv'):
                csv_path = os.path.join(root, file)
                try:
                    df_ai = pd.read_csv(csv_path)
                except Exception as e:
                    print(f"Error loading AI data file at {csv_path}: {e}")
                if "response" not in df_ai.columns:
                    print(f"Error: The AI data file {csv_path} does not contain a 'response' column.")
                    return
                
                df_ai["text"] = df_ai["response"].astype(str)
                df_ai["labels"] = 0  # AI-generated

                # Sample AI responses
                df_ai_sampled = df_ai.sample(n=min(args.sample_size, len(df_ai)), random_state=42)

                # Combine AI and human samples
                df_validation = pd.concat(
                    [df_real_sampled[["text", "labels"]], df_ai_sampled[["text", "labels"]]],
                    ignore_index=True
                )

                # Shuffle dataset
                df_validation = df_validation.sample(frac=1, random_state=42).reset_index(drop=True)

                output_filename = csv_path.replace('.csv', '_validation_data.csv')
        
                # Save to CSV
                df_validation.to_csv(output_filename, index=False)
                print(f"Validation data saved to {output_filename}.")

if __name__ == "__main__":
    main() 