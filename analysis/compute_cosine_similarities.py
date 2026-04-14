import os
import json
import argparse
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Initialize the embedding model once
_sentence_model_path = os.environ.get("SENTENCE_MODEL_PATH", "sentence-transformers/all-MiniLM-L6-v2")
model = SentenceTransformer(_sentence_model_path)

def compute_similarity(a: str, b: str) -> float:
    """Compute cosine similarity between two strings."""
    embeddings = model.encode([a, b])
    return float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])

def process_file(filepath: Path):
    output_path = filepath.parent / f"{filepath.stem}_with_cosine_scores.json"
    if output_path.exists():
        print(f"✅ Skipping {filepath.name}: output already exists at {output_path.name}")
        return

    print(f"\n📂 Processing file: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []

    for entry in tqdm(data, desc=f"🔍 Processing {filepath.name}", unit="entry"):
        try:
            original = entry["original_message"]
            response = entry["response"]
            valid_responses = entry["all_valid_responses"]

            entry_result = {k: entry[k] for k in entry if k not in ["response", "all_valid_responses"]}
            entry_result["response"] = response
            entry_result["original_vs_response_similarity"] = compute_similarity(original, response)

            entry_result["valid_response_similarities"] = []
            for candidate in valid_responses:
                sim = compute_similarity(original, candidate)
                entry_result["valid_response_similarities"].append({
                    "candidate": candidate,
                    "similarity": sim
                })

            results.append(entry_result)
        except Exception as e:
            print(f"❌ Skipping entry due to error: {e}")
            continue

    print(f"💾 Saving results to {output_path}")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def process_folder(input_folder: Path):
    print(f"📁 Walking through folder: {input_folder}")

    for subdir, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith("random_response.json"):
                in_path = Path(subdir) / file
                process_file(in_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSON files and compute cosine similarities.")
    parser.add_argument("input_folder", type=str, help="Path to the folder containing input JSON files.")
    args = parser.parse_args()

    process_folder(Path(args.input_folder))
