import os
import csv
import argparse
import pandas as pd

def parse_filename(filename):
    """
    Extracts model, finetuning, context, style, and OPPU info from a filename.
    Converts them into booleans or integers for clean tabular use.
    """
    base = os.path.basename(filename)
    parts = base.split("__")
    model = parts[0]
    ft = 1 if parts[1] == "ft" else 0
    context = 1 if parts[2] == "ctx1" else 0
    style = int(parts[3].replace("style", ""))
    oppu = 1 if parts[4].startswith("OPPU") else 0
    return model, ft, context, style, oppu

def count_significant_features(filepath):
    """
    Returns number of data rows in the CSV (excluding header).
    """
    try:
        with open(filepath, 'r') as f:
            return sum(1 for line in f) - 1  # subtract header
    except Exception as e:
        print(f"Failed reading {filepath}: {e}")
        return 0

def read_confusion_matrix(filepath):
    """
    Reads the second and third rows of the confusion matrix CSV.
    Returns values as a dict: {'0-0': ..., '1-0': ..., '0-1': ..., '1-1': ...}
    """
    try:
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            row1 = next(reader)
            row2 = next(reader)
            return {
                '0-0': int(row1[0]),
                '1-0': int(row1[1]),
                '0-1': int(row2[0]),
                '1-1': int(row2[1]),
            }
    except Exception as e:
        print(f"Failed reading confusion matrix from {filepath}: {e}")
        return {'0-0': None, '1-0': None, '0-1': None, '1-1': None}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root_folder", type=str, help="Path to the root folder")
    args = parser.parse_args()

    records = {}

    for subdir, _, files in os.walk(args.root_folder):
        for file in files:
            full_path = os.path.join(subdir, file)
            if file.endswith("_significant_features.csv"):
                model, ft, context, style, oppu = parse_filename(file)
                key = (model, ft, context, style, oppu)

                count = count_significant_features(full_path)
                if key not in records:
                    records[key] = {}
                records[key]['significant_feature_count'] = count

            elif file.endswith("_confusion_matrix.csv"):
                model, ft, context, style, oppu = parse_filename(file)
                key = (model, ft, context, style, oppu)

                confusion = read_confusion_matrix(full_path)
                if key not in records:
                    records[key] = {}
                records[key].update(confusion)

    # Convert to DataFrame
    rows = []
    for key, metrics in records.items():
        model, ft, context, style, oppu = key
        row = {
            "model": model,
            "finetuning": ft,
            "context": context,
            "style": style,
            "OPPU": oppu,
            "significant_feature_count": metrics.get("significant_feature_count", 0),
            "0-0": metrics.get("0-0"),
            "1-0": metrics.get("1-0"),
            "0-1": metrics.get("0-1"),
            "1-1": metrics.get("1-1")
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    output_path = os.path.join(args.root_folder, "significant_features_summary.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved summary to {output_path}")

if __name__ == "__main__":
    main()
