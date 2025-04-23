import argparse
import json
import os
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import BertTokenizer

# Try to import Empath to determine if it's available
try:
    from empath import Empath
    EMPATH_AVAILABLE = True
except ImportError:
    EMPATH_AVAILABLE = False
    print("Warning: Empath module is not installed. Empath validation will be skipped.")

# Import the Validator class from our utilities file
from utils import Validator

# Initialize the BERT tokenizer to be used throughout the script
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def find_validation_files(base_folder):
    validation_files = []
    for root, _, files in os.walk(base_folder):
        for file in files:
            if file.endswith("_validation_data.csv"):
                validation_files.append(os.path.join(root, file))
    return validation_files


def process_file(file_path, run_bert=True, run_empath=True):
    print(f"\nProcessing file: {file_path}")
    df = pd.read_csv(file_path)
    df['text'] = df['text'].astype(str)

    # =======================
    # Run BERT Validation
    # =======================
    if run_bert:
        print("Running BERT validation...")
        trainer, report, cm = Validator.bert_validate(df)

        report_file = file_path.replace('.json', '_bert_report.json')
        with open(report_file, "w") as f:
            json.dump(report, f, indent=4)
        print(f"BERT classification report saved to {report_file}")

        # Save confusion matrix values as CSV
        cm_file = file_path.replace('.csv', '_confusion_matrix.csv')
        cm_df = pd.DataFrame(cm)
        cm_df.to_csv(cm_file, index=False)
        print(f"BERT confusion matrix data saved to {cm_file}")

        print("Generating BERT predictions on a subsample of the validation dataset (100 rows) for quick testing...")

        df_sample = df.sample(n=100, random_state=42)
        test_dataset = Dataset.from_pandas(df_sample)

        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

        test_dataset = test_dataset.map(tokenize_function, batched=True)
        predictions_output = trainer.predict(test_dataset)
        preds = np.argmax(predictions_output.predictions, axis=-1)
        df_sample["bert_prediction"] = preds

        labelled_file = file_path.replace(".csv", "_labelled.csv")
        df_sample.to_csv(labelled_file, index=False)
        print(f"Labeled validation data (100 samples) saved to {labelled_file}")

    # =======================
    # Run Empath Validation
    # =======================
    if run_empath:
        if not EMPATH_AVAILABLE:
            print("Empath module not installed. Skipping.")
            return

        print("Running Empath validation...")
        significant_features, distance = Validator.empath_validate(df)

        features_file = file_path.replace('.csv', '_empath_significant_features.csv')
        significant_features.to_csv(features_file, index=False)
        print(f"Empath significant features saved to {features_file}")
        print(f"Euclidean distance between groups: {distance:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Validate multiple text datasets using BERT and/or Empath.")
    parser.add_argument(
        "--input_dir",
        type=str,
        defaul="simulation_results10",
        required=True,
        help="Directory containing subfolders with '_validation_data.csv' files."
    )
    parser.add_argument(
        "--validation",
        type=str,
        default="all",
        choices=["bert", "empath", "all"],
        help="Which validation to run."
    )
    args = parser.parse_args()

    run_bert = args.validation in ["bert", "all"]
    run_empath = args.validation in ["empath", "all"]

    validation_files = find_validation_files(args.input_dir)
    if not validation_files:
        print("No validation files found.")
        return

    for file_path in validation_files:
        process_file(file_path, run_bert, run_empath)


if __name__ == "__main__":
    main()
