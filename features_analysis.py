import os
import argparse
import pandas as pd
from feature_utils import extract_features, evaluate_features_single_dataset
from plotting_utils import parse_filename

def compute_features_for_all(folder_path):
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('validation_data.csv') and ('cosine' in filename or 'ml' in filename):
                full_path = os.path.join(root, filename)
                print(f'Computing features for {full_path}.')

                df = pd.read_csv(full_path)
                df['text'] = df['text'].fillna('').astype(str)

                feature_cache_path = full_path.replace(".csv", "_features.csv")
                _ = extract_features(df, cache_path=feature_cache_path)
                print(f'Saved features to {feature_cache_path}')


def evaluate_all_datasets(folder_path, label_source):
    assert label_source in ['labels', 'bert_prediction'], "label_source must be 'labels' or 'bert_prediction'"

    results_auc = []
    results_importances = []
    results_correlation = []

    suffix = "_from_bert" if label_source == "bert_prediction" else "_from_labels"

    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('validation_data_labelled.csv') and ('cosine' in filename or 'ml' in filename):
                full_path = os.path.join(root, filename)
                print(f'Evaluating {full_path}.')

                df = pd.read_csv(full_path)
                df['text'] = df['text'].fillna('').astype(str)

                feature_cache_path = full_path.replace("_labelled.csv", "_features.csv")
                base_output_path = full_path.replace(".csv", f"{suffix}")
                stats_output_path = base_output_path + "_feature_importance_stats.csv"
                correlation_output_path = base_output_path + "_feature_correlation_stats.csv"

                # Determine source type (cosine or ml)
                source_type = "cosine" if "cosine" in filename else "ml"

                # Evaluate from cached features
                auc, feature_importance, correlation_df = evaluate_features_single_dataset(
                    df, feature_cache_path, label_source=label_source
                )

                model, ft, context, style, oppu = parse_filename(filename)

                # Save AUC summary
                results_auc.append({
                    'model': model,
                    'ft': ft,
                    'context': context,
                    'style': style,
                    'oppu': oppu,
                    'auc': auc,
                    'label_source': label_source,
                    'source_type': source_type
                })

                # Save feature importance
                feature_dict = feature_importance.to_dict()
                feature_dict.update({
                    'model': model,
                    'ft': ft,
                    'context': context,
                    'style': style,
                    'oppu': oppu,
                    'label_source': label_source,
                    'source_type': source_type
                })
                results_importances.append(feature_dict)
                pd.DataFrame([feature_dict | {'auc': auc}]).to_csv(stats_output_path, index=False)
                print(f'Saved stats to {stats_output_path}')

                # Save correlation stats
                correlation_df.to_csv(correlation_output_path, index=False)
                print(f'Saved correlation stats to {correlation_output_path}')
                correlation_df['source_type'] = source_type
                results_correlation.append(correlation_df)

    # Save combined outputs
    pd.DataFrame(results_auc).to_csv(os.path.join(folder_path, f'auc_results{suffix}.csv'), index=False)
    pd.DataFrame(results_importances).to_csv(os.path.join(folder_path, f'importances{suffix}.csv'), index=False)
    pd.concat(results_correlation, ignore_index=True).to_csv(os.path.join(folder_path, f'correlation_results{suffix}.csv'), index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline to compute features and evaluate feature importance.")
    subparsers = parser.add_subparsers(dest="command")

    compute_parser = subparsers.add_parser("compute_features")
    compute_parser.add_argument("folder_path", type=str, help="Path to the folder with CSV files.")

    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("folder_path", type=str, help="Path to the folder with CSV files.")
    eval_parser.add_argument("label_source", type=str, choices=["labels", "bert_prediction"], help="Which label source to use.")

    args = parser.parse_args()

    if args.command == "compute_features":
        compute_features_for_all(args.folder_path)
    elif args.command == "evaluate":
        evaluate_all_datasets(args.folder_path, args.label_source)
