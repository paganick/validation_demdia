import os
import argparse
import pandas as pd
from feature_utils import evaluate_features_single_dataset, parse_filename

def main(folder_path, label_source):
    results_auc = []
    results_importances = []
    results_correlation = []

    assert label_source in ['labels', 'bert_prediction'], "label_source must be 'labels' or 'bert_prediction'"

    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('validation_data_labelled.csv'):
                full_path = os.path.join(root, filename)
                print(f'Processing {full_path}.')

                df = pd.read_csv(full_path)
                df['text'] = df['text'].fillna('').astype(str)

                # Build output suffix
                suffix = "_from_bert" if label_source == "bert_prediction" else "_from_labels"

                # Define paths
                feature_cache_path = full_path.replace(".csv", "_features.csv")
                base_output_path = full_path.replace(".csv", f"{suffix}")
                stats_output_path = base_output_path + "_feature_importance_stats.csv"
                correlation_output_path = base_output_path + "_feature_correlation_stats.csv"  # New path for correlation stats

                # Compute features, stats, and correlation
                auc, feature_importance, correlation_df = evaluate_features_single_dataset(df, feature_cache_path, label_source=label_source)

                # Parse filename info
                model, ft, context, style, oppu = parse_filename(filename)

                # Store AUC summary
                results_auc.append({
                    'model': model,
                    'ft': ft,
                    'context': context,
                    'style': style,
                    'oppu': oppu,
                    'auc': auc,
                    'label_source': label_source
                })

                # Store and save feature importances
                feature_dict = feature_importance.to_dict()
                feature_dict.update({
                    'model': model,
                    'ft': ft,
                    'context': context,
                    'style': style,
                    'oppu': oppu,
                    'label_source': label_source
                })
                results_importances.append(feature_dict)

                # Save current feature stats to CSV
                pd.DataFrame([feature_dict | {'auc': auc}]).to_csv(stats_output_path, index=False)
                print(f'Saved stats to {stats_output_path}')

                # Save correlation stats to CSV
                correlation_df.to_csv(correlation_output_path, index=False)
                print(f'Saved correlation stats to {correlation_output_path}')

                # Store correlation results for later aggregation (optional)
                results_correlation.append(correlation_df)

    # Optionally save aggregate results
    pd.DataFrame(results_auc).to_csv(os.path.join(folder_path, f'auc_results{suffix}.csv'), index=False)
    pd.DataFrame(results_importances).to_csv(os.path.join(folder_path, f'importances{suffix}.csv'), index=False)

    # Aggregate and save correlation stats (optional)
    correlation_combined_df = pd.concat(results_correlation, ignore_index=True)
    correlation_combined_df.to_csv(os.path.join(folder_path, f'correlation_results{suffix}.csv'), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate feature importance from validation datasets.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing the simulation results")
    parser.add_argument("label_source", type=str, choices=["labels", "bert_prediction"], help="Label source to evaluate against")
    args = parser.parse_args()
    main(args.folder_path, args.label_source)
