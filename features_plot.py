import os
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plotting_utils import parse_filename, make_label

def main(folder_path, label_source):
    assert label_source in ['labels', 'bert_prediction'], "label_source must be 'labels' or 'bert_prediction'"
    suffix = "_from_labels_feature_importance_stats.csv" if label_source == "labels" else "_from_bert_feature_importance_stats.csv"
    title_label = "Ground Truth" if label_source == "labels" else "BERT Predictions"
    plot_suffix = "from_labels" if label_source == "labels" else "from_bert"

    results_auc = []
    results_importances = []
    results_folder = os.path.join(folder_path, 'features')
    os.makedirs(results_folder, exist_ok=True)

    # Step 1: Load all matching _feature_labels_stats.csv files
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(suffix):
                full_path = os.path.join(root, filename)
                df = pd.read_csv(full_path)

                auc = df['auc'].iloc[0]
                model = df['model'].iloc[0]
                ft = df['ft'].iloc[0]
                context = df['context'].iloc[0]
                style = df['style'].iloc[0]
                oppu = df['oppu'].iloc[0]

                results_auc.append({
                    'model': model,
                    'ft': ft,
                    'context': context,
                    'style': style,
                    'oppu': oppu,
                    'auc': auc
                })

                feature_cols = [col for col in df.columns if col not in ['model', 'ft', 'context', 'style', 'oppu', 'auc']]
                feature_dict = df[feature_cols].iloc[0].to_dict()
                feature_dict.update({
                    'model': model,
                    'ft': ft,
                    'context': context,
                    'style': style,
                    'oppu': oppu
                })
                results_importances.append(feature_dict)

    # Step 2: Create summary DataFrames
    auc_df = pd.DataFrame(results_auc)
    importance_df = pd.DataFrame(results_importances)

    sort_columns = ['model', 'ft', 'context', 'style', 'oppu']
    auc_df_sorted = auc_df.sort_values(by=sort_columns, ascending=[True, False, False, False, False])
    importance_df_sorted = importance_df.sort_values(by=sort_columns, ascending=[True, False, False, False, False])

    # Step 3: Sort by AUC
    auc_df_sorted_by_auc = auc_df_sorted.sort_values(by='auc', ascending=True)
    auc_order = auc_df_sorted_by_auc.index
    importance_df_sorted_by_auc = importance_df_sorted.loc[auc_order]
    importance_df_indexed = importance_df_sorted_by_auc.set_index(sort_columns)

    # Step 4: Coloring by model
    unique_models = importance_df_sorted_by_auc['model'].unique()
    model_palette = sns.color_palette("tab10", len(unique_models))
    model_color_dict = dict(zip(unique_models, model_palette))
    row_colors = importance_df_sorted_by_auc['model'].map(model_color_dict)

    # Step 5: Plot AUC bar plot
    fig, ax = plt.subplots(figsize=(14, 6))
    labels = [
        make_label(row['model'], row['ft'], row['context'], row['style'], row['oppu'])
        for _, row in auc_df_sorted_by_auc.iterrows()
    ]
    bars = ax.bar(labels, auc_df_sorted_by_auc['auc'], color=[model_color_dict[m] for m in auc_df_sorted_by_auc['model']])
    ax.set_ylabel('AUC', fontsize=14)
    ax.set_title(f'AUC Score per Dataset (Ordered by AUC) — {title_label}', fontsize=16)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.grid(True, axis='y')

    handles = [plt.Line2D([0], [0], marker='o', color='w', label=model,
                          markerfacecolor=color, markersize=10) for model, color in model_color_dict.items()]
    ax.legend(handles=handles, title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, f'auc_{plot_suffix}.png'), dpi=600, bbox_inches='tight')
    plt.show()

    # Step 6: Plot heatmap of feature importances
    fig, ax = plt.subplots(figsize=(14, max(6, 0.3 * len(importance_df_indexed))))  # dynamic height
    heatmap_data = importance_df_indexed.drop(columns=sort_columns, errors='ignore')
    heatmap_data = heatmap_data.select_dtypes(include=[float, int])  # only keep numeric columns
    # Reorder columns by mean importance across rows
    heatmap_data = heatmap_data[heatmap_data.mean().sort_values(ascending=False).index]
    heatmap_data = heatmap_data.div(heatmap_data.sum(axis=1), axis=0)
    sns.heatmap(heatmap_data,
            annot=True, cmap='viridis', fmt=".2f", ax=ax,
            cbar_kws={'label': 'Importance'}, linewidths=0.5, linecolor='gray')
    
    # sns.heatmap(importance_df_indexed.drop(columns=sort_columns, errors='ignore'),
    #             annot=True, cmap='viridis', fmt=".2f", ax=ax,
    #             cbar_kws={'label': 'Importance'}, linewidths=0.5, linecolor='gray')
    
    plt.title(f'Feature Importances across Datasets (AUC Ordered) — {title_label}', fontsize=16)
    plt.xlabel('Features', fontsize=14)
    plt.ylabel('Dataset (Sorted by AUC)', fontsize=14)
    ax.set_yticklabels(labels, rotation=0)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, f'feature_importance_{plot_suffix}.png'), dpi=600, bbox_inches='tight')
    plt.show()

    # Step 7: Plot heatmap of correlation signs (+1/-1)
    sign_suffix = "_from_labels_feature_correlation_stats.csv" if label_source == "labels" else "_from_bert_feature_correlation_stats.csv"
    results_signs = []


    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(sign_suffix):
                full_path = os.path.join(root, filename)
                df = pd.read_csv(full_path)

                # Parse the filename for metadata
                model, ft, context, style, oppu = parse_filename(filename)

                # Map sign string to +1/-1
                sign_map = {'positive': 1, 'negative': -1}
                df['sign'] = df['correlation_sign'].map(sign_map)

                feature_dict = df.set_index('feature')['sign'].to_dict()
                feature_dict.update({
                    'model': model,
                    'ft': ft,
                    'context': context,
                    'style': style,
                    'oppu': oppu
                })

                results_signs.append(feature_dict)


    sign_df = pd.DataFrame(results_signs)
    sign_df_sorted = sign_df.sort_values(by=sort_columns, ascending=[True, False, False, False, False])
    sign_df_ordered = sign_df_sorted.loc[auc_order]
    sign_df_indexed = sign_df_ordered.set_index(sort_columns)

    # Plot the +/-1 heatmap
    fig, ax = plt.subplots(figsize=(14, max(6, 0.3 * len(sign_df_indexed))))
    heatmap_signs = sign_df_indexed.drop(columns=sort_columns, errors='ignore')
    heatmap_signs = heatmap_signs[heatmap_data.columns]  # match same column order

    sns.heatmap(heatmap_signs,
                annot=True, cmap='coolwarm', fmt=".0f", ax=ax,
                cbar_kws={'label': 'Correlation Sign'}, center=0,
                linewidths=0.5, linecolor='gray', vmin=-1, vmax=1)

    plt.title(f'Correlation Signs of Features across Datasets — {title_label}', fontsize=16)
    plt.xlabel('Features', fontsize=14)
    plt.ylabel('Dataset (Sorted by AUC)', fontsize=14)
    ax.set_yticklabels(labels, rotation=0)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, f'feature_signs_{plot_suffix}.png'), dpi=600, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot AUC and feature importance heatmap from simulation results.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing the simulation results")
    parser.add_argument("label_source", type=str, choices=["labels", "bert_prediction"], help="Which label source to plot from")
    args = parser.parse_args()
    main(args.folder_path, args.label_source)
