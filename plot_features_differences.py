import os
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from feature_utils import parse_filename  # assumes you have this to parse metadata

def compute_feature_differences(df, label_col='label'):
    """Compute difference in mean between label=1 and label=0 for each feature."""
    grouped = df.groupby(label_col).mean()
    if 0 in grouped.index and 1 in grouped.index:
        return (grouped.loc[0] - grouped.loc[1]).to_dict()
    else:
        return {}


def main(folder_path, label_col="label"):
    differences_list = []

    # Collect model names to define color palette
    model_names = set()

    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('validation_data_labelled_features_with_text_and_label.csv'):
                full_path = os.path.join(root, filename)
                df = pd.read_csv(full_path)

                model, ft, context, style, oppu = parse_filename(filename)
                model_names.add(model)

                # Filter only numeric features
                numeric_features = df.select_dtypes(include='number').drop(columns=[label_col])
                df_numeric = pd.concat([numeric_features, df[label_col]], axis=1)

                diffs = compute_feature_differences(df_numeric, label_col)
                if not diffs:
                    continue

                diffs.update({
                    'model': model,
                    'ft': ft,
                    'context': context,
                    'style': style,
                    'oppu': oppu
                })

                differences_list.append(diffs)

    if not differences_list:
        print("No differences computed. Exiting.")
        return

    # Create DataFrame
    diff_df = pd.DataFrame(differences_list)
    sort_cols = ['model', 'ft', 'context', 'style', 'oppu']
    diff_df = diff_df.set_index(sort_cols)

    # Add sorting based on L1 norm
    diff_df['abs_total_diff'] = diff_df.abs().sum(axis=1)

    diff_df_sorted = diff_df.sort_values(by='abs_total_diff')
    diff_df_features = diff_df_sorted.drop(columns='abs_total_diff')
    
    # EXCLUDE FEATURES
    features_to_exclude = ['spelling_grammar_errors', 'has_link', 'has_question_mark', 'has_exclamation_mark', 'has_emoji']  # replace with the features you want to skip
    diff_df_features = diff_df_features.drop(columns=features_to_exclude)

    # Assign colors by model
    unique_models = sorted(model_names)
    model_palette = sns.color_palette("tab10", len(unique_models))
    model_color_dict = dict(zip(unique_models, model_palette))

    # Plot one figure per feature
    for feature in diff_df_features.columns:
        # Extract all values for this feature across configurations
        feature_values = []
        labels = []
        colors = []

        for idx, row in diff_df_features.iterrows():
            model, ft, context, style, oppu = idx
            config_label = f"{model}_ft{ft}_ctx{context}_style{style}_oppu{oppu}"
            val = row[feature]
            feature_values.append((config_label, val, model))

        # Sort by absolute difference
        feature_values_sorted = sorted(feature_values, key=lambda x: abs(x[1]))

        # Unpack sorted values
        labels = [x[0] for x in feature_values_sorted]
        bars = [x[1] for x in feature_values_sorted]
        colors = [model_color_dict[x[2]] for x in feature_values_sorted]

        # Plot
        plt.figure(figsize=(14, 6))
        plt.bar(labels, bars, color=colors)
        plt.title(f"{feature} — Mean Difference (AI [label=0] - Human [label=1])", fontsize=16)
        plt.ylabel(f"Difference in {feature}", fontsize=14)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(folder_path, f"feature_diff_{feature}.png"), dpi=300)
        plt.close()

    # Reset index to get metadata as columns again
    heatmap_df = diff_df_features.copy().reset_index()

    # Combine model config into a single label for the row
    heatmap_df['config'] = heatmap_df.apply(
        lambda row: f"{row['model']}_ft{row['ft']}_ctx{row['context']}_style{row['style']}_oppu{row['oppu']}",
        axis=1
    )

    # Set index to config label and drop metadata
    heatmap_data = heatmap_df.drop(columns=['model', 'ft', 'context', 'style', 'oppu']).set_index('config')

    # Normalize each feature (column) using z-score
    scaler = StandardScaler()
    heatmap_data_normalized = pd.DataFrame(
        scaler.fit_transform(heatmap_data),
        index=heatmap_data.index,
        columns=heatmap_data.columns
    )

    # Plot heatmap
    plt.figure(figsize=(14, max(6, 0.3 * len(heatmap_data_normalized))))  # dynamic height
    sns.heatmap(
        heatmap_data_normalized,
        cmap='vlag',
        center=0,
        linewidths=0.5,
        linecolor='gray',
        robust=True 
    )
    plt.title("Normalized Feature Differences (AI [0] - Human [1]) per Configuration", fontsize=16)
    plt.xlabel("Feature")
    plt.ylabel("Model Configuration")
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, "feature_differences_heatmap.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(14, max(6, 0.3 * len(heatmap_data_normalized))))  # dynamic height
    sns.heatmap(
        heatmap_data_normalized,
        cmap='vlag',
        center=0,
        annot=heatmap_data.round(2),  # raw signed values
        fmt='g',
        cbar_kws={'label': 'Z-scored (per feature)'}
    )
    plt.title("Normalized Feature Differences (AI [0] - Human [1]) per Configuration", fontsize=16)
    plt.xlabel("Feature")
    plt.ylabel("Model Configuration")
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, "feature_differences_heatmap1.png"), dpi=300)
    plt.close()


        # Now analyze the effect of each binary configuration parameter on feature differences
    binary_flags = ['ft', 'context', 'style', 'oppu']

    for feature in diff_df_features.columns:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()  # To index subplots easily

        for i, flag in enumerate(binary_flags):
            data = []

            for idx, row in diff_df_features.iterrows():
                model, ft, context, style, oppu = idx
                val = row[feature]
                flags = {'ft': ft, 'context': context, 'style': style, 'oppu': oppu}
                data.append({
                    'model': model,
                    'value': val,
                    'abs_value': abs(val),
                    'flag': flags[flag]
                })

            df_flag = pd.DataFrame(data)

            sns.boxplot(data=df_flag, x='flag', y='abs_value', palette="Set2", ax=axes[i])
            axes[i].set_title(f"{flag} ON vs OFF")
            axes[i].set_xlabel(f"{flag}")
            axes[i].set_ylabel(f"|Diff in {feature}|")

        plt.suptitle(f"Impact of Config Flags on |Mean Difference| for {feature}", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(folder_path, f"feature_impact_grouped_{feature}.png"), dpi=300)
        plt.close()

    toggles = ['ft', 'context', 'style', 'oppu']
    feature_names = diff_df_features.columns

    for toggle in toggles:
        diffs_matrix = []
        row_labels = []

        # Get actual toggle values (e.g., 0 and 10 for 'style')
        toggle_vals = sorted(diff_df_features.index.get_level_values(toggle).unique())
        if len(toggle_vals) != 2:
            print(f"Skipping toggle '{toggle}': expected 2 distinct values, got {toggle_vals}")
            continue

        off_val, on_val = toggle_vals[0], toggle_vals[1]

        for idx1 in diff_df_features.index:
            idx1_dict = dict(zip(diff_df_features.index.names, idx1))

            if idx1_dict[toggle] == off_val:
                idx2_dict = idx1_dict.copy()
                idx2_dict[toggle] = on_val
                idx2 = tuple(idx2_dict[col] for col in diff_df_features.index.names)

                if idx2 in diff_df_features.index:
                    vec1 = diff_df_features.loc[idx1]
                    vec2 = diff_df_features.loc[idx2]
                    diff = vec2 - vec1  # Raw difference (ON - OFF)

                    diffs_matrix.append(diff.values)
                    label = f"{idx1_dict['model']}_ft{idx1_dict['ft']}_ctx{idx1_dict['context']}_style{idx1_dict['style']}_oppu{idx1_dict['oppu']} → {toggle}={on_val}"
                    row_labels.append(label)

        if diffs_matrix:
            diff_df = pd.DataFrame(diffs_matrix, columns=feature_names, index=row_labels)

            # Normalize each column (z-score)
            scaler = StandardScaler()
            diff_df_normalized = pd.DataFrame(
                scaler.fit_transform(diff_df),
                index=diff_df.index,
                columns=diff_df.columns
            )

            # Plot heatmap
            plt.figure(figsize=(14, max(6, 0.3 * len(diff_df_normalized))))
            sns.heatmap(
                diff_df_normalized,
                cmap="seismic",
                center=0,
                annot=diff_df.round(2),
                fmt='g',
                cbar_kws={'label': 'Z-scored (per feature)'}
            )
            plt.title(f"Effect of toggling '{toggle}' ({off_val} → {on_val}) on Feature Differences", fontsize=16)
            plt.ylabel("Paired Configs (OFF → ON)")
            plt.xlabel("Feature")
            plt.tight_layout()
            plt.savefig(os.path.join(folder_path, f"heatmap_toggle_{toggle}.png"), dpi=300)
            plt.close()

            # Use raw signed differences and compute symmetric color scale
            max_abs = np.abs(diff_df.values).max()
            
            plt.figure(figsize=(14, max(6, 0.3 * len(diff_df_normalized))))
            sns.heatmap(
                diff_df,
                cmap="seismic",             # or "RdBu_r", "coolwarm"
                center=0,
                vmin=-max_abs,
                vmax=max_abs,
                annot=diff_df.round(2),     # show raw diffs as annotations
                fmt='g',
                cbar_kws={'label': 'Signed Difference'}
            )
            plt.title(f"Effect of toggling '{toggle}' ({off_val} → {on_val}) on Feature Differences", fontsize=16)
            plt.ylabel("Paired Configs (OFF → ON)")
            plt.xlabel("Feature")
            plt.tight_layout()
            plt.savefig(os.path.join(folder_path, f"heatmap_toggle_{toggle}_2.png"), dpi=300)
            plt.close()

        else:
            print(f"No matching config pairs found for toggle: {toggle}")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute and visualize feature differences by label.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing CSV files")

    args = parser.parse_args()
    main(args.folder_path)
