import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from feature_utils import parse_filename

def main(folder_path):
    results = []

    # Traverse and process files
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith("confusion_matrix.csv"):
                filepath = os.path.join(root, file)
                try:
                    df = pd.read_csv(filepath, header=0)
                    cm = df.values
                    tn, fp = cm[0, 0], cm[0, 1]
                    fn, tp = cm[1, 0], cm[1, 1]

                    total = tn + fp + fn + tp
                    correct = tn + tp
                    total_0 = tn + fp
                    correct_0 = tn

                    model, ft, context, style, oppu = parse_filename(file)

                    results.append({
                        "model": model,
                        "ft": ft,
                        "context": context,
                        "style": style,
                        "oppu": oppu,
                        "accuracy": correct / total,
                        "class_0_accuracy": correct_0 / total_0 if total_0 > 0 else None,
                        "label": f"{model}_ft{ft}_ctx{context}_style{style}_oppu{oppu}"
                    })
                except Exception as e:
                    print(f"Failed to process {file}: {e}")

    results_df = pd.DataFrame(results)

    # Plotting setup
    sns.set(style="whitegrid")
    palette = sns.color_palette("tab10")
    model_palette = {model: palette[i % len(palette)] for i, model in enumerate(results_df["model"].unique())}

    # Accuracy plot (sorted)
    results_df_sorted = results_df.sort_values(by="accuracy", ascending=True)
    results_df_sorted["label"] = pd.Categorical(
        results_df_sorted["label"], categories=results_df_sorted["label"], ordered=True
    )
    plt.figure(figsize=(14, 10))
    sns.barplot(
        x="label", y="accuracy", hue="model", data=results_df_sorted,
        dodge=False, palette=model_palette
    )
    plt.xticks(rotation=90)
    plt.title("Overall Accuracy per Configuration (Sorted)")
    min_val = results_df_sorted["accuracy"].min()
    plt.ylim(min_val - 0.02, 1.0)
    plt.tight_layout()
    plt.legend(loc='center right')
    plt.savefig(os.path.join(folder_path, "overall_accuracy_barplot.png"))
    plt.close()

    # Class-0 accuracy plot (sorted)
    results_df_sorted_c0 = results_df.sort_values(by="class_0_accuracy", ascending=True, na_position="last")
    results_df_sorted_c0["label"] = pd.Categorical(
        results_df_sorted_c0["label"], categories=results_df_sorted_c0["label"], ordered=True
    )
    plt.figure(figsize=(14, 10)) 
    sns.barplot(
        x="label", y="class_0_accuracy", hue="model", data=results_df_sorted_c0,
        dodge=False, palette=model_palette
    )
    plt.xticks(rotation=90)
    plt.title("Class-0 Accuracy per Configuration (Sorted)")
    min_val = results_df_sorted_c0["class_0_accuracy"].min()
    plt.ylim(min_val - 0.02, 1.0)
    plt.legend(loc='center right')
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, "class0_accuracy_barplot.png"))
    plt.close()

    # -------- Heatmap generation --------
    toggles = ['ft', 'context', 'style', 'oppu']
    metrics = ['accuracy', 'class_0_accuracy']

    for metric in metrics:
        diff_rows = []

        for model in results_df["model"].unique():
            # Base configurations with all toggles off
            base_config = results_df[
                (results_df["model"] == model) &
                (results_df["ft"] == 0) &
                (results_df["context"] == 0) &
                (results_df["style"] == 0) &
                (results_df["oppu"] == 0)
            ]
            if base_config.empty:
                continue
            base_metric = base_config.iloc[0][metric]
            row_id = f"{model}_0000"
            row = {"base": row_id}

            for toggle in toggles:
                # Create a modified config where one toggle is set to 1
                query = {
                    "model": model,
                    "ft": 0,
                    "context": 0,
                    "style": 0,
                    "oppu": 0
                }
                if (toggle == 'style'):
                    query[toggle] = 10
                else:
                    query[toggle] = 1

                modified = results_df[
                    (results_df["model"] == query["model"]) &
                    (results_df["ft"] == query["ft"]) &
                    (results_df["context"] == query["context"]) &
                    (results_df["style"] == query["style"]) &
                    (results_df["oppu"] == query["oppu"])
                ]
                if not modified.empty:
                    mod_metric = modified.iloc[0][metric]
                    if pd.notnull(base_metric) and pd.notnull(mod_metric):
                        row[toggle] = mod_metric - base_metric
                    else:
                        row[toggle] = None
                else:
                    row[toggle] = None

            diff_rows.append(row)

        diff_df = pd.DataFrame(diff_rows).set_index("base")
        diff_df = diff_df.astype(float)  # Ensure numerical dtype for heatmap
        sns.heatmap(diff_df, annot=True, cmap="RdBu_r", center=0, fmt=".3f")
        plt.figure(figsize=(8, max(6, len(diff_df) * 0.5)))
        sns.heatmap(diff_df, annot=True, cmap="RdBu_r", center=0, fmt=".3f")
        plt.title(f"Effect of Enabling Each Toggle on {metric}")
        plt.tight_layout()
        plt.savefig(os.path.join(folder_path, f"heatmap_toggle_effect_{metric}.png"))
        plt.close()

        # NEXT

        import itertools

        toggle_cols = ["ft", "context", "style", "oppu"]
        metric_diffs = {
            "accuracy": [],
            "class_0_accuracy": []
        }

        for metric in metric_diffs.keys():
            rows = []

            # Group by model (assuming comparisons are only meaningful within same model)
            for model_name, model_group in results_df.groupby("model"):
                # For all pairs of rows in that group
                for i, j in itertools.combinations(model_group.index, 2):
                    row_i = model_group.loc[i]
                    row_j = model_group.loc[j]

                    # Determine how many toggles differ
                    diff_mask = [row_i[toggle] != row_j[toggle] for toggle in toggle_cols]
                    diff_count = sum(diff_mask)

                    if diff_count == 1:
                        toggle_idx = diff_mask.index(True)
                        toggle_name = toggle_cols[toggle_idx]

                        # Identify OFF and ON configs
                        if row_i[toggle_name] == 0 and row_j[toggle_name] != 0 :
                            off_row, on_row = row_i, row_j
                        elif row_i[toggle_name] != 0 and row_j[toggle_name] == 0:
                            off_row, on_row = row_j, row_i
                        else:
                            continue  # skip non-binary or unclear cases

                        delta = on_row[metric] - off_row[metric]

                        label = (
                            f"{model_name}_ft{off_row['ft']}_ctx{off_row['context']}_"
                            f"oppu{off_row['oppu']}_style{off_row['style']}"
                        )

                        # Search if label is already in rows
                        found = False
                        for r in rows:
                            if r["base_label"] == label:
                                r[toggle_name] = delta
                                found = True
                                break

                        if not found:
                            new_row = {toggle: None for toggle in toggle_cols}
                            new_row[toggle_name] = delta
                            new_row["base_label"] = label
                            rows.append(new_row)

            # Create and plot heatmap
            diff_df = pd.DataFrame(rows).set_index("base_label").astype(float)
            plt.figure(figsize=(8, max(4, len(diff_df) * 0.5)))
            sns.heatmap(diff_df, annot=True, cmap="RdBu_r", center=0, fmt=".3f", cbar_kws={"label": f"Δ {metric}"})
            plt.title(f"Effect of toggling style/ft/context/oppu ON \nΔ {metric} = ON − OFF")
            plt.tight_layout()
            plt.savefig(os.path.join(folder_path, f"heatmap_toggles_{metric}.png"))
            plt.close()  


        base_style = 10

        # Only consider configs with style=10
        style10_df = results_df[results_df["style"] == base_style]

        for metric in metric_diffs.keys():
            rows = []

            # Group by model (assuming comparisons are only meaningful within same model)
            for model_name, model_group in style10_df.groupby("model"):
                # For all pairs of rows in that group
                for i, j in itertools.combinations(model_group.index, 2):
                    row_i = model_group.loc[i]
                    row_j = model_group.loc[j]

                    # Determine how many toggles differ
                    diff_mask = [row_i[toggle] != row_j[toggle] for toggle in toggle_cols]
                    diff_count = sum(diff_mask)

                    if diff_count == 1:
                        toggle_idx = diff_mask.index(True)
                        toggle_name = toggle_cols[toggle_idx]

                        # Identify OFF and ON configs
                        if row_i[toggle_name] == 0 and row_j[toggle_name] == 1:
                            off_row, on_row = row_i, row_j
                        elif row_i[toggle_name] == 1 and row_j[toggle_name] == 0:
                            off_row, on_row = row_j, row_i
                        else:
                            continue  # skip non-binary or unclear cases

                        delta = on_row[metric] - off_row[metric]

                        label = (
                            f"{model_name}_ft{off_row['ft']}_ctx{off_row['context']}_"
                            f"oppu{off_row['oppu']}_style{base_style}"
                        )

                        # Search if label is already in rows
                        found = False
                        for r in rows:
                            if r["base_label"] == label:
                                r[toggle_name] = delta
                                found = True
                                break

                        if not found:
                            new_row = {toggle: None for toggle in toggle_cols}
                            new_row[toggle_name] = delta
                            new_row["base_label"] = label
                            rows.append(new_row)

            # Create and plot heatmap
            diff_df = pd.DataFrame(rows).set_index("base_label").astype(float)
            plt.figure(figsize=(8, max(4, len(diff_df) * 0.5)))
            sns.heatmap(diff_df, annot=True, cmap="RdBu_r", center=0, fmt=".3f", cbar_kws={"label": f"Δ {metric}"})
            plt.title(f"Effect of toggling ft/context/oppu ON (style=10)\nΔ {metric} = ON − OFF")
            plt.tight_layout()
            plt.savefig(os.path.join(folder_path, f"heatmap_toggles_{metric}_style10.png"))
            plt.close()


        # Only consider configs with context=1
        base_context = 1
        context_df = style10_df[style10_df["context"] == base_context]

        for metric in metric_diffs.keys():
            rows = []

            # Group by model (assuming comparisons are only meaningful within same model)
            for model_name, model_group in context_df.groupby("model"):
                # For all pairs of rows in that group
                for i, j in itertools.combinations(model_group.index, 2):
                    row_i = model_group.loc[i]
                    row_j = model_group.loc[j]

                    # Determine how many toggles differ
                    diff_mask = [row_i[toggle] != row_j[toggle] for toggle in toggle_cols]
                    diff_count = sum(diff_mask)

                    if diff_count == 1:
                        toggle_idx = diff_mask.index(True)
                        toggle_name = toggle_cols[toggle_idx]

                        # Identify OFF and ON configs
                        if row_i[toggle_name] == 0 and row_j[toggle_name] == 1:
                            off_row, on_row = row_i, row_j
                        elif row_i[toggle_name] == 1 and row_j[toggle_name] == 0:
                            off_row, on_row = row_j, row_i
                        else:
                            continue  # skip non-binary or unclear cases

                        delta = on_row[metric] - off_row[metric]

                        label = (
                            f"{model_name}_ft{off_row['ft']}_ctx{off_row['context']}_"
                            f"oppu{off_row['oppu']}_style{base_style}"
                        )

                        # Search if label is already in rows
                        found = False
                        for r in rows:
                            if r["base_label"] == label:
                                r[toggle_name] = delta
                                found = True
                                break

                        if not found:
                            new_row = {toggle: None for toggle in toggle_cols}
                            new_row[toggle_name] = delta
                            new_row["base_label"] = label
                            rows.append(new_row)

            # Create and plot heatmap
            diff_df = pd.DataFrame(rows).set_index("base_label").astype(float)
            plt.figure(figsize=(8, max(4, len(diff_df) * 0.5)))
            sns.heatmap(diff_df, annot=True, cmap="RdBu_r", center=0, fmt=".3f", cbar_kws={"label": f"Δ {metric}"})
            plt.title(f"Effect of toggling ft/context/oppu ON (style=10, context=1)\nΔ {metric} = ON − OFF")
            plt.tight_layout()
            plt.savefig(os.path.join(folder_path, f"heatmap_toggles_{metric}_style10_context1.png"))
            plt.close()

         # Only consider configs with context=1
        base_ft = 1
        ft_df = context_df[context_df["ft"] == base_ft]

        for metric in metric_diffs.keys():
            rows = []

            # Group by model (assuming comparisons are only meaningful within same model)
            for model_name, model_group in ft_df.groupby("model"):
                # For all pairs of rows in that group
                for i, j in itertools.combinations(model_group.index, 2):
                    row_i = model_group.loc[i]
                    row_j = model_group.loc[j]

                    # Determine how many toggles differ
                    diff_mask = [row_i[toggle] != row_j[toggle] for toggle in toggle_cols]
                    diff_count = sum(diff_mask)

                    if diff_count == 1:
                        toggle_idx = diff_mask.index(True)
                        toggle_name = toggle_cols[toggle_idx]

                        # Identify OFF and ON configs
                        if row_i[toggle_name] == 0 and row_j[toggle_name] == 1:
                            off_row, on_row = row_i, row_j
                        elif row_i[toggle_name] == 1 and row_j[toggle_name] == 0:
                            off_row, on_row = row_j, row_i
                        else:
                            continue  # skip non-binary or unclear cases

                        delta = on_row[metric] - off_row[metric]

                        label = (
                            f"{model_name}_ft{off_row['ft']}_ctx{off_row['context']}_"
                            f"oppu{off_row['oppu']}_style{base_style}"
                        )

                        # Search if label is already in rows
                        found = False
                        for r in rows:
                            if r["base_label"] == label:
                                r[toggle_name] = delta
                                found = True
                                break

                        if not found:
                            new_row = {toggle: None for toggle in toggle_cols}
                            new_row[toggle_name] = delta
                            new_row["base_label"] = label
                            rows.append(new_row)

            # Create and plot heatmap
            diff_df = pd.DataFrame(rows).set_index("base_label").astype(float)
            plt.figure(figsize=(8, max(4, len(diff_df) * 0.5)))
            sns.heatmap(diff_df, annot=True, cmap="RdBu_r", center=0, fmt=".3f", cbar_kws={"label": f"Δ {metric}"})
            plt.title(f"Effect of toggling ft/context/oppu ON (style=10, context=1, ft=1)\nΔ {metric} = ON − OFF")
            plt.tight_layout()
            plt.savefig(os.path.join(folder_path, f"heatmap_toggles_{metric}_style10_context1_ft1.png"))
            plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model predictions and plot metrics.")
    parser.add_argument("folder_path", type=str, help="Path to folder containing validation CSV files")
    args = parser.parse_args()
    main(args.folder_path)
