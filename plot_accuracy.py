import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plotting_utils import parse_filename, make_label
from adjustText import adjust_text

plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.fontset'] = 'dejavusans'

def main(folder_path):
    results = []
    results_folder = os.path.join(folder_path, 'accuracy')
    os.makedirs(results_folder, exist_ok=True)
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
                        "style": style,
                        "context": context,
                        "ft": ft,
                        "oppu": oppu,
                        "accuracy": correct / total,
                        "class_0_accuracy": correct_0 / total_0 if total_0 > 0 else None,
                        "label": make_label(model, ft, context, style, oppu)
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

    # Class-0 accuracy plot (sorted)
    results_df_sorted_c0 = results_df.sort_values(by="class_0_accuracy", ascending=True, na_position="last")
    results_df_sorted_c0["label"] = pd.Categorical(
        results_df_sorted_c0["label"], categories=results_df_sorted_c0["label"], ordered=True
    )

    # Sort by: model (deepseek < llama < mistral), then style (0 < 10), context (0 < 1), ft (0 < 1), oppu (0 < 1)
    model_order = {"deepseek": 0, "llama": 1, "mistral": 2}
    results_df["model_order"] = results_df["model"].map(model_order)

    # sort_cols = ["model_order", "style", "context", "ft", "oppu"]
    sort_cols = ["model_order", "oppu", "ft", "context", "style"]
    results_df = results_df.sort_values(by=sort_cols, ascending=True)

    # Ensure the labels follow the new sorted order
    results_df["label"] = pd.Categorical(
        results_df["label"], categories=results_df["label"], ordered=True
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
    plt.savefig(os.path.join(results_folder, "overall_accuracy_barplot.png"))
    plt.close()

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
    plt.savefig(os.path.join(results_folder, "class0_accuracy_barplot.png"))
    plt.close()

     # Scatter plot: accuracy vs. class-0 accuracy
    scatter_df = results_df.dropna(subset=["class_0_accuracy"])

    # Determine plot bounds
    min_val = min(scatter_df["accuracy"].min(), scatter_df["class_0_accuracy"].min())
    max_val = max(scatter_df["accuracy"].max(), scatter_df["class_0_accuracy"].max())

    plt.figure(figsize=(12, 10))
    for model in scatter_df["model"].unique():
        model_data = scatter_df[scatter_df["model"] == model]
        plt.scatter(
            model_data["accuracy"], 
            model_data["class_0_accuracy"], 
            label=model, 
            color=model_palette[model],
            s=60,
            alpha=0.8,
            edgecolor='k'
        )
        # # ... inside your plotting code
        # texts = []
        # for _, row in scatter_df.iterrows():
        #     texts.append(
        #         plt.text(
        #             row["accuracy"],
        #             row["class_0_accuracy"],
        #             row["label"],
        #             fontsize=8,
        #             alpha=0.75
        #         )
        #     )

        # # Automatically adjust labels to minimize overlap
        # adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))


    # Add y = x line within bounds
    plt.plot([min_val, max_val], [min_val, max_val], ls="--", color="gray")

    plt.xlabel("Overall Accuracy")
    plt.ylabel("Class-0 Accuracy")
    plt.title("Accuracy vs. Class-0 Accuracy per Configuration")
    plt.xlim(min_val - 0.01, max_val + 0.01)
    plt.ylim(min_val - 0.01, max_val + 0.01)
    plt.legend(title="Model", loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, "scatter_accuracy_vs_class0.png"))
    plt.close()




    # -------- Heatmap generation --------
    toggles = ['style', 'context', 'ft', 'oppu']
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
        plt.savefig(os.path.join(results_folder, f"heatmap_toggle_effect_{metric}.png"))
        plt.close()

        # NEXT

        import itertools

        toggle_cols = ["style", "context", "ft", "oppu"]
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

                        label = (make_label(model_name, off_row['ft'], off_row['context'], off_row['style'], off_row['oppu']))

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
            plt.figure(figsize=(10, max(6, len(diff_df) * 0.6)))
            sns.heatmap(diff_df, annot=True, cmap="RdBu_r", center=0, fmt=".3f", cbar_kws={"label": f"Δ {metric}"})
            plt.title(f"Effect of toggling style/ft/context/oppu ON \nΔ {metric} = ON − OFF")
            plt.tight_layout()
            plt.savefig(os.path.join(results_folder, f"heatmap_toggles_{metric}.png"))
            plt.close()  


        # base_style = 10

        # # Only consider configs with style=10
        # style10_df = results_df[results_df["style"] == base_style]

        # for metric in metric_diffs.keys():
        #     rows = []

        #     # Group by model (assuming comparisons are only meaningful within same model)
        #     for model_name, model_group in style10_df.groupby("model"):
        #         # For all pairs of rows in that group
        #         for i, j in itertools.combinations(model_group.index, 2):
        #             row_i = model_group.loc[i]
        #             row_j = model_group.loc[j]

        #             # Determine how many toggles differ
        #             diff_mask = [row_i[toggle] != row_j[toggle] for toggle in toggle_cols]
        #             diff_count = sum(diff_mask)

        #             if diff_count == 1:
        #                 toggle_idx = diff_mask.index(True)
        #                 toggle_name = toggle_cols[toggle_idx]

        #                 # Identify OFF and ON configs
        #                 if row_i[toggle_name] == 0 and row_j[toggle_name] == 1:
        #                     off_row, on_row = row_i, row_j
        #                 elif row_i[toggle_name] == 1 and row_j[toggle_name] == 0:
        #                     off_row, on_row = row_j, row_i
        #                 else:
        #                     continue  # skip non-binary or unclear cases

        #                 delta = on_row[metric] - off_row[metric]

        #                 label = (make_label(model_name, off_row['ft'], off_row['context'], off_row['style'], off_row['oppu']))

        #                 # Search if label is already in rows
        #                 found = False
        #                 for r in rows:
        #                     if r["base_label"] == label:
        #                         r[toggle_name] = delta
        #                         found = True
        #                         break

        #                 if not found:
        #                     new_row = {toggle: None for toggle in toggle_cols}
        #                     new_row[toggle_name] = delta
        #                     new_row["base_label"] = label
        #                     rows.append(new_row)

        #     # Create and plot heatmap
        #     diff_df = pd.DataFrame(rows).set_index("base_label").astype(float)
        #     plt.figure(figsize=(10, max(6, len(diff_df) * 0.6)))
        #     sns.heatmap(diff_df, annot=True, cmap="RdBu_r", center=0, fmt=".3f", cbar_kws={"label": f"Δ {metric}"})
        #     plt.title(f"Effect of toggling ft/context/oppu ON (style=10)\nΔ {metric} = ON − OFF")
        #     plt.tight_layout()
        #     plt.savefig(os.path.join(results_folder, f"heatmap_toggles_{metric}_style10.png"))
        #     plt.close()


        # # Only consider configs with context=1
        # base_context = 1
        # context_df = style10_df[style10_df["context"] == base_context]

        # for metric in metric_diffs.keys():
        #     rows = []

        #     # Group by model (assuming comparisons are only meaningful within same model)
        #     for model_name, model_group in context_df.groupby("model"):
        #         # For all pairs of rows in that group
        #         for i, j in itertools.combinations(model_group.index, 2):
        #             row_i = model_group.loc[i]
        #             row_j = model_group.loc[j]

        #             # Determine how many toggles differ
        #             diff_mask = [row_i[toggle] != row_j[toggle] for toggle in toggle_cols]
        #             diff_count = sum(diff_mask)

        #             if diff_count == 1:
        #                 toggle_idx = diff_mask.index(True)
        #                 toggle_name = toggle_cols[toggle_idx]

        #                 # Identify OFF and ON configs
        #                 if row_i[toggle_name] == 0 and row_j[toggle_name] == 1:
        #                     off_row, on_row = row_i, row_j
        #                 elif row_i[toggle_name] == 1 and row_j[toggle_name] == 0:
        #                     off_row, on_row = row_j, row_i
        #                 else:
        #                     continue  # skip non-binary or unclear cases

        #                 delta = on_row[metric] - off_row[metric]

        #                 label = (make_label(model_name, off_row['ft'], off_row['context'], off_row['style'], off_row['oppu']))

        #                 # Search if label is already in rows
        #                 found = False
        #                 for r in rows:
        #                     if r["base_label"] == label:
        #                         r[toggle_name] = delta
        #                         found = True
        #                         break

        #                 if not found:
        #                     new_row = {toggle: None for toggle in toggle_cols}
        #                     new_row[toggle_name] = delta
        #                     new_row["base_label"] = label
        #                     rows.append(new_row)

        #     # Create and plot heatmap
        #     diff_df = pd.DataFrame(rows).set_index("base_label").astype(float)
        #     plt.figure(figsize=(10, max(6, len(diff_df) * 0.6)))
        #     sns.heatmap(diff_df, annot=True, cmap="RdBu_r", center=0, fmt=".3f", cbar_kws={"label": f"Δ {metric}"})
        #     plt.title(f"Effect of toggling ft/context/oppu ON (style=10, context=1)\nΔ {metric} = ON − OFF")
        #     plt.tight_layout()
        #     plt.savefig(os.path.join(results_folder, f"heatmap_toggles_{metric}_style10_context1.png"))
        #     plt.close()

        #  # Only consider configs with context=1
        # base_ft = 1
        # ft_df = context_df[context_df["ft"] == base_ft]

        # for metric in metric_diffs.keys():
        #     rows = []

        #     # Group by model (assuming comparisons are only meaningful within same model)
        #     for model_name, model_group in ft_df.groupby("model"):
        #         # For all pairs of rows in that group
        #         for i, j in itertools.combinations(model_group.index, 2):
        #             row_i = model_group.loc[i]
        #             row_j = model_group.loc[j]

        #             # Determine how many toggles differ
        #             diff_mask = [row_i[toggle] != row_j[toggle] for toggle in toggle_cols]
        #             diff_count = sum(diff_mask)

        #             if diff_count == 1:
        #                 toggle_idx = diff_mask.index(True)
        #                 toggle_name = toggle_cols[toggle_idx]

        #                 # Identify OFF and ON configs
        #                 if row_i[toggle_name] == 0 and row_j[toggle_name] == 1:
        #                     off_row, on_row = row_i, row_j
        #                 elif row_i[toggle_name] == 1 and row_j[toggle_name] == 0:
        #                     off_row, on_row = row_j, row_i
        #                 else:
        #                     continue  # skip non-binary or unclear cases

        #                 delta = on_row[metric] - off_row[metric]

        #                 label = (make_label(model_name, off_row['ft'], off_row['context'], off_row['style'], off_row['oppu']))

        #                 # Search if label is already in rows
        #                 found = False
        #                 for r in rows:
        #                     if r["base_label"] == label:
        #                         r[toggle_name] = delta
        #                         found = True
        #                         break

        #                 if not found:
        #                     new_row = {toggle: None for toggle in toggle_cols}
        #                     new_row[toggle_name] = delta
        #                     new_row["base_label"] = label
        #                     rows.append(new_row)

        #     # Create and plot heatmap
        #     diff_df = pd.DataFrame(rows).set_index("base_label").astype(float)
        #     plt.figure(figsize=(10, max(6, len(diff_df) * 0.6)))
        #     sns.heatmap(diff_df, annot=True, cmap="RdBu_r", center=0, fmt=".3f", cbar_kws={"label": f"Δ {metric}"})
        #     plt.title(f"Effect of toggling ft/context/oppu ON (style=10, context=1, ft=1)\nΔ {metric} = ON − OFF")
        #     plt.tight_layout()
        #     plt.savefig(os.path.join(results_folder, f"heatmap_toggles_{metric}_style10_context1_ft1.png"))
        #     plt.close()



    ########################

    # metrics = ['accuracy', 'class_0_accuracy']

    # for metric in metrics:
    #     diff_rows = []

    #     for model in results_df["model"].unique():
    #         # Base configurations with all toggles off
    #         base_config = results_df[
    #             (results_df["model"] == model) &
    #             (results_df["ft"] == 1) &
    #             (results_df["context"] == 1) &
    #             (results_df["style"] == 10) &
    #             (results_df["oppu"] == 1)
    #         ]
    #         if base_config.empty:
    #             continue
    #         base_metric = base_config.iloc[0][metric]
    #         row_id = f"{model}_1111"
    #         row = {"base": row_id}

    #         for toggle in toggles:
    #             # Create a modified config where one toggle is set to 1
    #             query = {
    #                 "model": model,
    #                 "ft": 1,
    #                 "context": 1,
    #                 "style": 10,
    #                 "oppu": 1
    #             }
                
    #             query[toggle] = 0

    #             modified = results_df[
    #                 (results_df["model"] == query["model"]) &
    #                 (results_df["ft"] == query["ft"]) &
    #                 (results_df["context"] == query["context"]) &
    #                 (results_df["style"] == query["style"]) &
    #                 (results_df["oppu"] == query["oppu"])
    #             ]
    #             if not modified.empty:
    #                 mod_metric = modified.iloc[0][metric]
    #                 if pd.notnull(base_metric) and pd.notnull(mod_metric):
    #                     row[toggle] = mod_metric - base_metric
    #                 else:
    #                     row[toggle] = None
    #             else:
    #                 row[toggle] = None

    #         diff_rows.append(row)

    #     diff_df = pd.DataFrame(diff_rows).set_index("base")
    #     diff_df = diff_df.astype(float)  # Ensure numerical dtype for heatmap
    #     sns.heatmap(diff_df, annot=True, cmap="RdBu_r", center=0, fmt=".3f")
    #     plt.figure(figsize=(10, max(6, len(diff_df) * 0.6)))
    #     sns.heatmap(diff_df, annot=True, cmap="RdBu_r", center=0, fmt=".3f")
    #     plt.title(f"Effect of Enabling Each Toggle on {metric}")
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(results_folder, f"heatmap_toggle_effect_{metric}_opposite.png"))
    #     plt.close()


    #     import itertools

    #     toggle_cols = ["style", "context", "ft", "oppu"]
    #     metric_diffs = {
    #         "accuracy": [],
    #         "class_0_accuracy": []
    #     }

    #     for metric in metric_diffs.keys():
    #         rows = []

    #         # Group by model (assuming comparisons are only meaningful within same model)
    #         for model_name, model_group in results_df.groupby("model"):
    #             # For all pairs of rows in that group
    #             for i, j in itertools.combinations(model_group.index, 2):
    #                 row_i = model_group.loc[i]
    #                 row_j = model_group.loc[j]

    #                 # Determine how many toggles differ
    #                 diff_mask = [row_i[toggle] != row_j[toggle] for toggle in toggle_cols]
    #                 diff_count = sum(diff_mask)

    #                 if diff_count == 1:
    #                     toggle_idx = diff_mask.index(True)
    #                     toggle_name = toggle_cols[toggle_idx]

    #                     # Identify OFF and ON configs
    #                     if row_i[toggle_name] == 0 and row_j[toggle_name] != 0 :
    #                         off_row, on_row = row_i, row_j
    #                     elif row_i[toggle_name] != 0 and row_j[toggle_name] == 0:
    #                         off_row, on_row = row_j, row_i
    #                     else:
    #                         continue  # skip non-binary or unclear cases

    #                     delta = off_row[metric] - on_row[metric]

    #                     label = (make_label(model_name, on_row['ft'], on_row['context'], on_row['style'], on_row['oppu']))

    #                     # Search if label is already in rows
    #                     found = False
    #                     for r in rows:
    #                         if r["base_label"] == label:
    #                             r[toggle_name] = delta
    #                             found = True
    #                             break

    #                     if not found:
    #                         new_row = {toggle: None for toggle in toggle_cols}
    #                         new_row[toggle_name] = delta
    #                         new_row["base_label"] = label
    #                         rows.append(new_row)

    #         # Create and plot heatmap
    #         diff_df = pd.DataFrame(rows).set_index("base_label").astype(float)
    #         plt.figure(figsize=(10, max(6, len(diff_df) * 0.6)))
    #         sns.heatmap(diff_df, annot=True, cmap="RdBu_r", center=0, fmt=".3f", cbar_kws={"label": f"Δ {metric}"})
    #         plt.title(f"Effect of toggling style/ft/context/oppu OFF \nΔ {metric} = OFF − ON")
    #         plt.tight_layout()
    #         plt.savefig(os.path.join(results_folder, f"heatmap_toggles_{metric}_opposite.png"))
    #         plt.close()  


    #     base_style = 0

    #     # Only consider configs with style=10
    #     style10_df = results_df[results_df["style"] == base_style]

    #     for metric in metric_diffs.keys():
    #         rows = []

    #         # Group by model (assuming comparisons are only meaningful within same model)
    #         for model_name, model_group in style10_df.groupby("model"):
    #             # For all pairs of rows in that group
    #             for i, j in itertools.combinations(model_group.index, 2):
    #                 row_i = model_group.loc[i]
    #                 row_j = model_group.loc[j]

    #                 # Determine how many toggles differ
    #                 diff_mask = [row_i[toggle] != row_j[toggle] for toggle in toggle_cols]
    #                 diff_count = sum(diff_mask)

    #                 if diff_count == 1:
    #                     toggle_idx = diff_mask.index(True)
    #                     toggle_name = toggle_cols[toggle_idx]

    #                     # Identify OFF and ON configs
    #                     if row_i[toggle_name] == 0 and row_j[toggle_name] == 1:
    #                         off_row, on_row = row_i, row_j
    #                     elif row_i[toggle_name] == 1 and row_j[toggle_name] == 0:
    #                         off_row, on_row = row_j, row_i
    #                     else:
    #                         continue  # skip non-binary or unclear cases

    #                     delta = off_row[metric] - on_row[metric]

    #                     label = (make_label(model_name, on_row['ft'], on_row['context'], on_row['style'], on_row['oppu']))

    #                     # Search if label is already in rows
    #                     found = False
    #                     for r in rows:
    #                         if r["base_label"] == label:
    #                             r[toggle_name] = delta
    #                             found = True
    #                             break

    #                     if not found:
    #                         new_row = {toggle: None for toggle in toggle_cols}
    #                         new_row[toggle_name] = delta
    #                         new_row["base_label"] = label
    #                         rows.append(new_row)

    #         # Create and plot heatmap
    #         diff_df = pd.DataFrame(rows).set_index("base_label").astype(float)
    #         plt.figure(figsize=(10, max(6, len(diff_df) * 0.6)))
    #         sns.heatmap(diff_df, annot=True, cmap="RdBu_r", center=0, fmt=".3f", cbar_kws={"label": f"Δ {metric}"})
    #         plt.title(f"Effect of toggling ft/context/oppu OFF (style=10)\nΔ {metric} = OFF − ON")
    #         plt.tight_layout()
    #         plt.savefig(os.path.join(results_folder, f"heatmap_toggles_{metric}_style10_opposite.png"))
    #         plt.close()


    #     # Only consider configs with context=1
    #     base_context = 0
    #     context_df = style10_df[style10_df["context"] == base_context]

    #     for metric in metric_diffs.keys():
    #         rows = []

    #         # Group by model (assuming comparisons are only meaningful within same model)
    #         for model_name, model_group in context_df.groupby("model"):
    #             # For all pairs of rows in that group
    #             for i, j in itertools.combinations(model_group.index, 2):
    #                 row_i = model_group.loc[i]
    #                 row_j = model_group.loc[j]

    #                 # Determine how many toggles differ
    #                 diff_mask = [row_i[toggle] != row_j[toggle] for toggle in toggle_cols]
    #                 diff_count = sum(diff_mask)

    #                 if diff_count == 1:
    #                     toggle_idx = diff_mask.index(True)
    #                     toggle_name = toggle_cols[toggle_idx]

    #                     # Identify OFF and ON configs
    #                     if row_i[toggle_name] == 0 and row_j[toggle_name] == 1:
    #                         off_row, on_row = row_i, row_j
    #                     elif row_i[toggle_name] == 1 and row_j[toggle_name] == 0:
    #                         off_row, on_row = row_j, row_i
    #                     else:
    #                         continue  # skip non-binary or unclear cases

    #                     delta = off_row[metric] - on_row[metric]

    #                     label = (make_label(model_name, on_row['ft'], on_row['context'], on_row['style'], on_row['oppu']))

    #                     # Search if label is already in rows
    #                     found = False
    #                     for r in rows:
    #                         if r["base_label"] == label:
    #                             r[toggle_name] = delta
    #                             found = True
    #                             break

    #                     if not found:
    #                         new_row = {toggle: None for toggle in toggle_cols}
    #                         new_row[toggle_name] = delta
    #                         new_row["base_label"] = label
    #                         rows.append(new_row)

    #         # Create and plot heatmap
    #         diff_df = pd.DataFrame(rows).set_index("base_label").astype(float)
    #         plt.figure(figsize=(10, max(6, len(diff_df) * 0.6)))
    #         sns.heatmap(diff_df, annot=True, cmap="RdBu_r", center=0, fmt=".3f", cbar_kws={"label": f"Δ {metric}"})
    #         plt.title(f"Effect of toggling ft/context/oppu OFF (style=10, context=1)\nΔ {metric} = OFF − ON")
    #         plt.tight_layout()
    #         plt.savefig(os.path.join(results_folder, f"heatmap_toggles_{metric}_style10_context1_opposite.png"))
    #         plt.close()

    #      # Only consider configs with context=1
    #     base_ft = 0
    #     ft_df = context_df[context_df["ft"] == base_ft]

    #     for metric in metric_diffs.keys():
    #         rows = []

    #         # Group by model (assuming comparisons are only meaningful within same model)
    #         for model_name, model_group in ft_df.groupby("model"):
    #             # For all pairs of rows in that group
    #             for i, j in itertools.combinations(model_group.index, 2):
    #                 row_i = model_group.loc[i]
    #                 row_j = model_group.loc[j]

    #                 # Determine how many toggles differ
    #                 diff_mask = [row_i[toggle] != row_j[toggle] for toggle in toggle_cols]
    #                 diff_count = sum(diff_mask)

    #                 if diff_count == 1:
    #                     toggle_idx = diff_mask.index(True)
    #                     toggle_name = toggle_cols[toggle_idx]

    #                     # Identify OFF and ON configs
    #                     if row_i[toggle_name] == 0 and row_j[toggle_name] == 1:
    #                         off_row, on_row = row_i, row_j
    #                     elif row_i[toggle_name] == 1 and row_j[toggle_name] == 0:
    #                         off_row, on_row = row_j, row_i
    #                     else:
    #                         continue  # skip non-binary or unclear cases

    #                     delta = off_row[metric] - on_row[metric]

    #                     label = (make_label(model_name, on_row['ft'], on_row['context'], on_row['style'], on_row['oppu']))

    #                     # Search if label is already in rows
    #                     found = False
    #                     for r in rows:
    #                         if r["base_label"] == label:
    #                             r[toggle_name] = delta
    #                             found = True
    #                             break

    #                     if not found:
    #                         new_row = {toggle: None for toggle in toggle_cols}
    #                         new_row[toggle_name] = delta
    #                         new_row["base_label"] = label
    #                         rows.append(new_row)

    #         # Create and plot heatmap
    #         diff_df = pd.DataFrame(rows).set_index("base_label").astype(float)
    #         plt.figure(figsize=(10, max(6, len(diff_df) * 0.6)))
    #         sns.heatmap(diff_df, annot=True, cmap="RdBu_r", center=0, fmt=".3f", cbar_kws={"label": f"Δ {metric}"})
    #         plt.title(f"Effect of toggling ft/context/oppu OFF (style=10, context=1, ft=1)\nΔ {metric} = OFF − ON")
    #         plt.tight_layout()
    #         plt.savefig(os.path.join(results_folder, f"heatmap_toggles_{metric}_style10_context1_ft1_opposite.png"))
    #         plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model predictions and plot metrics.")
    parser.add_argument("folder_path", type=str, help="Path to folder containing validation CSV files")
    args = parser.parse_args()
    main(args.folder_path)
