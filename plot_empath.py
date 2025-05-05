import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#from feature_utils import parse_filename

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


def plot_significant_features(input_folder):
    data = []

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith("empath_significant_features.csv"):
                filepath = os.path.join(root, file)
                try:
                    model, ft, context, style, oppu = parse_filename(file)
                    df = pd.read_csv(filepath)
                    sig_count = (df["adjusted_p_value"] < 0.05).sum()
                    data.append({
                        "model": model,
                        "ft": ft,
                        "context": context,
                        "style": style,
                        "oppu": oppu,
                        "significant_features": sig_count
                    })
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")

    summary_df = pd.DataFrame(data)

    if summary_df.empty:
        print("No data to plot.")
        return

    # Sort data by number of significant features (increasing order)
    summary_df = summary_df.sort_values(by="significant_features", ascending=True)

    # Create a more readable label
    summary_df["label"] = summary_df.apply(
        lambda row: f"{row['model']} | ft {row['ft']} | ctx {row['context']} | style {row['style']} | oppu {row['oppu']}",
        axis=1
    )

    plt.figure(figsize=(14, 7))
    barplot = sns.barplot(
        data=summary_df,
        x="label",
        y="significant_features",
        hue="model",
        dodge=False
    )
     # Set tick label colors based on model
    model_to_color = {
        model: color for model, color in zip(summary_df["model"].unique(), sns.color_palette())
    }

    ax = plt.gca()
    for tick_label, (_, row) in zip(ax.get_xticklabels(), summary_df.iterrows()):
        tick_label.set_color(model_to_color[row["model"]])
        tick_label.set_fontweight("bold")
    
    plt.title("Significant Features per Configuration (adjusted p < 0.05)")
    plt.xlabel("Configuration")
    plt.ylabel("Significant Features Count")
    plt.xticks(rotation=90)
    plt.tight_layout()

    output_path = os.path.join(input_folder, "significant_features_summary.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to {output_path}")



def plot_heatmap_significant_features(input_folder, value_column="adjusted_p_value"):
    assert value_column in {"adjusted_p_value", "difference"}, \
        "value_column must be either 'adjusted_p_value' or 'difference'"

    data = []
    row_labels = []
    num_significant_features = []
    label_colors = []
    all_features = set()

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith("empath_significant_features.csv"):
                filepath = os.path.join(root, file)
                try:
                    model, ft, context, style, oppu = parse_filename(file)
                    label = f"{model} | ft {ft} | ctx {context} | style {style} | oppu {oppu}"
                    df = pd.read_csv(filepath)
                    sig_df = df[df["adjusted_p_value"] < 0.05][["feature", value_column]]

                    if value_column == "adjusted_p_value":
                        row_dict = {
                            row["feature"]: -np.log10(max(row["adjusted_p_value"], 1e-10))
                            for _, row in sig_df.iterrows()
                        }
                    else:  # difference
                        row_dict = {
                            row["feature"]: row["difference"]
                            for _, row in sig_df.iterrows()
                        }

                    data.append(row_dict)
                    row_labels.append(label)
                    all_features.update(sig_df["feature"].tolist())
                    num_significant_features.append(len(sig_df))

                    if "Mistral" in label:
                        label_colors.append('blue')
                    elif "DeepSeek" in label:
                        label_colors.append('green')
                    else:
                        label_colors.append('orange')

                except Exception as e:
                    print(f"Error processing {filepath}: {e}")

    if not data:
        print("No data to plot.")
        return

    print(f"Number of unique features across all files: {len(all_features)}")

    heatmap_df = pd.DataFrame(data)
    heatmap_df.index = row_labels
    heatmap_df['num_significant_features'] = num_significant_features
    heatmap_df = heatmap_df.sort_values(by='num_significant_features', ascending=True)
    sorted_label_colors = [label_colors[row_labels.index(label)] for label in heatmap_df.index]

    plt.figure(figsize=(min(25, 0.6 * heatmap_df.shape[1] + 5), max(6, 0.3 * heatmap_df.shape[0])))

    if value_column == "adjusted_p_value":
        colorbar_label = r'$-\log_{10}$(p-value)'
        cmap = "coolwarm"
    else:
        colorbar_label = "Difference"
        cmap = "vlag"  # diverging for differences

    ax = sns.heatmap(
        heatmap_df.drop(columns='num_significant_features'),
        cmap=cmap,
        linewidths=0.3,
        linecolor='gray',
        cbar_kws={'label': colorbar_label}
    )

    plt.title(f"Heatmap of Significant Feature {value_column.replace('_', ' ').title()}")
    plt.xlabel("Feature")
    plt.ylabel("Configuration")
    plt.tight_layout()

    for tick, color in zip(ax.get_yticklabels(), sorted_label_colors):
        tick.set_color(color)

    output_filename = f"significant_features_heatmap_{value_column}.png"
    output_path = os.path.join(input_folder, output_filename)
    plt.savefig(output_path)
    plt.close()
    print(f"Heatmap saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot number of significant empath features.")
    parser.add_argument("input_folder", type=str, help="Path to the folder containing subfolders with CSVs.")
    parser.add_argument("--value_column", type=str, default="adjusted_p_value",
                        choices=["adjusted_p_value", "difference"],
                        help="Which value to plot in the heatmap (default: adjusted_p_value)")
    args = parser.parse_args()

    plot_significant_features(args.input_folder)
    plot_heatmap_significant_features(args.input_folder, value_column=args.value_column)




