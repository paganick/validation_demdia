import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
from plotting_utils import *

def plot_accuracy_vs_significant_features(input_folder):
    results = []
    empath_data = []
    results_folder = os.path.join(input_folder, 'trade_offs')
    os.makedirs(results_folder, exist_ok=True)

    # Load class_0_accuracy from confusion_matrix.csv files
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith("confusion_matrix.csv"):
                filepath = os.path.join(root, file)
                try:
                    df = pd.read_csv(filepath)
                    cm = df.values
                    tn, fp = cm[0, 0], cm[0, 1]
                    fn, tp = cm[1, 0], cm[1, 1]

                    total_0 = tn + fp
                    correct_0 = tn

                    model, ft, context, style, oppu = parse_filename(file)
                    label = make_label(model, ft, context, style, oppu)

                    results.append({
                        "model": model,
                        "ft": ft,
                        "context": context,
                        "style": style,
                        "oppu": oppu,
                        "class_0_accuracy": correct_0 / total_0 if total_0 > 0 else None,
                        "label": label
                    })
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")

    acc_df = pd.DataFrame(results)

    # Load significant feature counts from empath_significant_features.csv
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith("empath_significant_features.csv"):
                filepath = os.path.join(root, file)
                try:
                    model, ft, context, style, oppu = parse_filename(file)
                    df = pd.read_csv(filepath)
                    sig_count = (df["adjusted_p_value"] < 0.05).sum()
                    empath_data.append({
                        "model": model,
                        "ft": ft,
                        "context": context,
                        "style": style,
                        "oppu": oppu,
                        "significant_features": sig_count
                    })
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")

    empath_df = pd.DataFrame(empath_data)

    # Merge both dataframes on all config columns
    merged_df = pd.merge(acc_df, empath_df, on=["model", "ft", "context", "style", "oppu"])

    if merged_df.empty:
        print("No matching data to plot.")
        return

    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=merged_df, x="class_0_accuracy", y="significant_features", hue="model", palette="tab10")

    # Add labels with adjustText
    texts = []
    for _, row in merged_df.iterrows():
        texts.append(
            plt.text(row["class_0_accuracy"], row["significant_features"], row["label"], fontsize=8, alpha=0.75)
        )
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    plt.xlabel("Class-0 Accuracy")
    plt.ylabel("Number of Significant Features (p < 0.05)")
    plt.title("Class-0 Accuracy vs Significant Features")
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, "class0_accuracy_vs_significant_features.png"))
    plt.close()


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot significant empath features and accuracy correlations.")
    parser.add_argument("input_folder", type=str, help="Path to the folder containing subfolders with CSVs.")
    args = parser.parse_args()

    plot_accuracy_vs_significant_features(args.input_folder)
