#!/usr/bin/env python3
"""
Common plotting utilities for research figure generation.
Contains shared functions for parsing, styling, and data processing.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import copy
from contextlib import contextmanager
from pathlib import Path
from typing import List, Dict, Union, Tuple, Optional

# === Color Palettes ===
MODEL_PALETTE = {
    "DeepSeek-R1-Distill-Llama-8B": "#ff6f61",     # Bright coral red
    "Llama-3.1-8B": "#bb86fc",                     # Bright purple
    "Llama-3.1-8B-Instruct": "#9a4dff",            # Vivid violet
    "Llama-3.1-70B": "#5a189a",                    # Dark violet

    "Mistral-7B-v0.1": "#4fc3f7",                  # Light cyan-blue
    "Mistral-7B-Instruct-v0.2": "#0288d1",         # Bright medium blue

    "gemma-3-4b-it": "#81c784",                    # Light green
    "Qwen2.5-7B-Instruct": "#ffb74d",              # Warm orange (amber)

    "Apertus-8B-2509": "#a1887f",                  # Muted brownish-gray
}

DATASET_PALETTE = {
    'results_bluesky': '#1E88E5',    # Blue
    'results_twitter': '#000000',   # Black for Twitter/X
    'results_reddit': '#FF4500',     # Reddit orange
    'Bluesky': '#1E88E5',           # Blue (alternative names)
    'Twitter/X': '#000000',         # Black for Twitter/X
    'Twitter': '#000000',                 # Black for Twitter/X
    'X': '#000000',                 # Black for Twitter/X
    'Reddit': '#FF4500'              # Reddit orange
}

# === Dataset Name Normalization ===
def normalize_dataset_name(dataset_path):
    """
    Normalize dataset name by extracting the base name.
    Examples:
      'results/results_bluesky' -> 'results_bluesky'
      'results_bluesky' -> 'results_bluesky'
      '/path/to/results_bluesky' -> 'results_bluesky'
    """
    basename = os.path.basename(dataset_path)
    return basename

def get_dataset_color(dataset_path):
    """Get color for a dataset, handling subfolder paths."""
    normalized = normalize_dataset_name(dataset_path)
    return DATASET_PALETTE.get(normalized, '#808080')

# === Filename Parsing ===
def parse_filename(filename):
    """
    Extracts model, finetuning, context, style, OPPU, and persona info from a filename.
    Converts them into booleans or integers for clean tabular use.
    Handles cases where persona field might be empty (defaults to 1).
    """
    base = os.path.basename(filename)
    # Remove any file extensions and suffixes
    if '___' in base:
        base = base.split('___')[0]
    elif '__random' in base:
        base = base.split('__random')[0]
    
    parts = base.split("__")
    
    # Handle different numbers of parts
    if len(parts) < 5:
        print(f"Warning: filename has only {len(parts)} parts: {parts}")
        return None, None, None, None, None, None
    
    model = parts[0]
    ft = 1 if parts[1] == "ft" else 0
    context = 1 if parts[2] == "ctx1" else 0
    style = int(parts[3].replace("style", ""))
    oppu = 1 if parts[4].startswith("OPPU") and not parts[4].startswith("no_OPPU") else 0
    
    # Handle persona - default to 1 if field is missing
    if len(parts) > 5:
        persona = 0 if parts[5].startswith("no_persona") else 1
    else:
        persona = 1  # Default to persona on if field is missing
    
    return model, ft, context, style, oppu, persona

# === Label Generation ===
def make_label(model, ft, context, style, oppu, persona, with_model=True):
    """Create a label string from configuration parameters."""
    if with_model:
        label = model
    else:
        label = 'BL'
    if persona:
        label = label + " + PE"
    if style:
        label = label + " + SE"
    if context:
        label = label + " + CR"
    if ft:
        label = label + " + FT"
    return label

# === Marker Selection ===
def get_marker(row):
    """Get marker style based on row configuration."""
    if row['style'] == 0 and row['context'] == 0 and row['ft'] == 0 and row['oppu'] == 0 and row['persona'] == 0:
        return 'p'  # upside-down triangle
    elif row['style'] == 0 and row['context'] == 0 and row['ft'] == 0 and row['oppu'] == 0:
        return '^'  # triangle
    elif row['context'] == 0 and row['ft'] == 0 and row['oppu'] == 0:
        return 's'  # square
    elif row['ft'] == 0 and row['oppu'] == 0:
        return 'd'  # pentagon
    elif row['oppu'] == 0:
        return 'o'  # circle
    else:
        return 'x'  # fallback marker for other cases if any

# === Model Ordering ===
def get_ordered_models(model_names):
    """
    Create a custom ordering of models: non-instruct models first (alphabetical), 
    then instruct models (alphabetical).
    """
    # Separate instruct and non-instruct models
    non_instruct = []
    instruct = []
    
    for model in model_names:
        if "Instruct" in model or "-it" in model:
            instruct.append(model)
        else:
            non_instruct.append(model)
    
    # Sort each group alphabetically
    non_instruct.sort(key=str.lower)
    instruct.sort(key=str.lower)
    
    # Combine: non-instruct first, then instruct
    return non_instruct + instruct

# === Dataset Name Formatting ===
def format_dataset_name(dataset_name):
    """Convert dataset name to consistent display format."""
    # Normalize first to handle subfolders
    normalized = normalize_dataset_name(dataset_name)
    
    if 'twitter' in normalized.lower():
        return 'Twitter/X'
    elif 'bluesky' in normalized.lower():
        return 'Bluesky'
    elif 'reddit' in normalized.lower():
        return 'Reddit'
    return normalized.replace('results_', '').title()

# === Plotting Style Functions ===
def apply_presentation_style(presentation_mode: bool = False):
    """Applies consistent rcParams for plotting."""
    base_fontsize = 12
    color = "white" if presentation_mode else "black"
    
    rcParams.update({
        "axes.facecolor": "none" if presentation_mode else "white",
        "figure.facecolor": "none" if presentation_mode else "white",
        "text.color": color,
        "xtick.color": color,
        "ytick.color": color,
        "axes.labelcolor": color,
        "axes.edgecolor": color,
        "legend.edgecolor": color,
        "font.size": base_fontsize,
        "axes.titlesize": base_fontsize + 4,
        "axes.labelsize": base_fontsize,
        "xtick.labelsize": base_fontsize,
        "ytick.labelsize": base_fontsize,
        "legend.fontsize": base_fontsize,
        "legend.title_fontsize": base_fontsize + 2,
    })

@contextmanager
def with_plot_style(presentation_mode: bool = False):
    """Context manager to apply and restore plot style."""
    original = copy.deepcopy(rcParams)
    apply_presentation_style(presentation_mode)
    yield
    rcParams.update(original)

# === Confusion Matrix Parsing ===
def parse_confusion_matrix(cm_data: Union[str, List, np.ndarray]) -> np.ndarray:
    """Parse confusion matrix from various formats."""
    if isinstance(cm_data, np.ndarray):
        return cm_data
    elif isinstance(cm_data, list):
        return np.array(cm_data)
    elif isinstance(cm_data, str):
        lines = cm_data.strip().split('\n')
        matrix = []
        for line in lines:
            line = line.strip().strip('[]')
            numbers = [int(x) for x in line.split()]
            matrix.append(numbers)
        return np.array(matrix)
    else:
        raise ValueError(f"Unsupported confusion matrix format: {type(cm_data)}")

def calculate_metrics(cm: np.ndarray) -> Dict[str, float]:
    """Calculate accuracy from confusion matrix."""
    tn, fp = cm[0, 0], cm[0, 1]
    fn, tp = cm[1, 0], cm[1, 1]
    
    total = tn + fp + fn + tp
    correct = tn + tp
    
    accuracy = correct / total if total > 0 else None
    
    return {"accuracy": accuracy, "total_samples": total}

def calculate_accuracy(cm: np.ndarray) -> float:
    """Calculate accuracy from confusion matrix (simple version)."""
    tn, fp = cm[0, 0], cm[0, 1]
    fn, tp = cm[1, 0], cm[1, 1]
    
    total = tn + fp + fn + tp
    correct = tn + tp
    
    return correct / total if total > 0 else None

# === Confusion Matrix Field Detection ===
def is_confusion_matrix_data(value) -> bool:
    """Check if a value looks like confusion matrix data."""
    if isinstance(value, str):
        return "[[" in value and "]]" in value
    elif isinstance(value, list):
        if len(value) >= 2:
            try:
                arr = np.array(value)
                return arr.ndim == 2 and np.issubdtype(arr.dtype, np.number)
            except (ValueError, TypeError):
                return False
    return False

def find_confusion_matrix_fields(data) -> List[str]:
    """Find all fields in the JSON that contain confusion matrices."""
    cm_fields = []
    
    if isinstance(data, dict):
        for key, value in data.items():
            if "confusion_matrix" in key.lower() and is_confusion_matrix_data(value):
                cm_fields.append(key)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, dict):
                for key, value in item.items():
                    if key == "confusion_matrix" and is_confusion_matrix_data(value):
                        cm_fields.append(f"item_{i}_{key}")
    
    return cm_fields

def get_confusion_matrix_value(data, field_name: str):
    """Extract confusion matrix data from data structure."""
    if isinstance(data, dict):
        return data[field_name]
    elif isinstance(data, list):
        parts = field_name.split("_", 2)
        if len(parts) >= 3 and parts[0] == "item":
            item_index = int(parts[1])
            field = parts[2]
            return data[item_index][field]
    raise KeyError(f"Could not find field {field_name}")

# === File Filtering ===
def filter_for_baseline_persona(filepath: str) -> bool:
    """
    Check if a file matches baseline+persona criteria:
    - Contains 'random' (not ml_optimal or cosine_optimal)
    - Has no finetuning (noft)
    - Has no style (style0)
    - Has no context (ctx0)
    - Has persona (persona, not no_persona)
    """
    filename = os.path.basename(filepath)
    
    if 'random' not in filename:
        return False
    
    if 'ml_optimal' in filename or 'cosine_optimal' in filename:
        return False
    
    if '__noft__ctx0__style0__' in filename:
        if '__no_OPPU___' in filename or '__OPPU0__persona__' in filename:
            return True
    
    return False

# === Data Loading Functions (from old plotting_utils) ===
def load_uncertainty_data(folder_path):
    """Load accuracy data (with mean/std) from all trainer_results.json files in a folder."""
    results = []
    root_path = Path(folder_path)
    json_files = list(root_path.rglob("*trainer_results.json"))
    print(f"Found {len(json_files)} trainer_results.json files in {folder_path}")

    for filepath in json_files:
        try:
            model, ft, context, style, oppu, persona = parse_filename(str(filepath))

            # Only load random validation files if present
            if "random_validation" not in str(filepath):
                continue

            with open(filepath, "r") as f:
                data = json.load(f)

            cm_fields = find_confusion_matrix_fields(data)
            if not cm_fields:
                continue

            accuracies = []
            for field in cm_fields:
                cm_data = get_confusion_matrix_value(data, field)
                cm = parse_confusion_matrix(cm_data)
                acc = calculate_accuracy(cm)
                if acc is not None:
                    accuracies.append(acc)

            if len(accuracies) >= 1:
                results.append({
                    "model": model,
                    "ft": ft,
                    "context": context,
                    "style": style,
                    "oppu": oppu,
                    "persona": persona,
                    "accuracy_mean": np.mean(accuracies),
                    "accuracy_std": np.std(accuracies, ddof=1) if len(accuracies) > 1 else 0.0,
                    "response_type": "random",
                    "n_measurements": len(accuracies)
                })

        except Exception as e:
            print(f"  Failed to process {filepath}: {e}")
            continue

    return pd.DataFrame(results)


def load_uncertainty_data_multiple_folders(folder_paths):
    """Load and combine accuracy data from multiple dataset folders."""
    all_results = []

    for folder_path in folder_paths:
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} does not exist, skipping...")
            continue

        print(f"Processing folder: {folder_path}")
        df = load_uncertainty_data(folder_path)
        if not df.empty:
            df["dataset"] = normalize_dataset_name(folder_path)  # Use normalized name
            all_results.append(df)

    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        print(f"Combined data shape: {combined_df.shape}")
        return combined_df
    else:
        print("No valid data loaded.")
        return pd.DataFrame()