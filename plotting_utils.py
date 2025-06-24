import os

MODEL_PALETTE = {
    "DeepSeek-R1-Distill-Llama-8B": "#ff6f61",     # Bright coral red
    "Llama-3.1-8B": "#bb86fc",                     # Bright purple
    "Llama-3.1-8B-Instruct": "#9a4dff",            # Vivid violet

    "Mistral-7B-v0.1": "#4fc3f7",                  # Light cyan-blue
    "Mistral-7B-Instruct-v0.2": "#0288d1",         # Bright medium blue

    "gemma-3-4b-it": "#81c784",                    # Light green
    "Qwen2.5-7B-Instruct": "#ffb74d",              # Warm orange (amber)
}



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

def make_label(model, ft, context, style, oppu, with_model=True):
    def bold(val, prefix):
        return f"{prefix}{val}" if val == 0 else rf"$\bf{{{prefix}{val}}}$"
    if with_model:
        label = model
    else:
        label='BL'
    if(style):
        label = label + " + SE"
    if(context):
        label = label + " + CR"
    if(ft):
        label = label+" + FT"
    return label
    # return "_".join([
    #     model,
    #     bold(style, "style"),
    #     bold(context, "ctx"),
    #     bold(ft, "ft"),
    #     #bold(oppu, "oppu")
    # ])

def get_marker(row):
    if row['style'] == 0 and row['context'] == 0 and row['ft'] == 0 and row['oppu'] == 0:
        return '^'  # triangle
    elif row['context'] == 0 and row['ft'] == 0 and row['oppu'] == 0:
        return 's'  # square
    elif row['ft'] == 0 and row['oppu'] == 0:
        return 'd'  # pentagon
    elif row['oppu'] == 0:
        return 'o'  # circle
    else:
        return 'x'  # fallback marker for other cases if any

def get_marker_from_merged_row(row, suffix='_random'):
    keys = ['style', 'context', 'ft', 'oppu']  # keys expected by get_marker
    sub_row = {k: row[f"{k}{suffix}"] for k in keys}
    return get_marker(sub_row)
    