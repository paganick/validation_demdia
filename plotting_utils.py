import os

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

def make_label(model, ft, context, style, oppu):
    def bold(val, prefix):
        return f"{prefix}{val}" if val == 0 else rf"$\bf{{{prefix}{val}}}$"

    return "_".join([
        model,
        bold(style, "style"),
        bold(context, "ctx"),
        bold(ft, "ft"),
        bold(oppu, "oppu")
    ])
