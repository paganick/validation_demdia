#!/usr/bin/env python3
"""
Quantitative check for MODEL_PALETTE (simulation/src/plotting_utils.py): does the
9-model categorical palette stay distinguishable in grayscale (print) and under
simulated color-vision deficiency, not just "by eye"?

Motivated by Reviewer #1's R1.3 comment ("I printed the paper on a black-and-white
printer ... some of the figures are really unaccessible in grayscale").

For both the current MODEL_PALETTE and the pre-revision palette it replaced, this:
  1. Converts each hex color to CIE Lab and reports L* (perceptual lightness --
     what grayscale printing preserves), sorted.
  2. Reports pairwise Lab (CIE76) distance under normal vision.
  3. Reports pairwise |delta L*| (grayscale-only distinguishability).
  4. Simulates protanopia/deuteranopia (Machado/Coblis-style linear-RGB matrices,
     100% severity) and reports pairwise Lab distance under each.

Rule of thumb used throughout: dE/dL >= 12 is comfortably distinguishable, 8-12 is
an acceptable floor given every figure also carries the model name as a text label
(color is never the only identity cue), and < 8 is a real risk of two models
looking identical.

Usage:
    python analysis/validate_model_palette.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from simulation.src.plotting_utils import MODEL_PALETTE

PRE_REVISION_PALETTE = {
    "DeepSeek-R1-Distill-Llama-8B": "#ff6f61",
    "Llama-3.1-8B": "#bb86fc",
    "Llama-3.1-8B-Instruct": "#9a4dff",
    "Llama-3.1-70B": "#5a189a",
    "Mistral-7B-v0.1": "#4fc3f7",
    "Mistral-7B-Instruct-v0.2": "#0288d1",
    "gemma-3-4b-it": "#81c784",
    "Qwen2.5-7B-Instruct": "#ffb74d",
    "Apertus-8B-2509": "#a1887f",
}


def hex_to_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)]) / 255.0


def srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def rgb_to_xyz(rgb_lin):
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    return M @ rgb_lin


def xyz_to_lab(xyz):
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    x, y, z = xyz[0] / Xn, xyz[1] / Yn, xyz[2] / Zn

    def f(t):
        d = 6 / 29
        return np.where(t > d ** 3, t ** (1 / 3), t / (3 * d ** 2) + 4 / 29)

    fx, fy, fz = f(x), f(y), f(z)
    return np.array([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)])


def hex_to_lab(h):
    return xyz_to_lab(rgb_to_xyz(srgb_to_linear(hex_to_rgb(h))))


# Machado/Coblis-style simplified CVD simulation matrices, applied in linear RGB,
# 100% severity. Approximate but standard practice for a quick distinguishability
# check without extra dependencies (e.g. colorspacious).
CVD_MATRICES = {
    "protanopia": np.array([
        [0.567, 0.433, 0.000],
        [0.558, 0.442, 0.000],
        [0.000, 0.242, 0.758],
    ]),
    "deuteranopia": np.array([
        [0.625, 0.375, 0.000],
        [0.700, 0.300, 0.000],
        [0.000, 0.300, 0.700],
    ]),
}


def simulate_cvd_lab(hexcode, kind):
    lin = srgb_to_linear(hex_to_rgb(hexcode))
    sim_srgb = linear_to_srgb(CVD_MATRICES[kind] @ lin)
    return xyz_to_lab(rgb_to_xyz(srgb_to_linear(sim_srgb)))


def report(palette, label):
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    names = list(palette.keys())
    labs = {n: hex_to_lab(palette[n]) for n in names}

    print("\n-- Grayscale (L*) values, sorted --")
    for n in sorted(names, key=lambda n: labs[n][0]):
        print(f"  {n:30s} L*={labs[n][0]:5.1f}")

    def worst_pair(dist_fn, threshold):
        worst = None
        for i, n1 in enumerate(names):
            for n2 in names[i + 1:]:
                d = dist_fn(n1, n2)
                if worst is None or d < worst[0]:
                    worst = (d, n1, n2)
        flag = "  <-- LOW" if worst[0] < threshold else ""
        print(f"  min pairwise: {worst[0]:5.1f}  ({worst[1]} vs {worst[2]}){flag}")
        return worst[0]

    print("\n-- Pairwise Lab distance, normal vision (target >= 12) --")
    worst_pair(lambda a, b: np.linalg.norm(labs[a] - labs[b]), 12)

    print("\n-- Grayscale-only |delta L*| (target >= 8) --")
    worst_pair(lambda a, b: abs(labs[a][0] - labs[b][0]), 8)

    for kind in ["protanopia", "deuteranopia"]:
        sim_labs = {n: simulate_cvd_lab(palette[n], kind) for n in names}
        print(f"\n-- Pairwise Lab distance under {kind} (target >= 12) --")
        worst_pair(lambda a, b: np.linalg.norm(sim_labs[a] - sim_labs[b]), 12)


def main():
    report(PRE_REVISION_PALETTE, "Pre-revision MODEL_PALETTE (before R1.3 fix)")
    report(MODEL_PALETTE, "Current MODEL_PALETTE (plotting_utils.py)")


if __name__ == "__main__":
    main()
