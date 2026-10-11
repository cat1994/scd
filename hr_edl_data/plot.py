"""Portable figure styling; TeX is deliberately optional."""
import os
import matplotlib.pyplot as plt

def set_plt_params(use_serif=True, font_size=24, color="0"):
    # MathText is the reproducible default and needs no system TeX packages.
    plt.rcParams.update({"text.usetex": os.environ.get("SCD_USE_TEX") == "1",
                         "font.family": "serif" if use_serif else "sans-serif",
                         "font.size": font_size, "pdf.fonttype": 42,
                         "text.color": color, "axes.labelcolor": color})
    if plt.rcParams["text.usetex"]:
        plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}\usepackage{siunitx}"
