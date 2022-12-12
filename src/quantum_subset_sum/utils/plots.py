from typing import Tuple, List

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def box_strip_plot(df,
                   x: str,
                   y: str,
                   hue: str = None,
                   max_line: int = None,
                   title: str = None,
                   hue_order: List = None,
                   hue_length: int = None,
                   y_lim=None,
                   add_strip_plot: bool = False,
                   figsize: Tuple = (7, 6)):
    f, ax = plt.subplots(figsize=figsize)
    ax.set_yscale("symlog")

    sns.boxplot(x=df[x], y=df[y],
                hue=None if hue is None else df[hue],
                hue_order=hue_order)
    if add_strip_plot:
        sns.stripplot(x=df[x], y=df[y],
                      hue=None if hue is None else df[hue],
                      hue_order=hue_order, dodge=True, edgecolor='gray', linewidth=1)

    # Get the handles and labels. For this example it'll be 2 tuples
    # of length 4 each.
    handles, labels = ax.get_legend_handles_labels()
    # When creating the legend, only use the first two elements
    # to effectively remove the last two.
    if hue_order is not None:
        hue_length = len(hue_order)
    if hue_length is not None:
        l = plt.legend(handles[0:hue_length], labels[0:hue_length], bbox_to_anchor=(1.05, 1), loc=2,
                       borderaxespad=0.)

    if max_line is not None:
        ax.axhline(max_line, ls='--')
    ax.set_ylim(y_lim)
    plt.title(title)
    return f
