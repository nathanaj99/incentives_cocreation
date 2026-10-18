"""
Figure S12: similarity of the first AI draft vs. the final submission, by
group, for every metric. top = Study 1, bottom = Study 2.

Outputs: figs/figure_s12_top.png, figs/figure_s12_bottom.png
"""

import matplotlib.pyplot as plt
import pandas as pd

from common import (metric_cols, plot_similarity_distribution, pretty_map, processed,
                    relabel_group_legend, save_fig, set_style)

# Small multiples: designed at a smaller base font so ticks don't collide
set_style(font_size=15)


def plot_study(study, name):
    Y = pd.read_csv(processed(study, "Y.csv"))
    Y_drafts = pd.read_csv(processed(study, "Y_first_draft.csv"))
    y_cols = metric_cols(Y)
    Y["type"] = "Final"
    Y_drafts["type"] = "AI 1st Draft"

    fig, ax = plt.subplots(nrows=2, ncols=6, figsize=(24, 8))
    for idx, col in enumerate(y_cols):
        a = ax[idx // 6][idx % 6]
        cols = ["participant_id", "group_key", "type", col]
        plot_similarity_distribution(a, pd.concat([Y[cols], Y_drafts[cols]]), col, x="type",
                                     order=["AI 1st Draft", "Final"])
        a.set_ylabel("Similarity" if idx % 6 == 0 else "", fontsize=16)
        a.set_title(pretty_map[col], fontsize=18)
        a.grid(True, axis="y")
        if idx == 0:  # legend on the top-left panel only
            relabel_group_legend(a)
        else:
            a.legend_.remove()
        a.set_xlabel("")

    plt.subplots_adjust(wspace=0.25, hspace=0.25)
    save_fig(fig, name)


plot_study(1, "figure_s12_top")
plot_study(2, "figure_s12_bottom")
