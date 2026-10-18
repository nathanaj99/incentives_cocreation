"""
Figure S14: robustness of the PCA view (Fig. 1c) to the embedding model.
2-D PCA coordinates for each embedding model are precomputed in PCA.pkl,
in the order of common.build_text_corpus. top = Study 1, bottom = Study 2.

Outputs: figs/figure_s14_top.png, figs/figure_s14_bottom.png
"""

import pickle

import matplotlib.pyplot as plt

from common import (build_text_corpus, pca_group_handles, pca_stage_handles, pretty_map,
                    processed, save_fig, scatter_pca, set_style)

# Small multiples: designed at a smaller base font so ticks don't collide
set_style(font_size=15)


def plot_study(study, name):
    _, group, stage, _ = build_text_corpus(study)
    with open(processed(study, "PCA.pkl"), "rb") as f:
        pca = pickle.load(f)  # {embedding model: (n_texts x 2) coordinates}

    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(20, 8))
    for ix, (embedding, X2) in enumerate(pca.items()):
        a = ax[ix // 4][ix % 4]
        scatter_pca(a, X2, group, stage)
        a.set_xlabel("PC1" if ix // 4 == 1 else "")
        a.set_ylabel("PC2" if ix % 4 == 0 else "")
        a.set_title(pretty_map[embedding])

    # 7 models -> the last panel holds both legends
    ax[1][3].axis("off")
    leg1 = ax[1][3].legend(handles=pca_group_handles(), title="group", loc="upper left",
                           bbox_to_anchor=(0.15, 0.95), fontsize=16, frameon=False)
    ax[1][3].legend(handles=pca_stage_handles(), title=None, loc="upper left",
                    bbox_to_anchor=(0.15, 0.30), fontsize=16, frameon=False)
    ax[1][3].add_artist(leg1)

    plt.subplots_adjust(wspace=0.2, hspace=0.25)
    save_fig(fig, name)


plot_study(1, "figure_s14_top")
plot_study(2, "figure_s14_bottom")
