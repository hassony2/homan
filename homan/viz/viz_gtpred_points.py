# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
from libyana.conversions import npt


def viz_gtpred_points(fig=None,
                      images=None,
                      save_path="tmp.png",
                      pred_images=None,
                      point_s=1):
    sample_nb = len(images)
    col_nb = 1
    if pred_images is not None:
        col_nb += len(pred_images)
    fig_res = 4
    if fig is None:
        fig = plt.figure(figsize=(int(fig_res * (col_nb - 1)),
                                  int(sample_nb * fig_res)))
    else:
        fig.clf()
    for sample_idx in range(sample_nb):
        # First col
        ax = fig.add_subplot(sample_nb, col_nb, sample_idx * col_nb + 1)
        ax.axis("off")
        if images is not None and images[sample_idx] is not None:
            img = images[sample_idx]
            img = npt.numpify(img)
            ax.imshow(img)
        ax.set_title("input image")

        # Second col
        if pred_images is not None:
            for add_idx, pred_img_name in enumerate(pred_images):
                ax = fig.add_subplot(sample_nb, col_nb,
                                     sample_idx * col_nb + 2 + add_idx)
                img = pred_images[pred_img_name][sample_idx]
                img = npt.numpify(img)
                ax.imshow(img)
                ax.axis("off")
                ax.set_title(pred_img_name)
    fig.savefig(save_path, bbox_inches="tight")
