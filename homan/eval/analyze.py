#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import defaultdict
from copy import deepcopy
import numpy as np
from homan.eval import logutils, evalviz

import pandas as pd
from pathlib import Path


def make_exp_html(df_data,
                  plots,
                  metric_names=None,
                  destination=None,
                  compact=True,
                  sort_loss="add-s_obj",
                  drop_redundant=True):

    if not isinstance(destination, Path):
        destination = Path(destination)
    destination.mkdir(exist_ok=True, parents=True)
    main_plot_str = logutils.make_compare_plots(plots,
                                                local_folder=destination)
    if (sort_loss is None) or compact:
        print(df_data)
    else:
        print(df_data.sort_values(sort_loss))
    df_html = logutils.df2html(df_data,
                               local_folder=destination,
                               drop_redundant=drop_redundant,
                               collapsible=not compact)
    with (destination / "raw.html").open("w") as h_f:
        h_f.write(df_html)

    with open(destination / "add_js.txt", "rt") as j_f:
        js_str = j_f.read()
    with open("htmlassets/index.html", "rt") as t_f:
        html_str = t_f.read()
    with open(destination / "raw.html", "rt") as t_f:
        table_str = t_f.read()
    full_html_str = (html_str.replace("JSPLACEHOLDER", js_str).replace(
        "TABLEPLACEHOLDER", table_str).replace("PLOTPLACEHOLDER",
                                               main_plot_str))
    with open(destination / "index.html", "wt") as h_f:
        h_f.write(full_html_str)
    print("Average metrics")


def parse_res(res,
              folder,
              monitor_metrics,
              compact=True,
              gifs=False,
              no_videos=True,
              plots=None,
              video_resize=0.5):
    if plots is None:
        plots = defaultdict(list)
    opts = res["opts"]
    if compact:
        res_data = {}
    else:
        res_data = deepcopy(res["opts"])

    # Get last loss values
    if not compact:
        for metric in res["losses"]:
            res_data[metric] = res["losses"][metric][-1]
    else:
        if "losses" in res:
            for metric in ["iou_object", "v2d_person"]:
                res_data[metric] = res["losses"][metric][-1]
            if "loss" in res["losses"]:
                res_data["full_loss"] = res["losses"]["loss"][-1]
    # # Show number of metric steps
    for metric in monitor_metrics:
        if not compact:
            if metric in res["losses"]:
                res_data[f"{metric}_plot_vals"] = tuple(res["losses"][metric])
        if metric in res["losses"]:
            plots[metric].append(res["losses"][metric])
    metric_names = []
    all_errs = {}
    for metric in res["metrics"]:
        # if "chamfer" not in metric:
        res_data[metric] = np.mean(res["metrics"][metric])
        all_errs[metric] = res['metrics'][metric]
        if "init" not in metric:
            metric_names.append(metric)
            metric_names.append(f"{metric}_init")

    # Get last optimized image
    img_paths = res["imgs"]
    img_path = img_paths[list(img_paths)[-1]]
    for img_name, img_path in res["show_img_paths"].items():
        res_data[f"{img_name}_img_path"] = img_path
    res_data["final_video_path"] = str(folder / "final_points.mp4")

    # Generate gif
    if gifs and (not compact):
        gif_path = folder / "optim.gif"
        evalviz.make_gif(img_paths.values(), gif_path)
        res_data["optim_img_path"] = str(gif_path)
    # if not no_videos and (not compact):
    if not no_videos:
        video_path = folder / "optim.webm"
        evalviz.make_video(img_paths.values(),
                           video_path,
                           resize_factor=video_resize)
        res_data["optim_video_path"] = str(video_path)

    # Add folder root
    res_data["folder"] = str(folder)
    return opts, res_data, plots, metric_names, all_errs
