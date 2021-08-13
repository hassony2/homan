#!/usr/bin/env python
# -*- coding: utf-8 -*-

from itertools import cycle
import os
from functools import partial
from pathlib import Path
import shutil

from bokeh import plotting as plt
from bokeh import embed, layouts, palettes
from bokeh.models import HoverTool
import pandas as pd

# Creat unique ids for each new html div
HTML_IDX = [0]

COLORS = (palettes.Colorblind[max(palettes.Colorblind)] +
          palettes.Bokeh[max(palettes.Bokeh)])
DASH_PATTERNS = ("solid", "dashed")


def drop_redundant_columns(df):
    """
    If dataframe contains multiple lines, drop the ones for which the column
    contains equal values
    """
    if len(df) > 1:
        nunique = df.apply(pd.Series.nunique)
        # Drop columns with all identical values or all None
        cols_to_drop = nunique[nunique <= 1].index
        print(f"Dropping {list(cols_to_drop)}")
        df = df.drop(cols_to_drop, axis=1)
    return df


def make_collapsible(html_str, collapsible_idx=0):
    """
    Create collapsible button to selectively hide large html items such as images
    """
    pref = (
        f'<button data-toggle="collapse" data-target="#demo{collapsible_idx}">'
        "Toggle show image</button>"
        f'<div id="demo{collapsible_idx}" class="collapse">')
    suf = "</div>"
    return pref + html_str + suf


def make_compare_plots(plots,
                       local_folder,
                       tools="pan,wheel_zoom,box_zoom,reset,save"):
    bokeh_figs = []
    for metric, metric_vals in plots.items():
        cycle_colors = cycle(COLORS)
        cycle_dash_patterns = cycle(DASH_PATTERNS)
        bokeh_p = plt.figure(
            tools=tools,
            x_axis_label="iter_steps",
            title=metric,
        )
        bokeh_p.add_tools(HoverTool())
        for run_idx, vals in enumerate(metric_vals):
            color = next(cycle_colors)
            dash_pattern = next(cycle_dash_patterns)
            bokeh_p.line(
                list(range(len(vals))),
                vals,
                legend_label=f"{run_idx:03d}",
                color=color,
                line_dash=dash_pattern,
            )
        bokeh_figs.append(bokeh_p)
    bokeh_grid = layouts.gridplot([bokeh_figs])
    js_str, html_str = embed.components(bokeh_grid)
    with (local_folder / "add_js.txt").open("at") as js_f:
        js_f.write(js_str)
    return html_str


def plotvals2bokeh(plot_vals,
                   local_folder,
                   tools="pan,wheel_zoom,box_zoom,reset,save"):
    bokeh_p = plt.figure(tools=tools, height=100, width=150)
    bokeh_p.line(list(range(len(plot_vals))), plot_vals)
    bokeh_p.add_tools(HoverTool())
    js_str, html_str = embed.components(bokeh_p)
    with (local_folder / "add_js.txt").open("at") as js_f:
        js_f.write(js_str)

    return html_str


def path2video(path, local_folder="", collapsible=True, call_nb=HTML_IDX):
    if local_folder:
        local_folder = Path(local_folder) / "video"
        local_folder.mkdir(exist_ok=True, parents=True)

        ext = str(path.split(".")[-1])
        video_name = f"{call_nb[0]:04d}.{ext}"
        dest_img_path = local_folder / video_name
        shutil.copy(path, dest_img_path)
        rel_path = os.path.join("video", video_name)
    else:
        rel_path = path

    # Keep track of count number
    call_nb[0] += 1
    vid_str = ('<video controls> <source src="' + str(rel_path) +
               '" type="video/webm"></video>')
    if collapsible:
        vid_str = make_collapsible(vid_str, call_nb[0])
    return vid_str


def path2img(
    path,
    local_folder="",
    collapsible=True,
    call_nb=HTML_IDX,
):
    if local_folder:
        local_folder = Path(local_folder) / "imgs"
        local_folder.mkdir(exist_ok=True, parents=True)

        ext = str(path.split(".")[-1])
        img_name = f"{call_nb[0]:04d}.{ext}"
        dest_img_path = local_folder / img_name
        shutil.copy(path, dest_img_path)
        rel_path = os.path.join("imgs", img_name)
    else:
        rel_path = path

    # Keep track of count number
    call_nb[0] += 1
    img_str = '<img src="' + str(rel_path) + '"/>'
    if collapsible:
        img_str = make_collapsible(img_str, call_nb[0])
    return img_str


def highlight_row(data, row_idx=[1]):
    cycle_colors = cycle(COLORS)
    for idx in range(row_idx[0]):
        color = next(cycle_colors)
    row_idx[0] += 1
    color_strings = [f"background-color: {color}" for _ in range(len(data))]
    return color_strings


def highlight_cell(data, row_idx=[1]):
    cycle_colors = cycle(COLORS)
    for idx in range(row_idx[0]):
        color = next(cycle_colors)
    row_idx[0] += 1
    col_str = f'<div style="background-color:{color}">' + str(data) + "<div/>"
    return col_str


def df2html(df, local_folder="", drop_redundant=True, collapsible=True):
    """
    Convert df to html table, getting images for fields which contain 'img_path'
    in their name.
    """
    keys = list(df.keys())
    format_dicts = {}
    for key in keys:
        if "img_path" in key:
            format_dicts[key] = partial(path2img,
                                        local_folder=local_folder,
                                        collapsible=collapsible)
        elif "video_path" in key:
            format_dicts[key] = partial(path2video,
                                        local_folder=local_folder,
                                        collapsible=collapsible)
        elif key.endswith("plot_vals"):
            format_dicts[key] = partial(plotvals2bokeh,
                                        local_folder=local_folder)
        elif key in ["lr", "optimizer"]:
            format_dicts[key] = highlight_cell

    if drop_redundant:
        df = drop_redundant_columns(df)

    df_html = df.to_html(escape=False, formatters=format_dicts)
    # for col, formatter in format_dicts.items():
    #     df[col] = df[col].apply(formatter)
    # df_styled = df.style.apply(highlight_row, axis=1)
    # df_html = df_styled.render(escape=False)

    return df_html
