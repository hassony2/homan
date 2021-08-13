#!/usr/bin/env python
# -*- coding: utf-8 -*-

from IPython.display import HTML
from base64 import b64encode
import os


def display_video(video_path, compressed_path="tmp.mp4"):
    os.system(f"ffmpeg -i {video_path} -vcodec libx264 {compressed_path}")

    # Show video
    mp4 = open(compressed_path, 'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    vid_html = HTML("""
  <video width=400 controls>
        <source src="%s" type="video/mp4">
  </video>
  """ % data_url)
    return vid_html
