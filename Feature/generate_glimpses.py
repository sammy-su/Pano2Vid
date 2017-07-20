#!/usr/bin/env python
# encoding: utf-8

import os
import sys
import argparse
import numpy as np
import subprocess

from functools import partial
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from util.data_io import extract_duration
from util.VideoRecorder import VideoRecorder

angles = [(x, y) for x in xrange(-180,180,20)
                 for y in [-75,-45,-30, -20, -10, 0, 10, 20, 30, 45, 75]]

fovs = {
    "050": 104.3,
    "100": 65.5,
    "150": 46.4,
}

def compose_view(angle, f, src, dst_dir):
    x, y = angle
    if x < 0:
        suffix_x = "l{:03d}".format(abs(x))
    else:
        suffix_x = "r{:03d}".format(x)
    x0 = np.pi * x / 180.
    if y < 0:
        suffix_y = "d{:02d}".format(abs(y))
    else:
        suffix_y = "u{:02d}".format(y)
    y0 = np.pi * y / 180.
    fov = fovs[f]

    clipId = os.path.splitext(os.path.basename(src))[0]
    output = os.path.join(dst_dir,
                          "{0}-zoom{1}-{2}{3}".format(clipId, f, suffix_x, suffix_y))
    sys.stderr.write("Start generate {}\n".format(output))
    recorder = VideoRecorder(src, output, x0=x0, y0=y0, view_angle=fov)
    finished = False
    while not finished:
        try:
            finished = recorder.forward()
        except ZeroDivisionError as e:
            sys.stderr.write("Division by zero in {}\n".format(output))
            break

def split_video(src, output_dir):
    clip_length = 5
    stride = 5
    template = "ffmpeg -i {0} -c:v libx264 -an -ss {1} -t {2} {3} 2>/dev/null"

    video_length = extract_duration(src)
    videoId = os.path.splitext(os.path.basename(src))[0]

    clips = []
    start = 0
    clipId = 0
    while start < video_length:
        clipId += 1
        dst = os.path.join(output_dir, "{0}-c{1:03d}.mp4".format(videoId, clipId))
        cmd = template.format(src, start, clip_length, dst)
        #output = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).stdout.read()

        start += stride
        clips.append(dst)
    return clips

def generate_glimpses(src, output_dir, parallel=True):
    clips = split_video(src, output_dir)

    for clip in clips:
        for f in fovs:
            compose_view_ = partial(compose_view,
                                    f=f,
                                    src=clip,
                                    dst_dir=output_dir)
            if parallel:
                pool = Pool()
                pool.map(compose_view_, angles)
            else:
                for angle in angles:
                    compose_view_(angle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dst', dest="dst", type=str, default=None)
    parser.add_argument('src', type=str)
    args = parser.parse_args()

    videoId = os.path.splitext(os.path.basename(args.src))[0]
    if args.dst is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), videoId)
    else:
        output_dir = args.dst

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    generate_glimpses(args.src, output_dir)

if __name__ == "__main__":
    main()
