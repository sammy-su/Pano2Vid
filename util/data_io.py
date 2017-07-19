#!/usr/bin/env python
# encoding: utf-8

import os
import sys
import cPickle
import subprocess
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_ROOT = os.path.join(ROOT, "videos")

def extract_duration(path):
    length_regexp = 'Duration: (\d{2}):(\d{2}):(\d{2})\.\d+,'
    re_length = re.compile(length_regexp)

    cmd = "ffprobe {} 2>&1 | grep Duration".format(path)
    output = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).stdout.read()
    matches = re_length.search(output)
    if matches:
        video_length = int(matches.group(1)) * 3600 + \
                       int(matches.group(2)) * 60 + \
                       int(matches.group(3))
        video = "/".join(path.split("/")[-2:])
    else:
        raise IOError("Can't read vidoe length for {}".format(path))
    return video_length

def count_frames(path):
    ffprobe_template = ("ffprobe -v error -count_frames "
                        "-select_streams v:0 "
                        "-show_entries stream=nb_read_frames "
                        "-of default=nokey=1:noprint_wrappers=1 {}")
    ffprobe_cmd = ffprobe_template.format(path)
    FNULL = open(os.devnull, 'w')
    output = subprocess.Popen(ffprobe_cmd,
                              shell=True,
                              stderr=FNULL,
                              stdout=subprocess.PIPE).stdout.read()
    frs = int(output)
    return frs

def isvideo(path):
    valid_suffix = [".mkv", ".mp4", ".webm", ".avi"]
    video, suffix = os.path.splitext(os.path.basename(path))
    return suffix in valid_suffix

def load_pkl(path):
    with open(path, 'rb') as fin:
        data = cPickle.load(fin)
    return data

def dump_pkl(path, obj):
    if os.path.exists(path):
        raise ValueError("{} exists!".format(path))
    with open(path, 'wb') as fout:
        cPickle.dump(obj, fout, protocol=2)

def find_video_path(video, categories=None):
    if categories is None:
        categories = ["hiking", "mountain_climbing", "parade", "soccer"]
    for category in categories:
        category_dir = os.path.join(VIDEO_ROOT, category)
        suffixes = ["mp4", "mkv", "webm"]
        for suffix in suffixes:
            video_in = os.path.join(category_dir, "{0}.{1}".format(video, suffix))
            if os.path.exists(video_in):
                return video_in
    raise ValueError("Can't find video file for {}".format(video))

