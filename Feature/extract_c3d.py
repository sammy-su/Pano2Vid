#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from util.data_io import load_pkl, dump_pkl, isvideo, count_frames

def generate_c3d_prototxt(src, dst_root):
    # Collect C3D entries for the input video
    if not os.path.isfile(src) or not isvideo(src):
        raise ValueError("{} is not a valid video file.".format(src))

    videoId = os.path.splitext(os.path.basename(src))[0]
    dst_dir = os.path.join(dst_root, videoId)
    if not os.path.isdir(dst_dir):
        try:
            os.makedirs(dst_dir)
        except:
            sys.stderr.write("Create {} fails.\n".format(dst_dir))

    try:
        frs = count_frames(src)
    except:
        raise IOError("Unable to read {}".format(src))

    entries = []
    start = 0
    step_size = 16
    dst_prefix = os.path.join(dst_dir, videoId)
    while (start+step_size+1) < frs:
        entry_in = "{0} {1} 0\n".format(src, start)
        entry_out = "{0}-{1:06d}\n".format(dst_prefix, start)
        entries.append((entry_in, entry_out))
        start += step_size

    # Output C3D prototxt
    path_in = os.path.join(dst_dir, "input.prototxt")
    path_out = os.path.join(dst_dir, "output.prototxt")
    with open(path_in, 'w') as fin, open(path_out, 'w') as fout:
        for entry_in, entry_out in entries:
            fin.write(entry_in)
            fout.write(entry_out)
    return dst_dir

def read_feature(path):
    with open(path, 'rb') as fin:
        s = np.fromfile(fin, dtype=np.int32, count=5)
        m = np.fromfile(fin, dtype=np.float32)
    return m

c3d_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "C3D")
c3d_bin = os.path.join(c3d_dir, "build/tools/extract_image_features.bin")
c3d_model_dir = os.path.join(c3d_dir, "examples/c3d_feature_extraction")
c3d_template = os.path.join(c3d_model_dir, "prototxt/c3d_sport1m_feature_extractor_video.prototxt")
c3d_model = os.path.join(c3d_model_dir, "conv3d_deepnetA_sport1m_iter_1900000")

def extract_c3d(dst_dir, use_gpu=False):
    batch_size = 25
    if not os.path.isdir(dst_dir):
        raise ValueError("{} does not exist.".format(dst_dir))

    # Build C3D input proto
    input_proto = os.path.join(dst_dir, "input.prototxt")
    c3d_proto_path = os.path.join(dst_dir, "c3d_extractor.prototxt")
    with open(c3d_proto_path, "w") as fout, open(c3d_template, "r") as fin:
        for line in fin:
            if line.rstrip().split()[0] == "source:":
                fout.write("    source: \"{}\"\n".format(input_proto))
            elif line.rstrip().split()[0] == "batch_size:":
                fout.write("    batch_size: {}\n".format(batch_size))
            else:
                fout.write(line)

    # Read output entries
    output_proto = os.path.join(dst_dir, "output.prototxt")
    with open(output_proto, 'r') as fin:
        output_entries = fin.readlines()
        num_clips = len(output_entries)

    layers = ['fc6-1', 'fc7-1']
    # Extract C3D
    num_batch = (num_clips + batch_size - 1) / batch_size
    template = ("GLOG_logtosterr=1 {0} {1} {2} "
                "{3} {4} {5} {6} {7}")
    gpu = 0 if use_gpu else -1
    cmd = template.format(c3d_bin, c3d_proto_path, c3d_model,
                          gpu, batch_size, num_batch, output_proto,
                          " ".join(layers))
    os.system(cmd)

    # Compact C3D input single pkl file
    features = {}
    for layer in layers:
        feature = []
        for line in output_entries:
            prefix = line.rstrip()
            feature_path = "{0}.{1}".format(prefix, layer)
            if not os.path.exists(feature_path):
                continue
            f = read_feature(feature_path)
            assert f.shape[0] == 4096
            feature.append(f)
        feature = np.vstack(feature).mean(axis=0)
        features[layer] = feature

    output = "{}.pkl".format(dst_dir)
    dump_pkl(output, features)
    shutil.rmtree(dst_dir)
    return output

def generate_feature(src, dst_root):
    if not os.path.isdir(dst_root):
        os.makedirs(dst_root)

    dst_dir = generate_c3d_prototxt(src, dst_root)
    dst = extract_c3d(dst_dir)
    return dst

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dst', dest='dst', type=str, default=None)
    parser.add_argument('src', dest='src', type=str)
    args = parser.parse_args()

    if args.dst is None:
        dst_root = os.path.dirname(os.path.abspath(__file__))
    else:
        dst_root = args.dst
    generate_feature(args.src, dst_root)

if __name__ == "__main__":
    main()

