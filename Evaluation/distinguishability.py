#!/usr/bin/env python
# encoding: utf-8

import os
import argparse
import numpy as np

from collections import defaultdict
import sklearn.metrics as metrics
from util import *

def split_data(features, **kwargs):
    if 'root' in kwargs:
        root = kwargs['root']
    else:
        root = "."
    split_name = os.path.join(root, 'splits.json')
    splits = load_json(split_name)

    split_features = defaultdict(list)
    for video, feature in features.viewitems():
        if not splits.has_key(video):
            continue
        # split contains tuple (split, category)
        split = splits[video][0]
        split_features[split].append(feature)
    for video, feature in split_features.iteritems():
        split_features[video] = np.vstack(feature)
    return split_features

def run_exp(humancam, trajectories):
    def transform_features(split, split_x):
        train_x = []
        for i, x in split_x.viewitems():
            if i == split:
                test_x = x
            else:
                train_x.append(x)
        train_x = np.vstack(train_x)
        return train_x, test_x

    results = []
    for split in humancam:
        pos_train_x, pos_test_x = transform_features(split, humancam)
        neg_train_x, neg_test_x = transform_features(split, trajectories)

        train_x, train_y = construct_samples(pos_train_x, neg_train_x)
        test_x, test_y = construct_samples(pos_test_x, neg_test_x)

        model = fit_model(train_x, train_y)
        acc = model.score(test_x, test_y)
        err = (1. - acc) * 100
        results.append(err)
    results = np.array(results)
    result = results.mean(axis=0)
    std = results.std(axis=0)
    print("{:.2f}".format(result))
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('negative', type=str)
    args = parser.parse_args()

    humancam = load_humancam()
    humancam = split_data(humancam)

    trajectories = load_pkl(args.negative)
    trajectories = split_data(trajectories)

    run_exp(humancam, trajectories)

if __name__ == "__main__":
    main()
