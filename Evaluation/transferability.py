#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np

import sklearn.metrics as metrics
from collections import defaultdict
from util import *

def run_multiclass(train, test):
    def construct_multiclass(data):
        X = []
        y = []
        for label, category in enumerate(categories):
            feature = data[category]
            X.append(feature)
            y.append(np.full(feature.shape[0], label, dtype=np.int))
        X = np.vstack(X)
        y = np.hstack(y)
        return y, X

    categories = sorted(train.keys())
    train_y, train_x = construct_multiclass(train)
    test_y, test_x = construct_multiclass(test)

    lr = fit_model(train_x, train_y)
    predict_y = lr.predict(test_x)
    acc = metrics.accuracy_score(test_y, predict_y)
    return acc

def construct_category_features(features):
    category_features = defaultdict(list)
    splits = load_json('splits.json')

    for v, f in features.viewitems():
        if v not in splits:
            continue
        c = splits[v][1]
        category_features[c].append(f)
    for c, f in category_features.items():
        category_features[c] = np.vstack(f)
    return category_features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('test', type=str)
    args = parser.parse_args()

    humancam = load_humancam()
    humancam = construct_category_features(humancam)

    trajectories = load_pkl(args.test)
    trajectories = construct_category_features(trajectories)

    human2auto = run_multiclass(humancam, trajectories)
    auto2human = run_multiclass(trajectories, humancam)
    print "{0:.3f}\t{1:.3f}".format(human2auto, auto2human)

if __name__ == "__main__":
    main()
