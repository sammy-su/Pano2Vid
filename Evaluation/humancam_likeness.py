#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np

from collections import defaultdict
from multiprocessing import Pool

from util import *

def run_fold(video, humancam, trajectories):
    # Construct feature and  train model
    neg_train_x = []
    test_x = {}
    for approach, features in trajectories.viewitems():
        for v, f in features.viewitems():
            if v == video:
                test_x[approach] = f
            else:
                neg_train_x.append(f)
    neg_train_x = np.vstack(neg_train_x)
    train_x, train_y = construct_samples(humancam, neg_train_x)
    lr = fit_model(train_x, train_y)

    # Rank video by prob
    instances = []
    for approach, feature in test_x.viewitems():
        probs = predict_model(lr, feature)
        for prob in probs:
            instances.append((prob, approach))
    instances.sort(key=lambda x: x[0], reverse=True)

    # Compute average ranking
    num_instance = float(len(instances))
    rankings = defaultdict(list)
    for i, instance in enumerate(instances,1):
        rank = i / num_instance
        approach = instance[1]
        rankings[approach].append(rank)

    results = [np.array(rankings[approach]).mean() for approach in trajectories]
    display_results(results, prefix=video)
    return results

def display_results(results, prefix=None):
    result_str = "\t".join(["{:.3f}".format(r) for r in results])
    output = "{0}\t{1}\n".format(prefix, result_str)
    sys.stderr.write(output)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parallel', action="store_true")
    parser.add_argument('negatives', nargs='+', type=str)
    args = parser.parse_args()

    humancam = load_humancam()
    humancam = np.vstack(f for f in humancam.viewvalues())

    trajectories = {}
    for path in args.negatives:
        name = os.path.splitext(os.path.basename(path))[0]
        features = load_pkl(path)
        trajectories[name] = features

    videos = [set(f.keys()) for f in trajectories.viewvalues()]
    videos = set.intersection(*videos)

    # Display the evaluated methods
    header = "\t".join(t for t in trajectories.keys())
    sys.stdout.write("{}\n".format(header))

    if args.parallel:
        pool = Pool()
    results = []
    for video in videos:
        if args.parallel:
            result = pool.apply_async(run_fold, args=(video, humancam, trajectories))
        else:
            result = run_fold(video, humancam, trajectories)
        results.append(result)

    if args.parallel:
        pool.close()
        pool.join()
        results = [result.get() for result in results]

    mean = np.mean(np.array(results), axis=0)
    display_results(mean, prefix="Mean")

if __name__ == "__main__":
    main()
