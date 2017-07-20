#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import json
import numpy as np

def load_json(path):
    if not os.path.exists(path):
        raise ValueError("Can't find {}".format(path))
    with open(path,'r') as fin:
        data = json.load(fin)
    return data

def load_humanEdit(**kwargs):
    """
    :root: path to directory contain the human edit trajectories
    :returns: humanEdit dict(videoId -> ndarray)
    """

    if 'root' in kwargs:
        root = kwargs['root']
    else:
        root = "."
    path = os.path.join(root, 'human_edit.json')
    data = load_json(path)

    humanEdit = {}
    for videoId, editors in data.viewitems():
        video_trajs = []
        for trajs in editors.viewvalues():
            for traj in trajs.viewvalues():
                traj = np.array(traj)
                video_trajs.append(traj)
        video_trajs = np.stack(video_trajs)
        humanEdit[videoId] = video_trajs * np.pi / 180.
    return humanEdit

def load_trajectories(path):
    data = load_json(path)
    trajectories = {}
    for videoId, trajs in data.viewitems():
        trajs = [np.array(traj) for traj in trajs.viewvalues()]
        trajs = np.stack(trajs)
        trajectories[videoId] = trajs * np.pi / 180.
    return trajectories

def compute_distance(trajs_src, trajs_dst, metric="cosine"):
    polars_src = trajs_src[:,:,0]
    polars_dst = trajs_dst[:,:,0]

    zs_src = np.sin(trajs_src[:,:,1])
    zs_dst = np.sin(trajs_dst[:,:,1])
    rs_src = np.cos(trajs_src[:,:,1])
    rs_dst = np.cos(trajs_dst[:,:,1])

    if metric == "overlap":
        # dst fov
        if trajs_dst.shape[2] == 3:
            dst_fov = trajs_dst[:,:,2]
        elif trajs_dst.shape[2] == 2:
            dst_fov = np.full(trajs_dst.shape[:2], np.pi * 65.5 / 180.)
        else:
            raise ValueError("Invalid dst trajectories shape.")
        # src fov
        if trajs_src.shape[2] == 3:
            src_fov = trajs_src[:,:,2]
        elif trajs_src.shape[2] == 2:
            src_fov = np.full(trajs_src.shape[:2], np.pi * 65.5 / 180.)
        else:
            raise ValueError("Invalid src trajectories shape.")

    distances = []
    for i in xrange(trajs_src.shape[0]):
        delta_polars = np.cos(polars_dst - polars_src[i,:])
        cos_sims = zs_src[i] * zs_dst + rs_src[i] * rs_dst * delta_polars
        if metric == "cosine":
            dists = cos_sims
        elif metric == "overlap":
            ids_0 = abs(cos_sims-1.) < 1e-8
            ids_pi = abs(cos_sims+1.) < 1e-8

            delta_angles = np.arccos(cos_sims)
            delta_angles[ids_0] = 0.
            delta_angles[ids_pi] = np.pi

            threshold = (src_fov[i] + dst_fov) / 2
            dists = 1. - np.divide(delta_angles, threshold)
            dists[dists<0.] = 0.
        else:
            raise ValueError("Invalid distance metric.")
        distances.append(dists)
    distances = np.stack(distances)
    return distances

def humanedit_metric(trajs, humanedits, pool='traj', metric='overlap'):
    results = {}
    for videoId in humanedits:
        if videoId not in trajs:
            raise ValueError("Missing video: {}".format(videoId))
        traj = trajs[videoId]
        traj_length = traj.shape[1]
        humanedit = humanedits[videoId][:,:traj_length,:]

        distances = compute_distance(traj, humanedit, metric=metric)
        if pool == 'traj':
            distance = distances.mean(axis=2).max(axis=1).mean()
        elif pool == 'frame':
            distance = distances.max(axis=1).mean(axis=1).mean()
        else:
            raise ValueError("Invalid pooling method.")
        results[videoId] = distance
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pool', choices=['traj', 'frame'], default='traj')
    parser.add_argument('--metric', choices=['cosine', 'overlap'], default='overlap')
    parser.add_argument('target', type=str)
    args = parser.parse_args()

    humanedits = load_humanEdit()
    trajs = load_trajectories(args.target)

    results = humanedit_metric(trajs, humanedits, pool=args.pool, metric=args.metric)
    result = np.array(results.values()).mean()
    print "{:.3f}".format(result)
