
import os
import cPickle as pickle
import json
import numpy as np

from sklearn.linear_model import LogisticRegression

def load_pkl(path):
    with open(path, 'rb') as fin:
        data = pickle.load(fin)
    return data

def load_json(path):
    with open(path, 'r') as fin:
        data = json.load(fin)
    return data

def load_humancam(**kwargs):
    """
    :root: path to directory contain the human edit trajectories
    :returns: humanEdit dict(videoId -> ndarray)
    """

    if 'root' in kwargs:
        root = kwargs['root']
    else:
        root = "data"
    path = os.path.join(root, 'humancam-c3d.pkl')
    features = load_pkl(path)
    return features

def construct_samples(pos_x, neg_x):
    x = np.vstack([pos_x, neg_x])
    y = np.hstack([np.ones(pos_x.shape[0]),
                   np.zeros(neg_x.shape[0])])
    return x, y

def fit_model(train_x, train_y):
    lr = LogisticRegression()
    lr.fit(train_x, train_y)
    return lr

def predict_model(lr, test_x):
    pos_idx = list(lr.classes_).index(1)
    predict_probs = lr.predict_proba(test_x)
    probs = predict_probs[:, pos_idx]
    return probs

