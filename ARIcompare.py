# -*- coding: utf-8 -*-
"""
Analysis of ARI scores
@author: francois courbier
"""
import numpy as np
import itertools as it
from videomodel import VideoModel
import matplotlib.pyplot as plt

# Instanciate model object
ml = VideoModel(videofile="06-11-22.mp4", features_nb=15)

# Compute all possible sublists of features
features_sublists = []
features_list = np.arange(ml.features_nb)
for k in range(1, ml.features_nb + 1):
    features_sublists = features_sublists +\
        list(it.combinations(features_list, k))

# Get ARI scores and sort them
ARI_scores = np.loadtxt("ARI.out", delimiter=',')
ARI = ARI_scores[:, 1]
ARI_scores = np.loadtxt("ARI_norm.out", delimiter=',')
ARI_norm = ARI_scores[:, 1]
ARI_sort_index = np.flip(np.argsort(ARI), 0)
ARI_norm_sort_index = np.flip(np.argsort(ARI_norm), 0)

# Display top of ARI scores
print("Ranking | ARI score |     Features    |ARI normalisation|    Features")
for k in range(20):
    index = ARI_sort_index[k]
    index_norm = ARI_norm_sort_index[k]
    print("%s | %s | %s | %s | %s"
          % (repr(k).rjust(7), ("%.4f" % ARI[index]).rjust(9),
             repr(features_sublists[index]).rjust(15),
             ("%.4f" % ARI_norm[index_norm]).rjust(15),
             repr(features_sublists[index_norm]).rjust(15)))
