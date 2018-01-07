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
ARI_scores = np.loadtxt("ARI_norm.out", delimiter=',')
ARI = ARI_scores[:, 1]
ARI_prop = ARI_scores[:, 2]
ARI_sort_index = np.flip(np.argsort(ARI), 0)
ARI_prop_sort_index = np.flip(np.argsort(ARI_prop), 0)
ARI_sort = [ARI[k] for k in ARI_sort_index]

# Display top of ARI scores
print("Ranking | ARI score |     Features    | ARI propagation |     Features")
for k in range(20):
    index = ARI_sort_index[k]
    index_prop = ARI_prop_sort_index[k]
    print("%s | %s | %s | %s | %s"
          % (repr(k).rjust(7), ("%.4f" % ARI[index]).rjust(9),
             repr(features_sublists[index]).rjust(15),
             ("%.4f" % ARI_prop[index_prop]).rjust(15),
             repr(features_sublists[index_prop]).rjust(15)))

# Rank of all features
features_ranking = [features_sublists[ARI_sort_index[k]]
                    for k in range(len(features_sublists))]

all_feat_rank = np.argmax([len(features_ranking[k])
                           for k in range(len(features_ranking))])

print("Rank of all features: %d" % all_feat_rank)

# First 5 features list
for k in range(len(features_ranking)):
    if (len(features_ranking[k]) == 5):
        print("First 5 features list at rank %d: %s, ARI: %.4f" %
              (k, str(features_ranking[k]), ARI[k]))
        break

# Number of features per ARI ranking
features_ranking_len = [len(features_ranking[k])
                        for k in range(len(features_ranking))]
plt.plot(range(1, 601), features_ranking_len[:600])
plt.title("Number of features per ARI ranking")
plt.xlabel("ARI ranking")
plt.ylabel("Number of features")
plt.show()

# ARI
plt.plot(range(1, 600 + 1), ARI_sort[:600])
plt.title("ARI")
plt.xlabel("ARI ranking")
plt.ylabel("ARI")
plt.show()

# Distribution of ARI
plt.title("Distribution of ARI")
plt.xlabel("ARI")
plt.ylabel("Number of sublists of features")
plt.hist(ARI, bins=100)
plt.show()

# Percentile
prt = np.percentile(ARI, 99)
print("Percentile 99: %.4f" % prt)
