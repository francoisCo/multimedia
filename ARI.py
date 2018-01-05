# -*- coding: utf-8 -*-
"""
ARI compuation for all subset of features
@author: francois courbier
"""
import numpy as np
from videomodel import VideoModel
from sklearn.cluster import KMeans
import sys
import itertools as it

import time
start_time = time.time()

# Instanciate model object
ml = VideoModel(videofile="06-11-22.mp4", features_nb=15)

# Get X, headers and videotime
if(len(sys.argv) == 2):
    # Case of command line argument with dataset
    ml.loadXfromfile(sys.argv[1])
else:
    # Case no command line argument
    ml.extractDataFromVideo()

# Instantiate Kmeans clustering
kmeans = KMeans(n_clusters=3, random_state=0)

# Compute all possible sublists of features
features_sublists = []
features_list = np.arange(ml.features_nb)
for k in range(1, ml.features_nb + 1):
    features_sublists = features_sublists +\
        list(it.combinations(features_list, k))

print("|"*100)
progress_rate = 0
ARI_res = np.zeros((0, 3))
sublists_nb = len(features_sublists)
for k in range(sublists_nb):
    # Keep only relevant features
    X_sub = ml.X[:, features_sublists[k]]

    # Compute ARI scores for all subdatasets (modulo 25)
    ARI_scores = ml.computeARI(X_sub, kmeans, plot=False)
    line = k, ARI_scores[0], ARI_scores[1]
    ARI_res = np.append(ARI_res, [line], axis=0)

    # Print progress
    current_progress_rate = int((k/sublists_nb)*100)
    if(current_progress_rate > progress_rate):
        sys.stdout.write("|")
        progress_rate = current_progress_rate

np.savetxt('ARI.out', ARI_res, delimiter=',')
print("\nARI saved in ARI.out")
print("--- %s seconds ---" % (time.time() - start_time))
