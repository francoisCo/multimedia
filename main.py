# -*- coding: utf-8 -*-
"""
Clustering of 06-11-22.mp4
@author: francois courbier
"""
import numpy as np
from videomodel import VideoModel
from sklearn.cluster import KMeans
import sys
from sklearn.preprocessing import StandardScaler

# Instanciate model object
ml = VideoModel(videofile="06-11-22.mp4", features_nb=15)

# Get X, headers and videotime
if(len(sys.argv) == 2):
    # Case of command line argument with dataset
    ml.loadXfromfile(sys.argv[1])
else:
    # Case no command line argument
    ml.extractDataFromVideo()

# Normalisation
ml.X = StandardScaler().fit_transform(ml.X)

# Display features through time
x = np.arange(1, ml.videotime + 1)
for i in range(0, ml.features_nb):
    ml.displayFigureSeconds(ml.X[::25, i], x, ml.headers[i])

# Compute and display Kmeans clustering
kmeans = KMeans(n_clusters=3,
                random_state=0).fit(ml.X[::25, [2, 3, 5, 9, 12, 13]])
ml.displayFigureSeconds(kmeans.labels_, x,
                        "Clusters (k-means) for subdataset 1st frames / sec ")

# Keep only relevant features
X_sub = ml.X[:, [2, 3, 5, 9, 12, 13]]

# Compute ARI scores for all subdatasets (modulo 25)
ml.computeARI(X_sub, kmeans, plot=True)
