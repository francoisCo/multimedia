# -*- coding: utf-8 -*-
"""
Display features evolution though time of 06-11-22.mp4
@author: francois courbier
"""
import numpy as np
import cv2
import background as bg
from sklearn.cluster import KMeans

features_nb = 15
X = np.zeros((0, features_nb))
headers = []

# Get video and frame rate
video = cv2.VideoCapture('06-11-22.mp4')
fps = video.get(cv2.CAP_PROP_FPS)

# Extract features from video frames
nbrFrame = 0
ret = True
while ret:
    ret, frame = video.read()
    nbrFrame += 1
    videotime = nbrFrame / fps
    if (nbrFrame % 25 == 1):
        features, headers = bg.computeFeatures(frame)
        X = np.append(X, [features], axis=0)

# Display features through time
x = np.arange(1, videotime + 1)
for i in range(0, features_nb):
    bg.displayFigureSeconds(X[:, i], x, headers[i])

# Compute and display Kmeans clustering
kmeans = KMeans(n_clusters=3, random_state=0).fit(X[:, [2, 3, 5, 9]])
bg.displayFigureSeconds(kmeans.labels_, x, "Clusters (k-means)")
