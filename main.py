# -*- coding: utf-8 -*-
"""
Clustering of 06-11-22.mp4
@author: francois courbier
"""
import numpy as np
import background as bg
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
import sys

features_nb = 15
X = np.zeros((0, features_nb))
headers = []

# Get X, headers and videotime
if(len(sys.argv) == 2):
    # Case of command line argument with dataset
    X = np.loadtxt(sys.argv[1], delimiter=',', skiprows=1)
    videotime = int(X.shape[0] / 25)
    headers = ["Mean of gray", "Variance of gray",
               "Mean of gradient of gray",
               "Mean of blue", "Variance of blue",
               "Mean of Green", "Variance of Green",
               "Mean of Red", "Variance of R",
               "Mean of Hue", "Variance of Hue",
               "Mean of Saturation", "Variance of Saturation",
               "Mean of Value", "Variance of Value"]
else:
    # Case no command line argument
    X, headers, videotime = bg.extractDataFromVideo()

# Display features through time
x = np.arange(1, videotime + 2)
for i in range(0, features_nb):
    bg.displayFigureSeconds(X[1::25, i], x, headers[i])

# Compute and display Kmeans clustering
kmeans = KMeans(n_clusters=3, random_state=0).fit(X[::25, [2, 3, 5, 9]])
bg.displayFigureSeconds(kmeans.labels_, x,
                        "Clusters (k-means) for subdataset 1st frames / sec ")

# Clustering performance evalutation (Adjusted Rand index)

labels_true = np.concatenate((np.array([1 for i in range(12)]),
                              np.array([2 for i in range(12, 45)]),
                              np.array([1 for i in range(45, 62)]),
                              np.array([2 for i in range(62, 85)]),
                              np.array([0 for i in range(85, 2341)]),
                              np.array([1 for i in range(2341, 2351)])))
ARI = []
# Keep only relevant features
X_sub = X[:, [2, 3, 5, 9]]
# Reach 25 multiple length
X_sub_tailed = np.append(X_sub, [X_sub[-1] for i in range(3)], axis=0)

# Compute ARI scores for all subdatasets (modulo 25)
for i in range(25):
    kmeans.fit(X_sub_tailed[i::25, :])
    labels_pred = kmeans.labels_
    ARI.append(metrics.adjusted_rand_score(labels_true, labels_pred))

plt.title("Adjusted rand score")
plt.plot(range(25), ARI)
plt.xlabel("Subdataset (nth frame chosen per second)")
plt.ylabel("ARI value")
plt.show()

print("Average ARI : %.2f %%" % np.mean(ARI))
