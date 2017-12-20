# -*- coding: utf-8 -*-
"""
Display features evolution though time of 06-11-22.mp4
@author: francois courbier
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

features_nb = 15
features_data = np.zeros((0, features_nb))
headers = []

# Get video and frame rate
video = cv2.VideoCapture('06-11-22.mp4')
fps = video.get(cv2.CAP_PROP_FPS)


def computeFeatures(BGR_image):
    """Return basic features computed from an BGR image.

    @param BGR_image: image in BGR format
    """
    attributes = np.zeros(0)

    # Compute features from gray
    gray_image = cv2.cvtColor(BGR_image, cv2.COLOR_BGR2GRAY)
    mean_gray = np.mean(gray_image)
    var_gray = np.var(gray_image)
    mean_grad_gray = np.mean(np.gradient(gray_image))

    # Compute features from BGR
    b, g, r = cv2.split(BGR_image)
    mean_B = np.mean(b)
    var_B = np.var(b)
    mean_G = np.mean(g)
    var_G = np.var(g)
    mean_R = np.mean(r)
    var_R = np.var(r)

    # Compute features from HSV
    hsv = cv2.cvtColor(BGR_image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mean_H = np.mean(h)
    var_H = np.var(h)
    mean_S = np.mean(s)
    var_S = np.var(s)
    mean_V = np.mean(v)
    var_V = np.var(v)

    # Gather return values
    features = np.append(attributes, [mean_gray, var_gray,
                                      mean_grad_gray,
                                      mean_B, var_B,
                                      mean_G, var_G,
                                      mean_R, var_R,
                                      mean_H, var_H,
                                      mean_S, var_S,
                                      mean_V, var_V])
    headers = ("Mean of gray", "Variance of gray",
               "Mean of gradient of gray",
               "Mean of blue", "Variance of blue",
               "Mean of Green", "Variance of Green",
               "Mean of Red", "Variance of R",
               "Mean of Hue", "Variance of Hue",
               "Mean of Saturation", "Variance of Saturation",
               "Mean of Value", "Variance of Value")
    return features, headers


def displayFigureSeconds(points, seq_seconds, title):
    """Display points through time

    @param points: array of numerical values
    @param seq_seconds: sequence of seconds
    @title: title of the figure
    """
    fig = plt.figure(figsize=(8, 4), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot(seq_seconds, points, 'o')
    L1 = ax.axvline(x=11, ls="dashed", c="r")
    L2 = ax.axvline(x=85, ls="dashed", c="g")
    L3 = ax.axvline(x=39*60, ls="dashed", c="y")
    ax.legend([L1, L2, L3], ["End of opening titles", "End of TV coverage",
              "End of debate"])
    ax.set_title(title)
    ax.set_xlabel("time (seconds)")
    ax.set_ylabel("cluster value")
    plt.show()


# Extrat features from video frames
nbrFrame = 0
ret = True
while ret:
    ret, frame = video.read()
    nbrFrame += 1
    videotime = nbrFrame / fps
    if (nbrFrame % 25 == 1):
        features, headers = computeFeatures(frame)
        features_data = np.append(features_data, [features], axis=0)

# Display features through time
x = np.arange(1, videotime + 1)
for i in range(0, features_nb):
    displayFigureSeconds(features_data[:, i], x, headers[i])

# Compute Kmeans clustering
kmeans = KMeans(n_clusters=3, random_state=0).fit(features_data)

# Display clustering through time
displayFigureSeconds(kmeans.labels_, x, "clusters (k-means)")

# Export Dataset
#np.savetxt("data.out", features_data, delimiter=",")