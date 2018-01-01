# -*- coding: utf-8 -*-
"""
Background functions
@author: francois courbier
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt


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


def extractDataFromVideo(videofile='06-11-22.mp4', features_nb=15):
    """Return dataset X, headers, and videotime

    @videofile: filename of the video to process
    @features_nb: number of features
    """
    X = np.zeros((0, features_nb))

    # Get video and frame rate
    video = cv2.VideoCapture('06-11-22.mp4')
    fps = video.get(cv2.CAP_PROP_FPS)

    # Extract features from video frames
    nbrFrame = 0
    ret = True
    while ret:
        ret, frame = video.read()
        nbrFrame += 1
        if (ret):
            features, headers = computeFeatures(frame)
            X = np.append(X, [features], axis=0)
    video.release()
    videotime = int(nbrFrame / fps)
    return X, headers, videotime


def displayFigureSeconds(points, seq_seconds, title):
    """Display points through time

    @param points: array of numerical values
    @param seq_seconds: sequence of seconds
    @title: title of the figure
    """
    plt.figure(figsize=(11, 4), dpi=100)
    plt.plot(seq_seconds, points, 'o')

    plt.axvline(x=1, ls="dashed", c="red")
    plt.axvline(x=12, ls="dashed", c="green")
    plt.axvline(x=45, ls="dashed", c="yellow")
    plt.axvline(x=62, ls="dashed", c="blue")
    plt.axvline(x=85, ls="dashed", c="orange")
    plt.axvline(x=39*60+1, ls="dashed", c="violet")

    plt.xscale('log')

    plt.xticks([1, 3, 12, 25, 45, 49, 62, 65, 85, 500, 2341],
               ["1''", "Beginning credits", "12''",
               "TV coverage", "45''",
                "Artificial video", "1'02''",
                "Cont. of TV coverage", "1'25''",
                "Main debate", "End credits at 39'01''"], rotation=80)

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.show()
