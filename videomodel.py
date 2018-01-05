# -*- coding: utf-8 -*-
"""
videomodel class
@author: francois courbier
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn import metrics


class VideoModel:
    """Class of the analytics model of the video .

    Attributes:
        X (numpy array): Dataset containing features extraction
        headers (List): List of string headers
        videofile (String): filename of the video to analyse
        videotime (int): time of the video in second
        video (videocapture): the video object from CV2
        features_nb (int): number of features
        fps (int): frame per seconds of the video
        labels_true (numpy array): sequence of true labels
    """
    def __init__(self, videofile, features_nb):
        self.X = np.zeros((0, features_nb))
        self.headers = ["Mean of gray", "Variance of gray",
                        "Mean of gradient of gray",
                        "Mean of blue", "Variance of blue",
                        "Mean of Green", "Variance of Green",
                        "Mean of Red", "Variance of R",
                        "Mean of Hue", "Variance of Hue",
                        "Mean of Saturation", "Variance of Saturation",
                        "Mean of Value", "Variance of Value"]
        self.videofile = videofile
        self.video = cv2.VideoCapture(self.videofile)
        self.fps = int(self.video.get(cv2.CAP_PROP_FPS))
        self.features_nb = features_nb
        self.labels_true = np.concatenate((
                np.array([1 for i in range(12)]),
                np.array([2 for i in range(12, 45)]),
                np.array([1 for i in range(45, 62)]),
                np.array([2 for i in range(62, 85)]),
                np.array([0 for i in range(85, 2341)]),
                np.array([1 for i in range(2341, 2351)])))

    def loadXfromfile(self, filename):
        """Load X from a dataset file

        @param filename: name of the dataset file
        """
        self.X = np.loadtxt(filename, delimiter=',', skiprows=1)
        # Reach 25 multiple length
        self.X = np.append(self.X, [self.X[-1] for i in range(3)], axis=0)
        self.videotime = int(self.X.shape[0] / self.fps)

    def computeFeatures(self, BGR_image):
        """Return basic features computed from an BGR image

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
        return features

    def extractDataFromVideo(self):
        """Load dataset X, headers, and videotime from videofile

        """
        X_ext = np.zeros((0, self.features_nb))

        # Extract features from video frames
        nbrFrame = 0
        ret = True
        while ret:
            ret, frame = self.video.read()
            nbrFrame += 1
            if (ret):
                features, headers = self.computeFeatures(frame)
                X_ext = np.append(X_ext, [features], axis=0)
        self.video.release()
        self.videotime = int(nbrFrame / self.fps)
        self.X = X_ext

    def displayFigureSeconds(self, points, seq_seconds, title):
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

    def computeARI(self, X, kmeans, plot=False):
        """Compute ARI scores for all subdatasets (modulo fps) and
        return ARI mean and ARI mean after label propagation

        @param X: dataset
        @param kmeans: kmeans clustering algorithm
        @param labels_true : true labels
        """
        ARI = []
        ARI_propagation = []

        for i in range(self.fps):
            kmeans.fit(X[i::self.fps, :])
            labels_pred = kmeans.labels_

            labels_pred_prop = np.copy(kmeans.labels_)
            for k in range(labels_pred_prop.shape[0]):
                if k > 0 and k+1 < labels_pred_prop.shape[0] and \
                    labels_pred_prop[k] != labels_pred_prop[k+1] and \
                    labels_pred_prop[k] != labels_pred_prop[k-1] and \
                        labels_pred_prop[k-1] == labels_pred_prop[k+1]:
                            labels_pred_prop[k] = labels_pred_prop[k-1]

            ARI.append(metrics.adjusted_rand_score(self.labels_true,
                                                   labels_pred))
            ARI_propagation.append(
                    metrics.adjusted_rand_score(self.labels_true,
                                                labels_pred_prop))

        ARI_mean = np.mean(ARI)
        ARI_prop_mean = np.mean(ARI_propagation)
        if plot is True:
            plt.title("Adjusted rand score")
            plt.plot(range(self.fps), ARI, label="ARI")
            plt.plot(range(self.fps), ARI_propagation,
                     label="ARI after label propagation")
            plt.xlabel("Subdataset (nth frame chosen per second)")
            plt.ylabel("ARI value")
            plt.legend()
            plt.show()
            print("Average ARI: %.2f" % (ARI_mean))
            print("Average ARI after label propagation: %.2f"
                  % (ARI_prop_mean))

        return ARI_mean, ARI_prop_mean
