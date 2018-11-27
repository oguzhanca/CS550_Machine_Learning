#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: oguzhan
"""

import scipy.io
import imageio
import numpy as np
import pylab
import matplotlib.pyplot as plt
from PIL import Image
import time


class Kmeans:
    def __init__(self, k):
        self.K = k
        self.num_of_K = len(k)
        self.all_error = []


    def init_rand_centroids(self, im, num_clusters):
        initial_points = np.arange(im.shape[0])
        np.random.shuffle(initial_points)
        centroid_locs = im[initial_points[:num_clusters], :]
        return centroid_locs


    def calc_Centr_Error(self, mse, k, labels):
        current_centroid = np.zeros((k, 3))
        error = 0
        for i in range(k):
            indexes = np.where(labels == i)[0]
            if len(indexes) != 0:
                values = im2[indexes, :]
                current_centroid[i, :] = np.mean(values, axis=0)
                error += np.sum((values[:, 0] - current_centroid[i, 0]).astype(np.float64) ** 2 +
                                (values[:, 1] - current_centroid[i, 1]).astype(np.float64) ** 2 +
                                (values[:, 2] - current_centroid[i, 2]).astype(np.float64) ** 2)
        mse += error / len(labels)
        return current_centroid, mse

    def predict(self, k, new_im):
        centroids = self.init_rand_centroids(new_im, k)

        iteration = 0
        mse = 0

        # Compute before while loop as initial run.
        dist = np.zeros((im2.shape[0], k))
        for i in range(k):
            dist[:, i] = np.linalg.norm(im2 - centroids[i], axis=1)
        labels = np.argmin(dist, axis=1)
        centroids, mse = self.calc_Centr_Error(mse, k, labels)

        while not (labels == np.argmin(dist, axis=1)).all():

            if iteration != 0:
                dist = np.zeros((im2.shape[0], k))
                dist[:, i] = np.linalg.norm(im2 - centroids[i], axis=1)
                labels = np.argmin(dist, axis=1)

            centroids, mse = self.calc_Centr_Error(mse, k, labels)
            iteration += 1

        return labels, centroids, mse


    def fit(self, im, im2):
        clusteredImages = np.zeros((self.num_of_K, im.shape[0], im.shape[1], 3), dtype=np.uint8)
        err = []


        for k in range(self.num_of_K):
            print("Computing for K =", self.K[k])

            labels, centroids, mse = self.predict(self.K[k], im2)

            labels = labels.reshape((im.shape[0], im.shape[1])).T
            clusteredImage = np.zeros((im.shape[0], im.shape[1], 3), dtype=np.uint8)

            for i in range(im.shape[0]):
                for j in range(im.shape[1]):
                    clusteredImage[i, j, :] = centroids[labels[j, i], :].astype(np.uint8)

            clusteredImages[k, :, :, :] = clusteredImage
            err.append(mse)

            print("Mean Squared Error = ", np.round(mse, decimals=2))
            print("Centroids =\n", np.around(centroids).astype(int))
            print("------------------------")

            imageio.imwrite("./q1_k" + str(self.K[k]) + ".png", clusteredImage)

        for i in range(self.num_of_K):
            pylab.figure()
            pylab.imshow(clusteredImages[i, :, :, :])
            pylab.title("K: " + str(self.K[i]))

            self.all_error.append(np.round(err[i], decimals=2))

        plt.figure()
        plt.plot(self.K, self.all_error)
        plt.title("Error Graph")
        plt.xlabel("Number of K")
        plt.ylabel("Mean squared error")
        plt.savefig("./Error_graph.png")
        plt.show()

#%% Main

# Read the image.
im = imageio.imread('./sample.jpg')
# Convert to np array.
im = np.array(im)
im2 = im.reshape(-1, 3)
print('\nimage read.\n')

K = (2, 4, 8)#, 16)#, 32, 64, 128)

start_time = time.time()

k_means = Kmeans(K)
k_means.fit(im, im2)
print("Error List: ", k_means.all_error)
print("\nK-values computed: ", K)
print("Time passed: ", np.round((time.time() - start_time), decimals=2), "seconds.")
print("Program finished.")
