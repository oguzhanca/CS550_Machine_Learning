import numpy as np
import matplotlib.pyplot as plt
import pylab
import imageio

def agglomerative(initial_data, k):

    iterations = 0
    while len(initial_data) != k:

        distance_matrix = np.zeros((len(initial_data), len(initial_data)), dtype=np.double)

        for i in range(initial_data.shape[0]):
            distance_matrix[i] = np.linalg.norm(initial_data - initial_data[i], axis=1)

        # Add 1000 to diagonal to get rid of 0s as min value.
        np.fill_diagonal(distance_matrix, 10000)
        # Check diagonal.
        #print(distance_matrix.diagonal())

        # Now get min indices.
        min_index_1, min_index_2 = np.argwhere(distance_matrix == np.min(distance_matrix))[0]
        #print(min_index_1, min_index_2)

        # Merge two cluster vectors. Basically take mean of two vectors.
        mn = np.mean((initial_data[min_index_1], initial_data[min_index_2]), axis=0)
        # Delete merged vectors.
        initial_data = np.delete(initial_data, [min_index_1, min_index_2], 0)
        # Add new mean vector.
        initial_data = np.append(initial_data, np.array([mn]), axis=0)

        iterations += 1

    new_clust_vecs = initial_data
    return new_clust_vecs



#%% Main
im = imageio.imread('./sample.jpg')
# Convert to np array.
im = np.array(im)
im2 = im.reshape(-1, 3)


initial_data = np.loadtxt('kmeans_256centroids.txt').astype(np.double)
k = 16
new_clusters = agglomerative(initial_data, k)

dist = np.zeros((im2.shape[0], k))
for i in range(k):
    dist[:, i] = np.linalg.norm(im2 - new_clusters[i], axis=1)
labels = np.argmin(dist, axis=1)

clustered = np.zeros((k, im.shape[0], im.shape[1], 3), dtype=np.uint8)

print('Centroids:\n', new_clusters.astype(int))

for k in range(1):

    labels = labels.reshape((im.shape[0], im.shape[1])).T

    clustered_im = np.zeros((im.shape[0], im.shape[1], 3), dtype=np.uint8)

    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            clustered_im[i, j, :] = new_clusters[labels[j, i], :].astype(np.uint8)

    clustered[k, :, :, :] = clustered_im

for i in range(1):

    plt.figure()
    pylab.imshow(clustered[i, :, :, :])


