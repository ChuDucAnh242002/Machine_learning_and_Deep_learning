import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets import make_blobs, make_moons

centers = [[0, 0], [1, 0], [1, 1]]
n_clusters = len(centers)
X, labels_true = make_blobs(n_samples=3000, centers=centers, cluster_std=[0.1, 0.1, 0.1])

print(X.shape)