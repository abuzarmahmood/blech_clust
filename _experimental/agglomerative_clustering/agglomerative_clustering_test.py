# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 19:46:34 2022

@author: abuza
"""

import numpy as np

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import cut_tree


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    
def calc_linkage(model):
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)
    return linkage_matrix
    
iris = load_iris()
X = iris.data
X = PCA(n_components = 2).fit_transform(X)

# setting distance_threshold=0 ensures we compute the full tree.
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

model = model.fit(X)
plt.title("Hierarchical Clustering Dendrogram")
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode="level", p=3)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()

linkage = calc_linkage(model)
plt.scatter(*X.T, c = cut_tree(linkage, n_clusters = 4).flatten())

# =============================================================================
# 
# =============================================================================
# =============================================================================
# X = np.stack([[9, 0],
#     [1, 4],
#     [2, 3],
#     [8, 5],
#     [1, 4],
#     [2, 5],
#     [3, 6]])
# =============================================================================
from scipy.cluster.hierarchy import linkage

X = np.random.random((2000,2))
Z = linkage(X, method="ward")
print(Z)

dendrogram(Z,truncate_mode="level", p=3)

import scipy
scipy.cluster.hierarchy.cut_tree(Z, n_clusters = 7)
