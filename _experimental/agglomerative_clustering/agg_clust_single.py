"""
Clustering for a single channel
"""
import os
import numpy as np
from sklearn.preprocessing import StandardScaler as scaler
from glob import glob
import umap
import warnings
from numba import errors as nb_errors
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from time import time
from scipy.cluster.hierarchy import cut_tree
#from tqdm import tqdm
import sys
import argparse

sys.path.append('/home/abuzarmahmood/Desktop/blech_clust')
from clustering import *

#os.environ['NUMBA_NUM_THREADS'] = "1"
warnings.filterwarnings("ignore", category=nb_errors.NumbaPerformanceWarning)


def register_labels(x, y):
    unique_x = np.unique(x)
    x_cluster_y = [np.unique(y[x == i]) for i in unique_x]
    return dict(zip(unique_x, x_cluster_y))


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


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    linkage_martix = calc_linkage(model)
    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


# Get name of directory with the data files
# Create argument parser
parser = argparse.ArgumentParser(description = 'Creates files with experiment info')
parser.add_argument('ind_num',  help = 'Integer index of file to load')
args = parser.parse_args()
ind = int(args.ind_num)
print(f'File index : {ind}')
############################################################
# Load data
data_dir = '/media/bigdata/projects/neuRecommend/test_data/raw'
data_files = sorted(glob(os.path.join(data_dir, '*.npy')))

ind = 0
slices_dejittered = np.load(data_files[ind])
scaled_slices, energy = scale_waveforms(slices_dejittered)
pca_slices, explained_variance_ratio = implement_pca(scaled_slices)

n_pc = 3
data = np.zeros((len(pca_slices), n_pc + 1))
data[:, :n_pc] = pca_slices[:, :n_pc]
data[:, n_pc] = energy[:]/np.max(energy)

standard_data = scaler().fit_transform(data)
dat_thresh = 10e3
train_set = standard_data[np.random.choice(np.arange(standard_data.shape[0]),
                           int(np.min((standard_data.shape[0], dat_thresh))))]

############################################################
# Binned Hierarchical clustering on UMAP data

start_t = time()
# UMAP data, this should "pre-cluster" it
umap_obj = umap.UMAP(n_components=2, random_state=0).fit(train_set)
umap_waveforms = umap_obj.transform(standard_data)

# Bin the UMAP data
total_bins = 10000
bin_num = int(np.round(total_bins ** (1/umap_waveforms.shape[1])))
#print(bin_num)
umap_frame = pd.DataFrame(umap_waveforms)
umap_frame.columns = [chr(x) for x in range(97, 97+len(umap_frame.columns))]
# Instead of even-width bins, use percentile bins
for x in umap_frame.columns:
    umap_frame[f'{x}_binned'] = pd.qcut(
        umap_frame[x],
        q=bin_num,
        labels=np.arange(bin_num),
        retbins=True)[0]
bin_cols = [x for x in umap_frame.columns if 'bin' in x]
non_bin_cols = [x for x in umap_frame.columns if x not in bin_cols]
standard_binned_frame = umap_frame.groupby(bin_cols).mean().dropna()
standard_binned_frame = standard_binned_frame.reset_index(drop=False)
bin_frame = standard_binned_frame[bin_cols]
standard_binned_frame = standard_binned_frame.drop(columns=bin_cols)
standard_binned_frame.shape

ward = AgglomerativeClustering(
    distance_threshold=0,
    n_clusters=None,
    linkage="ward").fit(standard_binned_frame)
binned_agg_predictions = ward.labels_

linkage = calc_linkage(ward)

clust_range = np.arange(2, 8)
clust_label_list = [
    cut_tree(
        linkage,
        n_clusters=this_num
    ).flatten()
    for this_num in clust_range
]


cut_label_array = np.stack(clust_label_list)

label_cols = [f'label_{i}' for i in clust_range]
bin_frame[label_cols] = pd.DataFrame(cut_label_array.T)  # ward.labels_
fin_agg_frame = umap_frame.merge(bin_frame, how='left')

fin_label_cols = fin_agg_frame[label_cols]

end_t = time()
time_taken = end_t - start_t
print(f'Time taken : {time_taken : .2f} sec')
