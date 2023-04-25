"""
Testing approximations to speed up clustering
"""
import matplotlib
import os
import tables
import numpy as np
import sys
import json
import pylab as plt
import matplotlib.cm as cm
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler as scaler
from sklearn.decomposition import PCA as pca
from sklearn.mixture import GaussianMixture as gmm
from glob import glob
from clustering import *
import sys
import umap
import warnings
from numba
import pandas as pd
import math
from sklearn.cluster import AgglomerativeClustering
from time import time
from scipy.cluster.hierarchy import cut_tree
from tqdm import tqdm
from matplotlib.patches import ConnectionPatch

os.environ('NUMBA_NUM_THREADS') = 1
warnings.filterwarnings("ignore", category=numba.errors.NumbaPerformanceWarning)

def register_labels(x,y):
    unique_x = np.unique(x)
    x_cluster_y = [np.unique(y[x==i]) for i in unique_x]
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


############################################################
# Load data
data_dir = '/media/bigdata/projects/neuRecommend/test_data/raw'
data_files = sorted(glob(os.path.join(data_dir, '*.npy')))

ind = 15
slices_dejittered = np.load(data_files[ind])
#scaled_slices, energy = scale_waveforms(slices_dejittered)
scaled_slices = slices_dejittered
pca_slices, explained_variance_ratio = implement_pca(scaled_slices)

n_pc = 3
#data = np.zeros((len(pca_slices), n_pc + 1))
data = np.zeros((len(pca_slices), n_pc))
data[:,:n_pc] = pca_slices[:,:n_pc]
#data[:,n_pc] = energy[:]/np.max(energy)

standard_data = scaler().fit_transform(data)
dat_thresh = 10e3
train_set = standard_data[np.random.choice(np.arange(standard_data.shape[0]),
                int(np.min((standard_data.shape[0],dat_thresh))))]

############################################################
# Binned Hierarchical clustering on UMAP data

start_t = time()
# UMAP data, this should "pre-cluster" it
umap_obj = umap.UMAP(n_components = 2, random_state = 0).fit(train_set)
umap_waveforms = umap_obj.transform(standard_data)

# Bin the UMAP data
total_bins = 10000
bin_num = int(np.round(total_bins ** (1/umap_waveforms.shape[1])))
print(bin_num)
umap_frame = pd.DataFrame(umap_waveforms)
umap_frame.columns = [chr(x) for x in range(97, 97+len(umap_frame.columns))]
# Instead of even-width bins, use percentile bins
for x in umap_frame.columns:
    umap_frame[f'{x}_binned'] = pd.qcut(
                                    umap_frame[x], 
                                    q = bin_num,
                                    labels = np.arange(bin_num),
                                    retbins = True)[0]
bin_cols = [x for x in umap_frame.columns if 'bin' in x]
non_bin_cols = [x for x in umap_frame.columns if x not in bin_cols]
standard_binned_frame = umap_frame.groupby(bin_cols).mean().dropna() 
standard_binned_frame = standard_binned_frame.reset_index(drop=False)
bin_frame = standard_binned_frame[bin_cols]
standard_binned_frame = standard_binned_frame.drop(columns = bin_cols)
standard_binned_frame.shape

ward = AgglomerativeClustering(
        distance_threshold =0, 
        n_clusters = None, 
        linkage="ward").fit(standard_binned_frame)
binned_agg_predictions = ward.labels_

linkage = calc_linkage(ward)


n_clusters = 7
cut_pred = cut_tree(linkage, n_clusters = n_clusters).flatten()

clust_range = np.arange(2,8)
clust_label_list = [
        cut_tree(
            linkage, 
            n_clusters = this_num
            ).flatten()
        for this_num in tqdm(clust_range)
        ]


cut_label_array = np.stack(clust_label_list)

label_cols = [f'label_{i}' for i in clust_range]
bin_frame[label_cols] = pd.DataFrame(cut_label_array.T)#ward.labels_
fin_agg_frame = umap_frame.merge(bin_frame, how = 'left')

fin_label_cols = fin_agg_frame[label_cols]

map_dict = {}
for i in range(len(clust_range)-1):
    map_dict[i] = register_labels(
            fin_label_cols.iloc[:,i],
            fin_label_cols.iloc[:,i+1]
            )

end_t = time()
time_taken = end_t - start_t
print(time_taken)

cmap = plt.get_cmap('tab10')
slice_mid = slices_dejittered.shape[1]//2
fig,ax = plt.subplots(len(clust_range), np.max(clust_range))
for row in range(len(clust_range)):
    #row = 5
    center = int(np.max(clust_range)//2)
    labels = fin_label_cols.T.iloc[row]
    unique_labels = np.unique(labels)
    med_label = int(np.median(unique_labels))
    ax_inds = unique_labels - med_label + center
    parent_unique_labels = np.unique(fin_label_cols.T.iloc[row-1])
    parent_med_label = int(np.median(parent_unique_labels))
    parent_ax_inds = parent_unique_labels - parent_med_label + center
    ax[row,0].set_ylabel(f'{clust_range[row]} Clusters')
    for x in unique_labels:
        this_dat = slices_dejittered[labels==x] 
        ax[row, ax_inds[x]].plot(this_dat[np.random.choice(len(this_dat), 200)].T,
            color = cmap(0), alpha = 0.15)
    if row > 0: 
        this_map = map_dict[row-1]
        for key,val in this_map.items():
            for child in val:
                con = ConnectionPatch(
                        xyA = (slice_mid,0), coordsA = ax[row-1, parent_ax_inds[key]].transData,
                        xyB = (slice_mid,0), coordsB = ax[row, ax_inds[child]].transData,
                        arrowstyle = "-|>"
                        )
                fig.add_artist(con)
plt.show()

#wanted_dat = slices_dejittered[fin_label_cols.T.iloc[2] == 2]
#plt.plot(wanted_dat[:200].T,
#        alpha = 0.3, color = cmap(0))
#plt.show()


#fin_predictions = fin_label_cols.iloc[:,-1]#fin_agg_frame['labels']
#cluster_waveforms = [slices_dejittered[fin_predictions==x] for x in np.unique(fin_predictions)]
#
#fig = plt.figure()
#ax0 = plt.subplot(1, 2, 1)
#ax_list = [plt.subplot(n_clusters,2,x+2) for x in np.arange(2*n_clusters, step = 2)]
#scatter = ax0.scatter(*umap_waveforms.T, 
#        c = fin_predictions,
#        cmap = 'brg',
#        )
#legend1 = ax0.legend(*scatter.legend_elements())
#ax0.add_artist(legend1)
#for this_ax, this_clust in zip(ax_list, cluster_waveforms):
#    this_ax.plot(this_clust[np.random.choice(len(this_clust), 30)].T,
#            color = 'k', alpha = 0.5)
#plt.show()
