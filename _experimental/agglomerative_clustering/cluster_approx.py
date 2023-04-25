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
import pandas as pd
import math
from sklearn.cluster import AgglomerativeClustering
from time import time

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


data_dir = '/media/bigdata/projects/neuRecommend/test_data/raw'
data_files = sorted(glob(os.path.join(data_dir, '*.npy')))

ind = 13
#for ind in range(len(data_files)):
#slices_dejittered = np.load(data_files[ind])[:,::10]
slices_dejittered = np.load(data_files[ind])
print(slices_dejittered.shape)

plt.imshow(zscore(slices_dejittered,axis=-1), aspect='auto');plt.show()

plt.plot(slices_dejittered[::100].T);plt.show()
#plt.plot(zscore(slices_dejittered[::100].T, axis=-1));plt.show()

scaled_slices, energy = scale_waveforms(slices_dejittered)
pca_slices, explained_variance_ratio = implement_pca(scaled_slices)

n_pc = 3
data = np.zeros((len(pca_slices), n_pc + 1))
data[:,:n_pc] = pca_slices[:,:n_pc]
data[:,n_pc] = energy[:]/np.max(energy)

standard_data = scaler().fit_transform(data)
dat_thresh = 10e3
train_set = standard_data[np.random.choice(np.arange(standard_data.shape[0]),
                int(np.min((standard_data.shape[0],dat_thresh))))]

############################################################

num_iter = int(1e3)
num_restarts = 10
thresh = 1e-4

i = 4
# If dataset is very large, take subsample for fitting
model = gmm(
        n_components = i, 
        max_iter = num_iter, 
        n_init = num_restarts, 
        tol = thresh).fit(train_set)

predictions = model.predict(standard_data)

cluster_waveforms = [slices_dejittered[predictions==x] for x in np.unique(predictions)]

umap_waveforms = umap.UMAP(n_components = 2).\
        fit_transform(train_set)

fig = plt.figure()
ax0 = plt.subplot(1, 2, 1)
ax_list = [plt.subplot(i,2,x+2) for x in np.arange(2*i, step = 2)]
scatter = ax0.scatter(*umap_waveforms.T, 
        #color = 'k', 
        c =model.predict(train_set), 
        cmap = 'brg',
        )
        #alpha = 0.1)
legend1 = ax0.legend(*scatter.legend_elements())
ax0.add_artist(legend1)
for this_ax, this_clust in zip(ax_list, cluster_waveforms):
    this_ax.plot(this_clust[np.random.choice(len(this_clust), 30)].T,
            color = 'k', alpha = 0.5)
plt.show()

#plt.hist(standard_data[:,-1], bins = 30);plt.show()
#plt.imshow(standard_data, aspect='auto');plt.colorbar();plt.show()

############################################################
total_bins = 10000
bin_num = int(np.round(math.log(total_bins, standard_data.shape[1])))
print(bin_num)
standard_frame = pd.DataFrame(standard_data)
standard_frame.columns = [chr(x) for x in range(97, 97+len(standard_frame.columns))]
# Instead of even-width bins, use percentile bins
for x in standard_frame.columns:
    standard_frame[f'{x}_binned'] = pd.qcut(
                                    standard_frame[x], 
                                    q = bin_num,
                                    labels = np.arange(bin_num),
                                    retbins = True)[0]
    #standard_frame[f'{x}_binned'] = pd.cut(
    #                                standard_frame[x], 
    #                                bins = bin_num,
    #                                labels = np.arange(bin_num),
    #                                retbins = True)[0]
bin_cols = [x for x in standard_frame.columns if 'bin' in x]
#bin_frame = standard_frame[bin_cols].drop_duplicates()
standard_binned_frame = standard_frame.groupby(bin_cols).mean().dropna() 
standard_binned_frame = standard_binned_frame.reset_index(drop=False)
bin_frame = standard_binned_frame[bin_cols]
standard_binned_frame = standard_binned_frame.drop(columns = bin_cols)
standard_binned_frame.shape

binned_model = gmm(
        n_components = i, 
        max_iter = num_iter, 
        n_init = num_restarts, 
        tol = thresh).fit(standard_binned_frame.to_numpy())

binned_predictions = binned_model.predict(standard_data)
binned_cluster_waveforms = [slices_dejittered[binned_predictions==x] for x in np.unique(binned_predictions)]

binned_umap = umap.UMAP(n_components = 2).\
        fit(standard_binned_frame)
umap_binned_waveforms = binned_umap.transform(standard_data) 

fig = plt.figure()
ax0 = plt.subplot(1, 2, 1)
ax_list = [plt.subplot(i,2,x+2) for x in np.arange(2*i, step = 2)]
scatter = ax0.scatter(*umap_binned_waveforms.T, 
        #color = 'k', 
        c =binned_model.predict(standard_data), 
        cmap = 'brg',
        )
        #alpha = 0.1)
legend1 = ax0.legend(*scatter.legend_elements())
ax0.add_artist(legend1)
for this_ax, this_clust in zip(ax_list, binned_cluster_waveforms):
    this_ax.plot(this_clust[np.random.choice(len(this_clust), 30)].T,
            color = 'k', alpha = 0.5)
plt.show()

############################################################
# Hierarchical clustering on binned data


ward = AgglomerativeClustering(n_clusters=i, linkage="ward").fit(standard_binned_frame)
binned_agg_predictions = ward.labels_

agg_bin_frame = bin_frame.copy()
agg_bin_frame['labels'] = binned_agg_predictions
agg_standard_frame = standard_frame.copy()

fin_agg_frame = agg_standard_frame.merge(agg_bin_frame)

agg_predictions = fin_agg_frame.labels
agg_cluster_waveforms = [slices_dejittered[agg_predictions==x] for x in np.unique(agg_predictions)]

fig = plt.figure()
ax0 = plt.subplot(1, 2, 1)
ax_list = [plt.subplot(i,2,x+2) for x in np.arange(2*i, step = 2)]
scatter = ax0.scatter(*umap_binned_waveforms.T, 
        #color = 'k', 
        c = agg_predictions,
        cmap = 'brg',
        )
        #alpha = 0.1)
legend1 = ax0.legend(*scatter.legend_elements())
ax0.add_artist(legend1)
for this_ax, this_clust in zip(ax_list, agg_cluster_waveforms):
    this_ax.plot(this_clust[np.random.choice(len(this_clust), 30)].T,
            color = 'k', alpha = 0.5)
plt.show()

############################################################
# Hierarchical clustering on standard data (no binning)

ward = AgglomerativeClustering(n_clusters=i, linkage="ward").fit(standard_data)
agg_predictions = ward.labels_

#agg_bin_frame = bin_frame.copy()
#agg_bin_frame['labels'] = binned_agg_predictions
#agg_standard_frame = standard_frame.copy()
#
#fin_agg_frame = agg_standard_frame.merge(agg_bin_frame)
#
#agg_predictions = fin_agg_frame.labels
agg_cluster_waveforms = [slices_dejittered[agg_predictions==x] for x in np.unique(agg_predictions)]

fig = plt.figure()
ax0 = plt.subplot(1, 2, 1)
ax_list = [plt.subplot(i,2,x+2) for x in np.arange(2*i, step = 2)]
scatter = ax0.scatter(*umap_binned_waveforms.T, 
        #color = 'k', 
        c = agg_predictions,
        cmap = 'brg',
        )
        #alpha = 0.1)
legend1 = ax0.legend(*scatter.legend_elements())
ax0.add_artist(legend1)
for this_ax, this_clust in zip(ax_list, agg_cluster_waveforms):
    this_ax.plot(this_clust[np.random.choice(len(this_clust), 30)].T,
            color = 'k', alpha = 0.5)
plt.show()

############################################################
# Binned Hierarchical clustering on UMAP data

start_t = time()
# UMAP data, this should "pre-cluster" it
umap_obj = umap.UMAP(n_components = 2).fit(train_set)
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
    #umap_frame[f'{x}_binned'] = pd.cut(
    #                                umap_frame[x], 
    #                                bins = bin_num,
    #                                labels = np.arange(bin_num),
    #                                retbins = True)[0]
bin_cols = [x for x in umap_frame.columns if 'bin' in x]
non_bin_cols = [x for x in umap_frame.columns if x not in bin_cols]
#bin_frame = umap_frame[bin_cols].drop_duplicates()
standard_binned_frame = umap_frame.groupby(bin_cols).mean().dropna() 
standard_binned_frame = standard_binned_frame.reset_index(drop=False)
bin_frame = standard_binned_frame[bin_cols]
standard_binned_frame = standard_binned_frame.drop(columns = bin_cols)
standard_binned_frame.shape

fig,ax = plt.subplots(1,2)
ax[0].scatter(*umap_waveforms.T, alpha = 0.7)
ax[1].scatter(*standard_binned_frame.to_numpy().T, alpha = 0.7)
plt.show()

#ward = AgglomerativeClustering(n_clusters=i, linkage="ward").fit(standard_binned_frame)
ward = AgglomerativeClustering(
        distance_threshold =0, 
        n_clusters = None, 
        linkage="ward").fit(standard_binned_frame)
binned_agg_predictions = ward.labels_

bin_frame['labels'] = ward.labels_
fin_agg_frame = umap_frame.merge(bin_frame, how = 'left')

fig,ax = plt.subplots(1,3)
ax[0].scatter(*umap_waveforms.T, alpha = 0.7)
ax[1].scatter(*standard_binned_frame.to_numpy().T, alpha = 0.7, c = binned_agg_predictions)
ax[2].scatter(*fin_agg_frame[non_bin_cols].to_numpy().T, alpha = 0.7, c = fin_agg_frame['labels'])
plt.show()

fin_predictions = fin_agg_frame['labels']
cluster_waveforms = [slices_dejittered[fin_predictions==x] for x in np.unique(fin_predictions)]

end_t = time()
time_taken = end_t - start_t
print(time_taken)

fig = plt.figure()
ax0 = plt.subplot(1, 2, 1)
ax_list = [plt.subplot(i,2,x+2) for x in np.arange(2*i, step = 2)]
scatter = ax0.scatter(*umap_waveforms.T, 
        #color = 'k', 
        c = fin_predictions,
        cmap = 'brg',
        )
        #alpha = 0.1)
legend1 = ax0.legend(*scatter.legend_elements())
ax0.add_artist(legend1)
for this_ax, this_clust in zip(ax_list, cluster_waveforms):
    this_ax.plot(this_clust[np.random.choice(len(this_clust), 30)].T,
            color = 'k', alpha = 0.5)
plt.show()

from scipy.cluster.hierarchy import dendrogram
plot_dendrogram(ward, truncate_mode = 'level', p = 4)
plt.show()
