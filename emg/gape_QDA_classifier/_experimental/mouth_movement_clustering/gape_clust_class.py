# Use the results in Li et al. 2016 to get gapes on taste trials

import os
import sys
from glob import glob

import numpy as np
import tables
import pylab as plt
import pandas as pd
from scipy.stats import zscore

# Have to be in blech_clust/emg/gape_QDA_classifier dir
os.chdir(os.path.expanduser('~/Desktop/blech_clust/emg/gape_QDA_classifier'))
sys.path.append('../..')
from utils.blech_utils import imp_metadata
from _experimental.gape_clust_funcs import (extract_movements,
                                            normalize_segments,
                                            extract_features,
                                            find_segment,
                                            calc_peak_interval,
                                            JL_process,
                                            gen_gape_frame,
                                            )

import itertools
from sklearn.cluster import KMeans, AgglomerativeClustering

# TODO: Add function to check for and chop up segments with double peaks

############################################################
# Load Data
############################################################


# Ask for the directory where the hdf5 file sits, and change to that directory
# Get name of directory with the data files
#data_dir = '/media/fastdata/KM45/KM45_5tastes_210620_113227_new'
data_dir = '/home/abuzarmahmood/Desktop/blech_clust/pipeline_testing/test_data_handling/test_data/KM45_5tastes_210620_113227_new'
metadata_handler = imp_metadata([[], data_dir])
data_dir = metadata_handler.dir_name
os.chdir(data_dir)

# Open the hdf5 file
hf5 = tables.open_file(metadata_handler.hdf5_name, 'r+')

# Extract taste dig-ins from experimental info file
info_dict = metadata_handler.info_dict
params_dict = metadata_handler.params_dict
pre_stim, post_stim = params_dict['spike_array_durations']
taste_names = info_dict['taste_params']['tastes']

############################################################
# Load and Process Data
############################################################
emg_output_dir = os.path.join(data_dir, 'emg_output')
# Get dirs for each emg CAR
dir_list = glob(os.path.join(emg_output_dir, 'emg*'))
dir_list = [x for x in dir_list if os.path.isdir(x)]

# Load the unique laser duration/lag combos and the trials that correspond
# to them from the ancillary analysis node
# Shape : (laser conditions x trials per laser condition)
trials = hf5.root.ancillary_analysis.trials[:]
laser_cond_num = len(trials)
unique_lasers = hf5.root.ancillary_analysis.laser_combination_d_l[:]

# Pull out a specific channel
num = 0
dir_name = dir_list[num]

emg_basename = os.path.basename(dir_name)
print(f'Processing {emg_basename}')

if 'emg_env.npy' not in os.listdir(dir_name):
    raise Exception(f'emg_env.py not found for {dir_name}')
    exit()

os.chdir(dir_name)

# Paths for plotting
plot_dir = f'emg_output/gape_classifier_plots/overview/{emg_basename}'
fin_plot_dir = os.path.join(data_dir, plot_dir)
if not os.path.exists(fin_plot_dir):
    os.makedirs(fin_plot_dir)

# Load the required emg data (the envelope and sig_trials)
env = np.load('emg_env.npy')
num_tastes, num_trials, time_len = env.shape
env = np.vstack(env)
sig_trials = np.load('sig_trials.npy').flatten()

# Now arrange these arrays by (laser condition X taste X trials X time)
# Shape : (laser conditions x tastes x trials x time)
env_final = np.reshape(
    env,
    (
        laser_cond_num,
        num_tastes,
        int(num_trials/laser_cond_num),
        time_len
    ),
)


# Shape : (laser conditions x tastes x trials)
sig_trials_final = np.reshape(
    sig_trials,
    (
        laser_cond_num,
        num_tastes,
        int(num_trials/laser_cond_num),
    ),
)

# Make an array to store gapes (with 1s)
gapes_Li = np.zeros(env_final.shape)
# Also make an array to store the time of first gape on every trial
first_gape = np.empty(sig_trials_final.shape, dtype=int)

segment_dat_list = []
inds = list(np.ndindex(sig_trials_final.shape[:3]))
for this_ind in inds:
    this_trial_dat = env_final[this_ind]

    ### Jenn Li Process ###
    # Get peak indices
    this_laser_prestim_dat = env_final[this_ind[0], :, :, :pre_stim]
    gape_peak_inds, first_gape[this_ind], sig_trials_final[this_ind] = \
            JL_process(
                    this_trial_dat, 
                    this_laser_prestim_dat,
                    sig_trials_final,
                    pre_stim,
                    post_stim,
                    this_ind,)
    gapes_Li[this_ind][gape_peak_inds] = 1

    ### AM Process ###
    segment_starts, segment_ends, segment_dat = extract_movements(
        this_trial_dat, size=200)

    (feature_array,
     feature_names,
     segment_dat,
     segment_starts,
     segment_ends) = extract_features(
        segment_dat, segment_starts, segment_ends)

    segment_bounds = list(zip(segment_starts, segment_ends))
    merged_dat = [feature_array, segment_dat, segment_bounds] 
    segment_dat_list.append(merged_dat)

#fig,ax = plt.subplots(len(taste_names), 1, sharex=True, sharey=True)
#for taste_num, taste_name in enumerate(taste_names):
#    ax[taste_num].imshow(gapes_Li[0, taste_num], aspect='auto',
#                         interpolation='gaussian')
#    ax[taste_num].set_title(taste_name)
#plt.show()

############################################################
## Cluster waveforms 
############################################################
# For each cluster, return:
# 1) Features
# 2) Mean waveform
# 3) Fraction of classifier gapes

gape_frame, scaled_features = gen_gape_frame(segment_dat_list, gapes_Li, inds)

############################################################
# Plot all segmented data for visual inspection
this_plot_dir = os.path.join(plot_dir, 'segmented_data')
if not os.path.exists(this_plot_dir):
    os.makedirs(this_plot_dir)

taste_groups = list(gape_frame.groupby(['taste']))
for taste_num, this_taste in taste_groups:
    trial_count = this_taste.trial.nunique()
    fig,ax = plt.subplots(trial_count,1, sharex=True, sharey=True,
                          figsize=(10,trial_count*2))
    for num, this_row in this_taste.iterrows():
        ax[this_row.trial].plot(np.arange(*this_row.segment_bounds), this_row.segment_raw,
                                linewidth = 3)
    for this_trial in range(env_final[:,taste_num].shape[1]):
        ax[this_trial].plot(env_final[:,taste_num][:,this_trial].flatten(),
                            color = 'k', linewidth = 0.5)
    fig.suptitle(str(this_taste.taste.unique()[0]))
    fig.savefig(os.path.join(this_plot_dir, 'taste_' + str(this_taste.taste.unique()[0]) + '.png'),
                dpi = 300, bbox_inches = 'tight')
    plt.close(fig)

############################################################

############################################################
# Compare clusters
############################################################
def median_zscore(x, axis=0):
    """
    Subtract median and divide by MAD
    """
    return (x - np.median(x, axis=axis)) / np.median(np.abs(x - np.median(x, axis=axis)), axis=axis)

n_components = 10
#gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=0)
#gmm.fit(X)
#labels = gmm.predict(X)
#kmeans = KMeans(n_clusters=n_components, random_state=0).fit(X)
#labels = kmeans.labels_
## Project features onto 3D
#pca = PCA(n_components=3)
#X_pca = pca.fit_transform(scaled_features)

# Use agglomerative clustering
clustering = AgglomerativeClustering(n_clusters=n_components).fit(scaled_features)
#clustering = AgglomerativeClustering(n_clusters=n_components).fit(X_pca)
labels = clustering.labels_

# Classifier gapes by labels
gape_bool = gape_frame['classifier'].values
class_gape_per_label = [np.mean(gape_bool[labels == x]) for x in range(n_components)]
class_gape_per_label = np.round(class_gape_per_label, 3)

# Plot
# Sorted features by label, concatenated with cluster_labels
sorted_labels = np.sort(labels)
sorted_features = np.stack(gape_frame['features'].values)[np.argsort(labels)]
#plot_dat = np.concatenate((sorted_features, sorted_labels[:,None]), axis=1)
plot_dat = sorted_features
plot_dat = median_zscore(plot_dat,axis=0)

# Plot n representative waveforms per cluster
# and overlay the mean waveform
plot_n = 50
clust_waveforms = []
mean_waveforms = []
for this_clust in range(n_components):
    clust_inds = np.where(labels == this_clust)[0]
    clust_dat = gape_frame['segment_raw'].values[clust_inds]
    # Get n random waveforms
    this_plot_n = np.min([plot_n, len(clust_dat)])
    rand_inds = np.random.choice(np.arange(len(clust_dat)), this_plot_n, replace=False)
    clust_waveforms.append(clust_dat[rand_inds])
    ## Get mean waveform
    #mean_waveforms.append(np.mean(clust_dat, axis=0))

# Plot
fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(1, 10)
ax = [fig.add_subplot(gs[0, :1]), 
      fig.add_subplot(gs[0, 2:7]),
      fig.add_subplot(gs[0, 8:])]
im = ax[1].imshow(plot_dat, 
          aspect='auto', cmap='viridis', interpolation='none',
          vmin=-5, vmax=5, origin='lower')
plt.colorbar(im, ax=ax[1])
ax[1].set_title('Features')
ax[1].set_xticks(np.arange(len(feature_names)))
ax[1].set_xticklabels(feature_names, rotation=90)
ax[2].barh(np.arange(n_components), class_gape_per_label)
ax[2].set_xlim(0,1)
ax[2].set_title('Mean Gape probability')
ax[2].set_xlabel('Probability')
ax[2].set_ylabel('Cluster')
# Add a subplot to plot the cluster labels and share the y-axis
# with the image
im = ax[0].imshow(sorted_labels[::-1, None], aspect='auto', cmap='tab20', interpolation='none',)
# Plot number of each cluster at center of cluster
for this_clust in range(n_components):
    ax[0].text(0, len(sorted_labels) - np.mean(np.where(sorted_labels == this_clust)), this_clust, 
               ha='center', va='center', color='k', fontsize=12)
ax[0].set_title('Cluster labels')
fig.suptitle('Gape cluster breakdown')
fig.savefig(os.path.join(this_plot_dir, 'gape_cluster_breakdown.png'), dpi=300, bbox_inches='tight')
plt.close(fig)

fig,ax = plt.subplots(n_components,1,sharex=True, sharey=True,
                      figsize=(3,n_components))
for this_clust in range(n_components):
    for this_wave in clust_waveforms[this_clust]:
        ax[this_clust].plot(this_wave, color='grey', alpha=0.3)
    ax[this_clust].set_title(f'Cluster {this_clust}, mean_prob = {class_gape_per_label[this_clust]}')
ax[-1].set_xlabel('Time (ms)')
plt.suptitle('Random waveforms from each cluster')
fig.savefig(os.path.join(this_plot_dir, 'gape_cluster_waveforms.png'), dpi=300, bbox_inches='tight')
plt.close(fig)

############################################################
############################################################
# If given a waveform or set of waveforms, find the closest cluster

# Cluster with highest probability of being a gape
gape_clust = np.argmax(class_gape_per_label)

# Extract waveforms from gape cluster
gape_waveforms = clust_waveforms[gape_clust]
