import os
import sys
from glob import glob

import numpy as np
import tables
import pylab as plt
import pandas as pd
import glob
from tqdm import tqdm
from matplotlib.patches import Patch

# Have to be in blech_clust/emg/gape_QDA_classifier dir
os.chdir(os.path.expanduser('~/Desktop/blech_clust/emg/gape_QDA_classifier/_experimental/mouth_movement_clustering'))
sys.path.append(os.path.expanduser('~/Desktop/blech_clust'))
sys.path.append(os.path.expanduser('~/Desktop/blech_clust/emg/gape_QDA_classifier'))
from utils.blech_utils import imp_metadata
from extract_scored_data import return_taste_orders, process_scored_data
from gape_clust_funcs import (extract_movements,
                                            normalize_segments,
                                            extract_features,
                                            find_segment,
                                            calc_peak_interval,
                                            JL_process,
                                            gen_gape_frame,
                                            threshold_movement_lengths,
                                            )

import itertools
from sklearn.cluster import KMeans, AgglomerativeClustering

############################################################
############################################################
data_dir = '/home/abuzarmahmood/Desktop/blech_clust/emg/gape_QDA_classifier/_experimental/mouth_movement_clustering/data/NB27'

# For each day of experiment, load env and table files
data_subdirs = sorted(glob.glob(os.path.join(data_dir,'*')))
# Make sure that path is a directory
data_subdirs = [subdir for subdir in data_subdirs if os.path.isdir(subdir)]
# Make sure that subdirs are in order
subdir_basenames = [os.path.basename(subdir).lower() for subdir in data_subdirs]

env_files = [glob.glob(os.path.join(subdir,'*env.npy'))[0] for subdir in data_subdirs]
# Load env and table files
# days x tastes x trials x time
envs = np.stack([np.load(env_file) for env_file in env_files])

############################################################
# Get scored data 
############################################################
# Extract dig-in from datasets
raw_data_dir = '/media/fastdata/NB_data/NB27'
# Find HDF5 files
h5_files = glob.glob(os.path.join(raw_data_dir,'**','*','*.h5'))
h5_files = sorted(h5_files)
h5_basenames = [os.path.basename(x) for x in h5_files]
# Make sure order of h5 files is same as order of envs
order_bool = [x in y for x,y in zip(subdir_basenames, h5_basenames)]
if not all(order_bool):
    raise Exception('Bubble bubble, toil and trouble')

# Run pipeline
all_taste_orders = return_taste_orders(h5_files)
fin_scored_table = process_scored_data(data_subdirs, all_taste_orders)

############################################################
# Extract mouth movements 
############################################################
pre_stim = 2000
post_stim = 5000
gapes_Li = np.zeros(envs.shape)

segment_dat_list = []
inds = list(np.ndindex(envs.shape[:3]))
for this_ind in inds:
    this_trial_dat = envs[this_ind]

    ### Jenn Li Process ###
    # Get peak indices
    this_day_prestim_dat = envs[this_ind[0], :, :, :pre_stim]
    gape_peak_inds = JL_process(
                        this_trial_dat, 
                        this_day_prestim_dat,
                        pre_stim,
                        post_stim,
                        this_ind,)
    if gape_peak_inds is not None:
        gapes_Li[this_ind][gape_peak_inds] = 1

    ### AM Process ###
    segment_starts, segment_ends, segment_dat = extract_movements(
        this_trial_dat, size=200)

    # Threshold movement lengths
    segment_starts, segment_ends, segment_dat = threshold_movement_lengths(
        segment_starts, segment_ends, segment_dat, 
        min_len = 50, max_len= 500)

    #plt.plot(this_trial_dat)
    #for i in range(len(segment_starts)):
    #    plt.plot(np.arange(segment_starts[i], segment_ends[i]),
    #             segment_dat[i], linewidth = 5, alpha = 0.5)
    #plt.show()

    (feature_array,
     feature_names,
     segment_dat,
     segment_starts,
     segment_ends) = extract_features(
        segment_dat, segment_starts, segment_ends)

    segment_bounds = list(zip(segment_starts, segment_ends))
    merged_dat = [feature_array, segment_dat, segment_bounds] 
    segment_dat_list.append(merged_dat)

gape_frame, scaled_features = gen_gape_frame(segment_dat_list, gapes_Li, inds)
# Bounds for gape_frame are in 0-7000 time
# Adjust to make -2000 -> 5000
# Adjust segment_bounds by removing pre_stim
all_segment_bounds = gape_frame.segment_bounds.values
adjusted_segment_bounds = [np.array(x)-pre_stim for x in all_segment_bounds]
gape_frame['segment_bounds'] = adjusted_segment_bounds

###############################
## Plot segments using gape_frame
#gape_frame_trials = list(gape_frame.groupby(['channel','taste','trial']))
#gape_trials_inds = [x[0] for x in gape_frame_trials]
#gape_trials_dat = [x[1] for x in gape_frame_trials]
#
#plot_n = 15
#fig,ax = plt.subplots(plot_n, 1, sharex=True, sharey=True,
#                      figsize = (10, plot_n*2))
#for i in range(plot_n):
#    this_ind = gape_trials_inds[i]
#    this_env = envs[this_ind]
#    this_gape_dat = gape_trials_dat[i]
#    ax[i].plot(this_env, c = 'k', zorder = 10)
#    for this_segment in this_gape_dat.segment_bounds:
#        ax[i].plot(np.arange(this_segment[0], this_segment[1]),
#                   this_env[this_segment[0]:this_segment[1]],
#                   linewidth = 5, alpha = 0.7)
#plt.show()

##############################

gape_frame.rename(columns={'channel': 'day_ind'}, inplace=True)

# Calculate segment centers
gape_frame['segment_center'] = [np.mean(x) for x in gape_frame.segment_bounds]

# Create segment bounds for fin_scored_table
fin_scored_table['segment_bounds'] = list(zip(fin_scored_table['rel_time_start'], fin_scored_table['rel_time_stop']))

# Make sure that each segment in gape_frame is in fin_score_table
score_match_cols = ['day_ind','taste','taste_trial']
gape_match_cols = ['day_ind','taste,','trial']

score_bounds_list = []
for ind, row in tqdm(gape_frame.iterrows()):
    day_ind = row.day_ind
    taste = row.taste
    taste_trial = row.trial
    segment_center = row.segment_center
    wanted_score_table = fin_scored_table.loc[
        (fin_scored_table.day_ind == day_ind) &
        (fin_scored_table.taste == taste) &
        (fin_scored_table.taste_trial == taste_trial)]
    if len(wanted_score_table):
        # Check if segment center is in any of the scored segments
        for _, score_row in wanted_score_table.iterrows():
            if (score_row.segment_bounds[0] <= segment_center) & (segment_center <= score_row.segment_bounds[1]):
                gape_frame.loc[ind, 'scored'] = True
                gape_frame.loc[ind, 'event_type'] = score_row.event  
                score_bounds_list.append(score_row.segment_bounds)
                break
            else:
                gape_frame.loc[ind, 'scored'] = False

scored_gape_frame = gape_frame.loc[gape_frame.scored == True]
scored_gape_frame['score_bounds'] = score_bounds_list

############################################################
# Test plots 
############################################################
# Plot gapes LI
##############################
mean_gapes_Li = np.mean(gapes_Li, axis=2)
# Smooth with gaussian filter
from scipy.ndimage import gaussian_filter1d
mean_gapes_Li = gaussian_filter1d(mean_gapes_Li, 75, axis=2)

fig, ax = plt.subplots(*mean_gapes_Li.shape[:2], sharex=True, sharey=True)
fig.suptitle('Smoothed Gapes Li (75ms SD Gaussian)')
for i in range(mean_gapes_Li.shape[0]):
    for j in range(mean_gapes_Li.shape[1]):
        ax[i,j].plot(mean_gapes_Li[i,j,:])
        ax[i,j].set_title('Day {}, Taste {}'.format(i,j))
plt.show()

# Plot scored data
##############################
#plot_group = list(fin_table.groupby(['day_ind','taste','taste_trial']))
#plot_inds = [x[0] for x in plot_group]
#plot_dat = [x[1] for x in plot_group]
#
#t = np.arange(-2000, 5000)
#
#event_types = fin_table.event.unique()
#cmap = plt.get_cmap('tab10')
#event_colors = {event_types[i]:cmap(i) for i in range(len(event_types))}
#
## Generate custom legend
#from matplotlib.patches import Patch
#
#legend_elements = [Patch(facecolor=event_colors[event], edgecolor='k',
#                         label=event) for event in event_types]
#
#plot_n = 15
#fig,ax = plt.subplots(plot_n, 1, sharex=True,
#                      figsize = (10, plot_n*2))
#for i in range(plot_n):
#    this_scores = plot_dat[i]
#    this_inds = plot_inds[i]
#    this_env = envs[this_inds]
#    ax[i].plot(t, this_env)
#    for _, this_event in this_scores.iterrows():
#        event_type = this_event.event
#        start_time = this_event.rel_time_start
#        stop_time = this_event.rel_time_stop
#        this_event_c = event_colors[event_type]
#        ax[i].axvspan(start_time, stop_time, 
#                      color=this_event_c, alpha=0.5, label=event_type)
#ax[0].legend(handles=legend_elements, loc='upper right',
#             bbox_to_anchor=(1.5, 1.1))
#ax[0].set_xlim([0, 5000])
#fig.subplots_adjust(right=0.75)
#plt.show()

# Plot scored, segmented data
#############################
plot_group = list(scored_gape_frame.groupby(['day_ind','taste','trial']))
plot_inds = [x[0] for x in plot_group]
plot_dat = [x[1] for x in plot_group]

t = np.arange(-2000, 5000)

event_types = scored_gape_frame.event_type.unique()
cmap = plt.get_cmap('tab10')
event_colors = {event_types[i]:cmap(i) for i in range(len(event_types))}

# Generate custom legend
legend_elements = [Patch(facecolor=event_colors[event], edgecolor='k',
                         label=event) for event in event_types]

plot_n = 15
fig,ax = plt.subplots(plot_n, 1, sharex=True, sharey=True,
                      figsize = (10, plot_n*2))
for i in range(plot_n):
    this_scores = plot_dat[i]
    this_inds = plot_inds[i]
    this_env = envs[this_inds]
    ax[i].plot(t, this_env, color = 'k')
    for _, this_event in this_scores.iterrows():
        event_type = this_event.event_type
        score_start = this_event.score_bounds[0]
        score_stop = this_event.score_bounds[1]
        segment_start = this_event.segment_bounds[0]
        segment_stop = this_event.segment_bounds[1]
        segment_inds = np.logical_and(t >= segment_start, t <= segment_stop) 
        segment_t = t[segment_inds]
        segment_env = this_env[segment_inds]
        ax[i].plot(segment_t, segment_env, color='k')
        ax[i].plot(segment_t, segment_env, linewidth = 5, alpha = 0.7)
        this_event_c = event_colors[event_type]
        ax[i].axvspan(score_start, score_stop, 
                      color=this_event_c, alpha=0.5, label=event_type)
ax[0].legend(handles=legend_elements, loc='upper right',
             bbox_to_anchor=(1.5, 1.1))
ax[0].set_xlim([0, 5000])
fig.subplots_adjust(right=0.75)
plt.show()
