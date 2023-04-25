from sklearn.mixture import GaussianMixture as gmm
from sklearn.decomposition import PCA as pca
from sklearn.preprocessing import StandardScaler as scaler
from scipy.stats import zscore
import blech_waveforms_datashader
import memory_monitor as mm
import pylab as plt
import json
import sys
from clustering import *
import numpy as np
import tables
import os
import shutil
import matplotlib
matplotlib.use('Agg')

import umap
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import cut_tree

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

############################################################
# | |    ___   __ _  __| |
# | |   / _ \ / _` |/ _` |
# | |__| (_) | (_| | (_| |
# |_____\___/ \__,_|\__,_|
############################################################


# Read blech.dir, and cd to that directory
f = open('blech.dir', 'r')
dir_name = []
for line in f.readlines():
    dir_name.append(line)
f.close()
os.chdir(dir_name[0][:-1])

electrode_num = int(sys.argv[1])

# Check if the directories for this electrode number exist -
# if they do, delete them (existence of the directories indicates a
# job restart on the cluster, so restart afresh)


def ifisdir_rmdir(dir_name):
    if os.path.isdir(dir_name):
        shutil.rmtree(dir_name)


dir_list = [f'./Plots/{electrode_num:02}',
            f'./spike_waveforms/electrode{electrode_num:02}',
            f'./spike_times/electrode{electrode_num:02}',
            f'./clustering_results/electrode{electrode_num:02}']
for this_dir in dir_list:
    ifisdir_rmdir(this_dir)
    os.makedirs(this_dir)

# Get the names of all files in the current directory, and find the .params and hdf5 (.h5) file
file_list = os.listdir('./')
hdf5_name = ''
params_file = ''
for files in file_list:
    if files[-2:] == 'h5':
        hdf5_name = files
    if files[-6:] == 'params':
        params_file = files

with open(params_file, 'r') as params_file_connect:
    params_dict = json.load(params_file_connect)

# Ideally one would access the params_dict and not have to define variables
# But one step at a time
for key, value in params_dict.items():
    globals()[key] = value

# Open up hdf5 file, and load this electrode number
hf5 = tables.open_file(hdf5_name, 'r')
exec(f"raw_el = hf5.root.raw.electrode{electrode_num:02}[:]")
hf5.close()

# High bandpass filter the raw electrode recordings
filt_el = get_filtered_electrode(
    raw_el,
    freq=[bandpass_lower_cutoff, bandpass_upper_cutoff],
    sampling_rate=sampling_rate)

# Delete raw electrode recording from memory
del raw_el

# Calculate the 3 voltage parameters
breach_rate = float(len(np.where(filt_el > voltage_cutoff)[0])
                    * int(sampling_rate))/len(filt_el)
test_el = np.reshape(filt_el[:int(sampling_rate)
                             * int(len(filt_el)/sampling_rate)],
                     (-1, int(sampling_rate)))
breaches_per_sec = [len(np.where(test_el[i] > voltage_cutoff)[0])
                    for i in range(len(test_el))]
breaches_per_sec = np.array(breaches_per_sec)
secs_above_cutoff = len(np.where(breaches_per_sec > 0)[0])
if secs_above_cutoff == 0:
    mean_breach_rate_persec = 0
else:
    mean_breach_rate_persec = np.mean(breaches_per_sec[
        np.where(breaches_per_sec > 0)[0]])

# And if they all exceed the cutoffs,
# assume that the headstage fell off mid-experiment
recording_cutoff = int(len(filt_el)/sampling_rate)
if breach_rate >= max_breach_rate and \
        secs_above_cutoff >= max_secs_above_cutoff and \
        mean_breach_rate_persec >= max_mean_breach_rate_persec:
    # Find the first 1 second epoch where the number of cutoff breaches
    # is higher than the maximum allowed mean breach rate
    recording_cutoff = np.where(breaches_per_sec >
                                max_mean_breach_rate_persec)[0][0]

# Dump a plot showing where the recording was cut off at
fig = plt.figure()
plt.plot(np.arange(test_el.shape[0]), np.mean(test_el, axis=1))
plt.plot((recording_cutoff, recording_cutoff),
         (np.min(np.mean(test_el, axis=1)),
          np.max(np.mean(test_el, axis=1))), 'k-', linewidth=4.0)
plt.xlabel('Recording time (secs)')
plt.ylabel('Average voltage recorded per sec (microvolts)')
plt.title('Recording cutoff time (indicated by the black horizontal line)')
fig.savefig(f'./Plots/{electrode_num:02}/cutoff_time.png', bbox_inches='tight')
plt.close("all")

#############################################################
# | __ )  ___  __ _(_)_ __    _ __  _ __ ___   ___ ___  ___ ___
# |  _ \ / _ \/ _` | | '_ \  | '_ \| '__/ _ \ / __/ _ \/ __/ __|
# | |_) |  __/ (_| | | | | | | |_) | | | (_) | (_|  __/\__ \__ \
# |____/ \___|\__, |_|_| |_| | .__/|_|  \___/ \___\___||___/___/
#            |___/          |_|
#############################################################

# Then cut the recording accordingly
filt_el = filt_el[:recording_cutoff*int(sampling_rate)]

slices, spike_times, polarity, mean_val, threshold = \
    extract_waveforms_abu(filt_el,
                          spike_snapshot=[spike_snapshot_before,
                                          spike_snapshot_after],
                          sampling_rate=sampling_rate)

# Extract windows from filt_el and plot with threshold overlayed
window_len = 0.2  # sec
window_count = 10
windows_in_data = len(filt_el) // (window_len * sampling_rate)
window_markers = np.linspace(0,
                             int(windows_in_data*(window_len * sampling_rate)),
                             int(windows_in_data))
window_markers = np.array([int(x) for x in window_markers])
chosen_window_inds = np.vectorize(np.int)(np.sort(np.random.choice(
    np.arange(windows_in_data), window_count)))
chosen_window_markers = [(window_markers[x-1], window_markers[x])
                         for x in chosen_window_inds]
chosen_windows = [filt_el[start:end] for (start, end) in chosen_window_markers]
# For each window, extract detected spikes
chosen_window_spikes = [np.array(spike_times)
                        [(spike_times > start)*(spike_times < end)] - start
                        for (start, end) in chosen_window_markers]

fig, ax = plt.subplots(len(chosen_windows), 1,
                       sharex=True, sharey=True, figsize=(10, 10))
for dat, spikes, this_ax in zip(chosen_windows, chosen_window_spikes, ax):
    this_ax.plot(dat, linewidth=0.5)
    this_ax.hlines(mean_val + threshold, 0, len(dat))
    this_ax.hlines(mean_val - threshold, 0, len(dat))
    if len(spikes) > 0:
        this_ax.scatter(spikes, np.repeat(mean_val, len(spikes)), s=5, c='red')
    this_ax.set_ylim((mean_val - 1.5*threshold,
                      mean_val + 1.5*threshold))
fig.savefig(f'./Plots/{electrode_num:02}/bandapass_trace_snippets.png',
            bbox_inches='tight', dpi=300)
plt.close(fig)

# Delete filtered electrode from memory
del filt_el, test_el


def img_plot(array):
    plt.imshow(array, interpolation='nearest', aspect='auto', cmap='jet')


# Dejitter these spike waveforms, and get their maximum amplitudes
# Slices are returned sorted by amplitude polaity
slices_dejittered, times_dejittered = \
    dejitter_abu3(slices,
                  spike_times,
                  polarity=polarity,
                  spike_snapshot=[spike_snapshot_before, spike_snapshot_after],
                  sampling_rate=sampling_rate)

spike_order = np.argsort(times_dejittered)
times_dejittered = times_dejittered[spike_order]
slices_dejittered = slices_dejittered[spike_order]
polarity = polarity[spike_order]

amplitudes = np.zeros((slices_dejittered.shape[0]))
amplitudes[polarity < 0] = np.min(slices_dejittered[polarity < 0], axis=1)
amplitudes[polarity > 0] = np.max(slices_dejittered[polarity > 0], axis=1)

# Calculate autocorrelation of dejittered slices to attempt to remove
# periodic noise
slices_autocorr = fftconvolve(slices_dejittered, slices_dejittered, axes=-1)

# Delete the original slices and times now that dejittering is complete
del slices
del spike_times

# Scale the dejittered slices by the energy of the waveforms
scaled_slices, energy = scale_waveforms(slices_dejittered)
# Scale the autocorrelations by zscoring the autocorr for each waveform
scaled_autocorr = zscore(slices_autocorr, axis=-1)

# Run PCA on the scaled waveforms
pca_slices, explained_variance_ratio = implement_pca(scaled_slices)
# Perform PCA on scaled autocorrelations
pca_autocorr, autocorr_explained_variance_ratio = implement_pca(
    scaled_autocorr)

# Save the pca_slices, energy and amplitudes to the
# spike_waveforms folder for this electrode
# Save slices/spike waveforms and their times to their respective folders
to_be_saved = ['slices_dejittered', 'times_dejittered',
               'pca_slices', 'pca_autocorr', 'energy', 'amplitudes']
save_paths = \
    [f'./spike_waveforms/electrode{electrode_num:02}/spike_waveforms.npy',
     f'./spike_times/electrode{electrode_num:02}/spike_times.npy',
     f'./spike_waveforms/electrode{electrode_num:02}/pca_waveforms.npy',
     f'./spike_waveforms/electrode{electrode_num:02}/pca_waveform_autocorrelation.npy',
     f'./spike_waveforms/electrode{electrode_num:02}/energy.npy',
     f'./spike_waveforms/electrode{electrode_num:02}/spike_amplitudes.npy']

for key, path in zip(to_be_saved, save_paths):
    np.save(path, globals()[key])

# Create file for saving plots, and plot explained variance ratios of the PCA
fig = plt.figure()
x = np.arange(len(explained_variance_ratio))
plt.plot(x, explained_variance_ratio, 'x')
plt.title('Variance ratios explained by PCs')
plt.xlabel('PC #')
plt.ylabel('Explained variance ratio')
fig.savefig(f'./Plots/{electrode_num:02}/pca_variance.png',
            bbox_inches='tight')
plt.close("all")

# Make an array of the data to be used for clustering,
# and delete pca_slices, scaled_slices, energy and amplitudes
n_pc = 3
data = np.zeros((len(pca_slices), n_pc + 2))
data[:, 2:] = pca_slices[:, :n_pc]
data[:, 0] = energy[:]/np.max(energy)
data[:, 1] = np.abs(amplitudes)/np.max(np.abs(amplitudes))
data = np.concatenate((data, pca_autocorr[:, :3]), axis=-1)

# Standardize features in the data since they
# occupy very uneven scales
standard_data = scaler().fit_transform(data)

# We can whiten the data and potentially use
# diagonal covariances for the GMM to speed things up
# Not sure how much this step helps
data = pca(whiten='True').fit_transform(standard_data)

del pca_slices
del scaled_slices
del energy
del slices_autocorr, scaled_autocorr, pca_autocorr

# Set a threshold on how many datapoints are used to FIT the gmm
dat_thresh = 10e3
train_set = data[np.random.choice(np.arange(data.shape[0]),
                                  int(np.min((data.shape[0], dat_thresh))))]

# Perform umap
umap_obj = umap.UMAP(n_components=3, random_state=0).fit(train_set)
umap_waveforms = umap_obj.transform(standard_data)

# Bin umap data for approximate clustering
total_bins = 10000
bin_num = int(np.round(total_bins ** (1/umap_waveforms.shape[1])))
# print(bin_num)
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

ward = AgglomerativeClustering(
    distance_threshold=0,
    n_clusters=None,
    linkage="ward").fit(standard_binned_frame)
binned_agg_predictions = ward.labels_

linkage = calc_linkage(ward)

clust_range = np.arange(2, max_clusters+1)
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

# Run GMM, from 2 to max_clusters
for i in range(max_clusters-1):
    # If dataset is very large, take subsample for fitting
    # train_set = data[np.random.choice(np.arange(data.shape[0]),
    #                int(np.min((data.shape[0],dat_thresh))))]
    # model = gmm(
    #        n_components = i+2,
    #        max_iter = num_iter,
    #        n_init = num_restarts,
    #        tol = thresh).fit(train_set)
    #
    #predictions = model.predict(data)
    predictions = fin_label_cols.iloc[:, i].values

    # Sometimes large amplitude noise waveforms cluster with the
    # spike waveforms because the amplitude has been factored out of
    # the scaled slices.
    # Run through the clusters and find the waveforms that are more than
    # wf_amplitude_sd_cutoff larger than the cluster mean.
    # Set predictions = -1 at these points so that they aren't
    # picked up by blech_post_process
    for cluster in range(i+2):
        cluster_points = np.where(predictions[:] == cluster)[0]
        this_cluster = predictions[cluster_points]
        cluster_amplitudes = amplitudes[cluster_points]
        cluster_amplitude_mean = np.mean(cluster_amplitudes)
        cluster_amplitude_sd = np.std(cluster_amplitudes)
        reject_wf = np.where(cluster_amplitudes <= cluster_amplitude_mean
                             - wf_amplitude_sd_cutoff*cluster_amplitude_sd)[0]
        this_cluster[reject_wf] = -1
        predictions[cluster_points] = this_cluster

    # Make folder for results of i+2 clusters, and store results there
    os.mkdir(f'./clustering_results/electrode{electrode_num:02}/clusters{i+2}')
    np.save(
        f'./clustering_results/electrode{electrode_num:02}/'
        f'clusters{i+2}/predictions.npy',
        predictions)

    # Create file, and plot spike waveforms for the different clusters.
    # Plot 10 times downsampled dejittered/smoothed waveforms.
    # Additionally plot the ISI distribution of each cluster
    os.mkdir(f'./Plots/{electrode_num:02}/{i+2}_clusters_waveforms_ISIs')

    fig, ax = plt.subplots()
    scatter = ax.scatter(*umap_waveforms[:,:2].T, 
            c = predictions,
            cmap = 'brg',
            )
    legend1 = ax.legend(*scatter.legend_elements())
    ax.add_artist(legend1)
    fig.savefig(f'./Plots/{electrode_num:02}/'
                f'{i+2}_clusters_waveforms_ISIs/solution_umap.png')
    plt.close(fig)

    x = np.arange(len(slices_dejittered[0])) + 1
    for cluster in range(i+2):
        cluster_points = np.where(predictions[:] == cluster)[0]

        if len(cluster_points) > 0:
            # downsample = False, Prevents waveforms_datashader
            # from FURTHER downsampling the given waveforms for plotting
            # Because in the previous version they were upsampled for clustering
            fig, ax = \
                blech_waveforms_datashader.waveforms_datashader(
                    slices_dejittered[cluster_points, :],
                    x,
                    downsample=False,
                    threshold=threshold,
                    dir_name="datashader_temp_el" + str(electrode_num))

            ax.set_xlabel('Sample ({:d} samples per ms)'.
                          format(int(sampling_rate/1000)))
            ax.set_ylabel('Voltage (microvolts)')
            ax.set_title('Cluster%i' % cluster)
            fig.savefig(f'./Plots/{electrode_num:02}/'
                        f'{i+2}_clusters_waveforms_ISIs/Cluster{cluster}_waveforms')
            plt.close("all")


            fig = plt.figure()
            cluster_times = times_dejittered[cluster_points]
            ISIs = np.ediff1d(np.sort(cluster_times))
            ISIs = ISIs/30.0
            max_ISI_val = 20
            bin_count = 100
            neg_pos_ISI = np.concatenate((-1*ISIs, ISIs), axis=-1)
            hist_obj = plt.hist(
                neg_pos_ISI,
                bins=np.linspace(-max_ISI_val, max_ISI_val, bin_count))
            plt.xlim([-max_ISI_val, max_ISI_val])
            # Scale y-lims by all but the last value
            upper_lim = np.max(hist_obj[0][:-1])
            if upper_lim:
                plt.ylim([0, upper_lim])
            plt.title("2ms ISI violations = %.1f percent (%i/%i)"
                      % ((float(len(np.where(ISIs < 2.0)[0])) /
                          float(len(cluster_times)))*100.0,
                         len(np.where(ISIs < 2.0)[0]),
                         len(cluster_times)) + '\n' +
                      "1ms ISI violations = %.1f percent (%i/%i)"
                      % ((float(len(np.where(ISIs < 1.0)[0])) /
                          float(len(cluster_times)))*100.0,
                         len(np.where(ISIs < 1.0)[0]), len(cluster_times)))
            fig.savefig(f'./Plots/{electrode_num:02}/'
                        f'{i+2}_clusters_waveforms_ISIs/Cluster{cluster}_ISIs')
            plt.close("all")
        else:
            file_path = f'./Plots/{electrode_num:02}/'\
                f'{i+2}_clusters_waveforms_ISIs/no_spikes_Cluster{cluster}'
            with open(file_path, 'w') as file_connect:
                file_connect.write('')

# Make file for dumping info about memory usage
f = open(f'./memory_monitor_clustering/{electrode_num:02}.txt', 'w')
print(mm.memory_usage_resource(), file=f)
f.close()
print(f'Electrode {electrode_num} complete.')
