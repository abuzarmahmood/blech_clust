import os
import sys
from glob import glob

import numpy as np
import tables
import pylab as plt

sys.path.append('../..')
from utils.blech_utils import imp_metadata


# Ask for the directory where the hdf5 file sits, and change to that directory
# Get name of directory with the data files
metadata_handler = imp_metadata(sys.argv)
data_dir = metadata_handler.dir_name
os.chdir(data_dir)


# Open the hdf5 file
hf5 = tables.open_file(metadata_handler.hdf5_name, 'r+')

# Extract taste dig-ins from experimental info file
info_dict = metadata_handler.info_dict
params_dict = metadata_handler.params_dict
pre_stim, post_stim = params_dict['spike_array_durations']
taste_names = info_dict['taste_params']['tastes']

stim_t = params_dict['spike_array_durations'][0]
psth_durs = params_dict['psth_params']['durations']
psth_durs[0] *= -1
psth_inds = [int(x + stim_t) for x in psth_durs]

unique_lasers = hf5.root.ancillary_analysis.laser_combination_d_l[:]

# Get gape data
emg_channels = hf5.get_node('/','emg_gape_classifier')
emg_channel_names = [x._v_pathname for x in emg_channels]
emg_channel_basenames = [os.path.basename(x) for x in emg_channel_names]
gape_list = [x.gapes_Li[:] for x in emg_channels]

mean_gapes = np.stack([x.mean(axis=2) for x in gape_list])
mean_gapes = mean_gapes[...,psth_inds[0]:psth_inds[1]]
kern_len = 300
kern = np.ones(kern_len)/kern_len
smooth_mean_gapes = np.empty(mean_gapes.shape)
inds = list(np.ndindex(smooth_mean_gapes.shape[:-1]))
for this_ind in inds:
    smooth_mean_gapes[this_ind] = np.convolve(
            mean_gapes[this_ind],
            kern,
            mode = 'same')

# Convert from ms-1 to s-1
smooth_mean_gapes *= 1000

plot_dir = 'emg_output/gape_classifier_plots'
fin_plot_dir = os.path.join(data_dir, plot_dir)
if not os.path.exists(fin_plot_dir):
    os.makedirs(fin_plot_dir)

fig,ax = plt.subplots(*mean_gapes.shape[:2], 
                      sharex=True, sharey=True)
t = np.arange(*psth_durs)
ax = np.atleast_2d(ax)
if ax.shape != mean_gapes.shape[:2]:
    ax = ax.T
assert ax.shape == mean_gapes.shape[:2], f"Axes shape : {ax.shape}" +\
        f" not equal to data shape {mean_gapes.shape[:2]}"
inds = list(np.ndindex(ax.shape))
for this_ind in inds:
    this_dat = smooth_mean_gapes[this_ind]
    for this_name, this_taste in zip(taste_names, this_dat): 
        ax[this_ind].plot(t, this_taste, label = this_name)
    ax[this_ind].set_ylabel(
            emg_channel_basenames[this_ind[0]] + '\n' +\
                    'Rate of Gapes (Hz)')
    ax[this_ind].set_title(f'Laser : {unique_lasers[this_ind[1]]}')
ax[inds[-1]].legend()
fig.suptitle(f'Mean Smooth Gapes (smoothing = {kern_len}ms)')
fig.savefig(
        os.path.join(fin_plot_dir, 'mean_gapes_smooth.png'),
        dpi = 300)
plt.close(fig)
#plt.show()

hf5.close()
