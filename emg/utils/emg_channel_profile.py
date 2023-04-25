"""
Plot filtered EMG and envelope
This is supposed to be a "profile" without knowing any information about
assignment of EMG channels (if there is more than one pair)
Suggested layout : Figure contains all channels x trials
                    Different tastes on different figures
"""
# Import stuff
import numpy as np
from scipy.signal import butter, filtfilt, freqz
import easygui
import os
import sys
from tqdm import tqdm
import shutil
from scipy.stats import zscore
import pylab as plt

# Get name of directory with the data files
if len(sys.argv) > 1:
    dir_name = os.path.abspath(sys.argv[1])
    if dir_name[-1] != '/':
        dir_name += '/'
else:
    dir_name = easygui.diropenbox(msg = 'Please select data directory')

os.chdir(dir_name)

# Load the data
# shape : channels x tastes x trials x time
emg_data = np.load('emg_data.npy')

# Ask the user for the names of the tastes in the dataset
tastes = easygui.multenterbox(
        msg = 'Enter the names of the tastes used in the experiments', 
        fields = ['Taste{:d}'.format(i+1) for i in range(emg_data.shape[1])])

time_limits = easygui.multenterbox(
        msg = 'Time limits for plotting [relative to stim delivery]', 
        fields = ['Stim Delivery (ms)', 'Pre stim (ms)', 'Post stim (ms)'])
time_limits = [int(x) for x in time_limits]
fin_time_limits = time_limits[1:]
fin_inds = [x + time_limits[0] for x in time_limits[1:]]
time_vec = np.arange(*fin_time_limits)

# Get coefficients for Butterworth filters
m, n = butter(2, 2.0*300.0/1000.0, 'highpass')
c, d = butter(2, 2.0*15.0/1000.0, 'lowpass')

# Iterate over trials and apply frequency filter
iters = list(np.ndindex(emg_data.shape[:-1])) 
emg_filt = np.zeros(emg_data.shape)
emg_env = np.zeros(emg_data.shape)
for this_iter in iters:
    temp_filt = filtfilt(m, n, emg_data[this_iter])
    emg_filt[this_iter] = temp_filt 
    emg_env[this_iter] = filtfilt(c, d, np.abs(temp_filt))

cut_emg_filt = emg_filt[...,fin_inds[0]:fin_inds[1]]
cut_emg_env = emg_env[...,fin_inds[0]:fin_inds[1]]

# Zscore every channel so all channels can be plotted on same plot
# DO NOT zscore every trial individually
cut_emg_filt = np.stack([zscore(x,axis=None) for x in cut_emg_filt])
cut_emg_env = np.stack([zscore(x,axis=None) for x in cut_emg_env])

fig,ax = plt.subplots(*cut_emg_filt.shape[1:3][::-1], 
        sharey=True, sharex=True, figsize = (20,cut_emg_filt.shape[2]))
inds = list(np.ndindex(ax.shape))
for trial, taste in inds:
    this_dat = cut_emg_filt[:, taste, trial]
    for num, val in enumerate(this_dat): 
        ax[trial,taste].plot(
                time_vec, 
                val + 3*num,
                linewidth = 1,
                label = num) 
    ax[trial,taste].set_ylim([-3, emg_filt.shape[0]*3])
    ax[trial,taste].set_ylabel(trial)
    if trial == 0:
        this_taste = tastes[taste]
        ax[trial,taste].set_title(this_taste)
        ax[trial,taste].legend()
    if trial == emg_filt.shape[2]-1:
        ax[trial,taste].set_xlabel('Time post-sitm (ms)')
fig.savefig('emg_filtered_plots.png', bbox_inches = 'tight')
plt.close(fig)
plt.show()

fig,ax = plt.subplots(*cut_emg_filt.shape[1:3][::-1], 
        sharey=True, sharex=True, figsize = (20,cut_emg_filt.shape[2]))
inds = list(np.ndindex(ax.shape))
for trial, taste in inds:
    this_dat = cut_emg_env[:, taste, trial]
    for num, val in enumerate(this_dat): 
        ax[trial,taste].plot(
                time_vec, 
                val + 3*num,
                linewidth = 1,
                label = num) 
    ax[trial,taste].set_ylim([-3, emg_filt.shape[0]*3])
    ax[trial,taste].set_ylabel(trial)
    if trial == 0:
        this_taste = tastes[taste]
        ax[trial,taste].set_title(this_taste)
        ax[trial,taste].legend()
    if trial == emg_filt.shape[2]-1:
        ax[trial,taste].set_xlabel('Time post-sitm (ms)')
fig.savefig('emg_envelope_plots.png', bbox_inches = 'tight')
plt.close(fig)
plt.show()
