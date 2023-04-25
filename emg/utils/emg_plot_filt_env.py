# Create plots for filtered emg and respective envelope

# Import stuff
import numpy as np
from scipy.signal import butter, filtfilt, periodogram
import easygui
import os
import pylab as plt
import sys
import glob
import json

# Ask for the directory where the data (emg_data.npy) sits
# Get name of directory with the data files
if len(sys.argv) > 1:
    dir_name = os.path.abspath(sys.argv[1])
    if dir_name[-1] != '/':
        dir_name += '/'
else:
    dir_name = easygui.diropenbox(msg = 'Please select data directory')
os.chdir(dir_name)

############################################################
## Load params
############################################################
# Extract info experimental info file
dir_basename = os.path.basename(dir_name[:-1])
json_path = glob.glob(os.path.join(dir_name, dir_basename + '.info'))[0]
with open(json_path, 'r') as params_file:
    info_dict = json.load(params_file)
tastes = info_dict['taste_params']['tastes']
print(f'Tastes : {tastes}'+'\n')

#tastes = easygui.multenterbox(
#        msg = 'Enter the names of the tastes used in the experiments', 
#        fields = ['Taste{:d}'.format(i+1) for i in range(emg_env.shape[0])])

# Pull pre_stim duration from params file
params_file_name = glob.glob('./**.params')[0]
with open(params_file_name,'r') as params_file_connect:
    params_dict = json.load(params_file_connect)
durations = params_dict['spike_array_durations']
pre_stim = int(durations[0])
plot_params = params_dict['psth_params']['durations']

fin_inds = [pre_stim - plot_params[0], pre_stim + plot_params[1]]
time_vec = np.arange(-plot_params[0], plot_params[1])
print(f'Plotting from {-plot_params[0]}ms pre_stim to {plot_params[1]}ms post_stim\n')

#time_limits = easygui.multenterbox(
#        msg = 'Time limits for plotting [relative to stim delivery]', 
#        fields = ['Stim Delivery (ms)', 'Pre stim (ms)', 'Post stim (ms)'])
#time_limits = [int(x) for x in time_limits]
#fin_time_limits = time_limits[1:]
#fin_inds = [x + time_limits[0] for x in time_limits[1:]]
#time_vec = np.arange(*fin_time_limits)

############################################################
## Load data and generate plots 
############################################################
emg_output_dir = os.path.join(dir_name, 'emg_output')
channel_dirs = glob.glob(os.path.join(emg_output_dir,'emg*'))
channel_dirs = [x for x in channel_dirs if os.path.isdir(x)]
channels_discovered = [os.path.basename(x) for x in channel_dirs]
print(f'Creating plots for : {channels_discovered}\n')

for this_dir in channel_dirs:
    os.chdir(this_dir)
    #emg_data = np.load('emg_data.npy')
    emg_filt = np.load('emg_filt.npy')
    emg_env = np.load('emg_env.npy')
    sig_trials = np.load('sig_trials.npy')


    colors = ['r','b']

    cut_emg_filt = emg_filt[...,fin_inds[0]:fin_inds[1]]
    cut_emg_env = emg_env[...,fin_inds[0]:fin_inds[1]]

    fig,ax = plt.subplots(*emg_filt.shape[:2][::-1], 
            sharey=True, sharex=True, figsize = (20,30))
    inds = list(np.ndindex(ax.shape))
    for trial, taste in inds:
        this_color = colors[int(sig_trials[taste,trial])]
        ax[trial,taste].plot(time_vec, cut_emg_filt[taste,trial], 
                c = this_color)
        if trial == 0:
            this_taste = tastes[taste]
            ax[trial,taste].set_title(this_taste)
        if trial == emg_filt.shape[1]-1:
            ax[trial,taste].set_xlabel('Time post-sitm (ms)')
    plt.suptitle('Red --> Not significant, Blue --> Significant')
    plt.subplots_adjust(top = 0.95)
    fig.savefig('emg_filtered_plots.png', bbox_inches = 'tight')
    plt.close(fig)
    #plt.show()

    fig,ax = plt.subplots(*emg_filt.shape[:2][::-1], 
            sharey=True, sharex=True, figsize = (20,30))
    inds = list(np.ndindex(ax.shape))
    for trial, taste in inds:
        this_color = colors[int(sig_trials[taste,trial])]
        ax[trial,taste].plot(time_vec, cut_emg_env[taste,trial], 
                c = this_color)
        if trial == 0:
            this_taste = tastes[taste]
            ax[trial,taste].set_title(this_taste)
        if trial == emg_filt.shape[1]-1:
            ax[trial,taste].set_xlabel('Time post-sitm (ms)')
    plt.suptitle('Red --> Not significant, Blue --> Significant')
    plt.subplots_adjust(top = 0.95)
    fig.savefig('emg_envelope_plots.png', bbox_inches = 'tight')
    plt.close(fig)
    plt.show()
