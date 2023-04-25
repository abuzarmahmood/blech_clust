import tables
import numpy as np
import easygui
import os
import matplotlib.pyplot as plt
import glob
import sys

sys.path.append('..')
from utils.blech_utils import imp_metadata

# Get name of directory with the data files
metadata_handler = imp_metadata(sys.argv)
dir_name = metadata_handler.dir_name
os.chdir(dir_name)
print(f'Processing : {dir_name}')

# Open the hdf5 file
hf5 = tables.open_file(metadata_handler.hdf5_name, 'r+')

# Grab the nodes for the available tastes
# Take trial counts from emg arrays
# TODO: This needs to be done for all emg "channels"
emg_data = np.load('emg_output/emg_data.npy')
num_trials = emg_data.shape[2]
num_tastes = emg_data.shape[1]

# Load the unique laser duration/lag combos and the trials 
# that correspond to them from the ancillary analysis node
trials = hf5.root.ancillary_analysis.trials[:]
unique_lasers = hf5.root.ancillary_analysis.laser_combination_d_l[:]

# Iterate over channels
output_list = glob.glob(os.path.join(dir_name,'emg_output/*'))
channel_dirs = sorted([x for x in output_list if os.path.isdir(x)])
channels_discovered = [os.path.basename(x) for x in channel_dirs]
channels_discovered = [x for x in channels_discovered if 'emg' in x]
print(f'Creating plots for : {channels_discovered}\n')

# Save order of EMG channels to emg_data_readme.txt
path_to_info = os.path.join(dir_name,'emg_output','emg_data_readme.txt')
channels_discovered_dict = dict(zip(range(len(channels_discovered)), channels_discovered))
out_str = '\n\n' + f'Output channel order : {channels_discovered_dict}'
with open(path_to_info,'a') as outfile:
    outfile.write(out_str)

final_gapes_list = []
final_ltps_list = []
final_sig_trials_list = []
final_emg_BSA_list = []

for num, this_dir in enumerate(channels_discovered):
    #os.chdir(this_dir)
    this_basename = channels_discovered[num]

    # Now run through the tastes, and stack up the BSA results 
    # for the EMG responses by trials
    #emg_BSA_results = hf5.root.emg_BSA_results.taste0_p[:, :, :]
    #for i in range(num_tastes - 1):
    #	exec("emg_BSA_results = np.vstack((emg_BSA_results[:], hf5.root.emg_BSA_results.taste" + str(i+1) + "_p[:, :, :]))")
    emg_BSA_results = [x[:] for x in \
            hf5.get_node('/emg_BSA_results',this_basename)._f_iter_nodes()\
            if 'taste' in x.name]
    emg_BSA_results = np.vstack(emg_BSA_results)

    ## Find the frequency with the maximum EMG power at each time point on each trial
    #max_freq = np.argmax(emg_BSA_results[:, :, :], axis = 2)
    ## Gapes are anything upto 4.6 Hz
    #gapes = np.array(max_freq <= 7, dtype = int)
    ## LTPs are from 5.95 Hz to 8.65 Hz
    #ltps = np.array((max_freq >= 10)*(max_freq <= 16), dtype = int)
    #Alternatively, gapes from 3.65-5.95 Hz (6-11). LTPs from 5.95 to 8.65 Hz (11-17) 
    gapes = np.sum(emg_BSA_results[:, :, 6:11], axis = 2)/\
            np.sum(emg_BSA_results, axis = 2)
    ltps = np.sum(emg_BSA_results[:, :, 11:], axis = 2)/\
            np.sum(emg_BSA_results, axis = 2)

    # Also load up the array of significant trials 
    # (trials where the post-stimulus response is at least 
    # 4 stdev above the pre-stimulus response)
    # TODO: Needs to refer to sig_trials within a channel
    sig_trials = np.load(f'emg_output/{this_basename}/sig_trials.npy').flatten()

    # TODO: These arrays need to be able to handle uneven trials
    # One way to do that would be to use the max number of deliveries
    # for the trial dimension

    # Now arrange these arrays by 
    # SHAPE : laser condition X taste X trial X time
    final_emg_BSA_results = np.empty((len(trials), 
                                        num_tastes, 
                                        int(num_trials/len(trials)),  
                                        emg_BSA_results.shape[1], 
                                        emg_BSA_results.shape[2]), 
                                    dtype = float) 
    final_gapes = np.empty((len(trials), 
                            num_tastes, 
                            int(num_trials/len(trials)),  
                            gapes.shape[1]), 
                        dtype = float)
    final_ltps = np.empty((len(trials), 
                            num_tastes, 
                            int(num_trials/len(trials)), 
                            ltps.shape[1]), 
                        dtype = float)
    final_sig_trials = np.empty((len(trials), 
                                num_tastes, 
                                int(num_trials/len(trials))), 
                            dtype = float)

    # Fill up these arrays
    for i in range(len(trials)):
        for j in range(num_tastes):
            final_emg_BSA_results[i, j, :, :, :] = \
                    emg_BSA_results[trials[i][np.where((trials[i] >= num_trials*j)*(trials[i] < num_trials*(j+1)) == True)], :, :]
            final_gapes[i, j, :,  :] = \
                    gapes[trials[i][np.where((trials[i] >= num_trials*j)*(trials[i] < num_trials*(j+1)) == True)], :]
            final_ltps[i, j, :, :] = \
                    ltps[trials[i][np.where((trials[i] >= num_trials*j)*(trials[i] < num_trials*(j+1)) == True)], :]
            final_sig_trials[i, j, :] = \
                    sig_trials[trials[i][np.where((trials[i] >= num_trials*j)*(trials[i] < num_trials*(j+1)) == True)]]

    final_gapes_list.append(final_gapes)
    final_ltps_list.append(final_ltps)
    final_sig_trials_list.append(final_sig_trials)
    final_emg_BSA_list.append(final_emg_BSA_results)

# SHAPE : channel x laser_cond x taste x trial x time
final_gapes_array = np.stack(final_gapes_list)
# SHAPE : channel x laser_cond x taste x trial x time
final_ltps_array = np.stack(final_ltps_list)
# SHAPE : channel x laser_cond x taste x trial 
final_sig_trials_array = np.stack(final_sig_trials_list)
# SHAPE : channel x laser_cond x taste x trial x time x freq
final_emg_BSA_array = np.stack(final_emg_BSA_list)

# Save under emg_BSA_results to segregate output better 
try:
    hf5.remove_node('/emg_BSA_results/gapes')
    hf5.remove_node('/emg_BSA_results/ltps')
    hf5.remove_node('/emg_BSA_results/sig_trials')
    hf5.remove_node('/emg_BSA_results/emg_BSA_results_final')
except:
    pass
hf5.create_array('/emg_BSA_results', 'gapes', final_gapes_array)
hf5.create_array('/emg_BSA_results', 'ltps', final_ltps_array)
hf5.create_array('/emg_BSA_results', 'sig_trials', final_sig_trials_array)
hf5.create_array('/emg_BSA_results', 'emg_BSA_results_final', final_emg_BSA_array)

hf5.flush()

hf5.close()
