# Necessary python modules
import easygui
import os
import tables
import sys
import numpy as np
import multiprocessing
import json
import glob
import pandas as pd
import shutil

# Necessary blech_clust modules
from utils import read_file
from utils.blech_utils import entry_checker

# Get blech_clust path
blech_clust_path = ('/').join(os.path.abspath(__file__).split('/')[0:-1])

############################################################

# Get name of directory with the data files
if len(sys.argv) > 1:
    dir_name = os.path.abspath(sys.argv[1])
    if dir_name[-1] != '/':
        dir_name += '/'
else:
    dir_name = easygui.diropenbox(msg = 'Please select data directory') + '/'

print(f'Processing : {dir_name}')

# Change to that directory
os.chdir(dir_name)

# Check that experimental_info json file is present
# If not present, refuse to cooperate
dir_basename = os.path.basename(dir_name[:-1])
json_path = glob.glob(os.path.join(dir_name, dir_basename + '.info'))
if len(json_path) == 0:
    raise Exception('Must have experimental info json before proceeding \n'\
            'Run blech_exp_info.py first \n'\
            '== Exiting ==')
    exit()

# Get the names of all files in this directory
file_list = os.listdir('./')

# Get the type of data files (.rhd or .dat)
#HANNAH CHANGE: ADDED TEST OF ONE FILE PER SIGNAL TYPE
try:
	file_list.index('auxiliary.dat')
	file_type = ['one file per signal type']
except:
	file_type = ['one file per channel']

# Create hdf5 file, and make groups for raw data, raw emgs, 
# digital outputs and digital inputs, and close

# Grab directory name to create the name of the hdf5 file
# If HDF5 present, use that, otherwise, create new one
h5_search = glob.glob('*.h5')
if len(h5_search):
    hdf5_name = h5_search[0] 
    print(f'HDF5 file found...Using file {hdf5_name}')
    hf5 = tables.open_file(hdf5_name, 'r+')
else:
    hdf5_name = str(os.path.dirname(dir_name)).split('/')[-1]+'.h5'
    print(f'No HDF5 found...Creating file {hdf5_name}')
    hf5 = tables.open_file(hdf5_name, 'w', title = hdf5_name[-1])

group_list = ['raw','raw_emg','digital_in','digital_out']
for this_group in group_list:
    if '/'+this_group in hf5:
        hf5.remove_node('/', this_group, recursive=True)
    hf5.create_group('/',this_group)
#hf5.create_group('/', 'raw')
#hf5.create_group('/', 'raw_emg')
#hf5.create_group('/', 'digital_in')
#hf5.create_group('/', 'digital_out')
hf5.close()
print('Created nodes in HF5')

# Create directories to store waveforms, spike times, clustering results, and plots
# And a directory for dumping files talking about memory usage in blech_process.py
# Check if dirs are already present, if they are, ask to delete and continue
# or abort
dir_list = ['spike_waveforms',
            'spike_times',
            'clustering_results',
            'Plots',
            'memory_monitor_clustering']
dir_exists = [x for x in dir_list if os.path.exists(x)]
recreate_msg = f'Following dirs are present :' + '\n' + f'{dir_exists}' + \
                '\n' + 'Overwrite dirs? (yes/y/n/no) ::: '

# If dirs exist, check with user
if len(dir_exists) > 0:
    recreate_str, continue_bool = entry_checker(\
            msg = recreate_msg,
            check_func = lambda x: x in ['y','yes','n','no'], 
            fail_response = 'Please enter (yes/y/n/no)')
# Otherwise, create all of them
else:
    continue_bool = True
    recreate_str = 'y'

# Break if user said n/no or gave exit signal
if continue_bool:
    if recreate_str in ['y','yes']:
        for x in dir_list:
            if os.path.exists(x):
                shutil.rmtree(x)
            os.makedirs(x)
    else:
        quit()
else:
    quit()

#os.mkdir('spike_waveforms')
#os.mkdir('spike_times')
#os.mkdir('clustering_results')
#os.mkdir('Plots')
#os.mkdir('memory_monitor_clustering')
print('Created dirs in data folder')

#Get lists of amplifier and digital input files
if file_type == ['one file per signal type']:
	electrodes_list = ['amplifier.dat']
	dig_in_list = ['digitalin.dat']
elif file_type == ['one file per channel']:
	electrodes_list = [name for name in file_list if name.startswith('amp-')]
	dig_in_list = [name for name in file_list if name.startswith('board-DI')]

#Use info file for port list calculation
info_file = np.fromfile(dir_name + '/info.rhd', dtype = np.dtype('float32'))
sampling_rate = int(info_file[2])

# Read the time.dat file for use in separating out the one file per signal type data
num_recorded_samples = len(np.fromfile(dir_name + '/' + 'time.dat', dtype = np.dtype('float32')))
total_recording_time = num_recorded_samples/sampling_rate #In seconds

check_str = f'Amplifier files: {electrodes_list} \nSampling rate: {sampling_rate} Hz'\
           f'\nDigital input files: {dig_in_list} \n ---------- \n \n'
print(check_str)

if file_type == ['one file per channel']:
	# Get the amplifier ports used
	ports = list(np.unique(np.array([f[4] for f in file_list if f[:4] == 'amp-'])))
	# Sort the ports in alphabetical order
	ports.sort()
	
	## Pull out the digital input channels used, and convert them to integers
	#dig_in = list(set(f[11:13] for f in file_list if f[:9] == 'board-DIN'))
	#for i in range(len(dig_in)):
	#	dig_in[i] = int(dig_in[i][0])
	#dig_in.sort()
	
	# Read dig-in data
	# Pull out the digital input channels used, 
	# and convert them to integers
	dig_in_files = [x for x in file_list if "DI" in x]
	dig_in = [x.split('-')[-1].split('.')[0] for x in dig_in_files]
	dig_in = sorted([int(x) for x in dig_in])
elif file_type == ['one file per signal type']:
	print("\tSingle Amplifier File Detected")
	#Import amplifier data and calculate the number of electrodes
	print("\t\tCalculating Number of Ports")
	amplifier_data = np.fromfile(dir_name + '/' + electrodes_list[0], dtype = np.dtype('uint16'))
	num_electrodes = int(len(amplifier_data)/num_recorded_samples)
	ports = list(np.arange(num_electrodes))
	del amplifier_data, num_electrodes
	#Import digin data and calculate the number of digins
	print("\t\tCalculating Number of Dig-Ins")
	dig_in_data = np.fromfile(dir_name + '/' + dig_in_list[0], dtype=np.dtype('uint16'))
	d_inputs_str = dig_in_data.astype('str')
	del dig_in_data
	d_in_str_int = d_inputs_str.astype('int64')
	del d_inputs_str
	d_diff = np.diff(d_in_str_int)
	del d_in_str_int
	dig_in = list(np.unique(np.abs(d_diff)) - 1)
	dig_in.remove(-1)
	del d_diff

check_str = f'ports used: {ports} \n sampling rate: {sampling_rate} Hz'\
            f'\n digital inputs on intan board: {dig_in}'

print(check_str)

with open(json_path[0], 'r') as params_file:
    info_dict = json.load(params_file)

all_car_group_vals = []
for region_name, region_elecs in info_dict['electrode_layout'].items():
    if not region_name == 'emg':
        for group in region_elecs:
            if len(group) > 0:
                all_car_group_vals.append(group)
all_electrodes = [electrode for region in all_car_group_vals \
                        for electrode in region]

emg_info = info_dict['emg']
emg_port = emg_info['port']
emg_channels = sorted(emg_info['electrodes'])


layout_path = glob.glob(os.path.join(dir_name,"*layout.csv"))[0]
electrode_layout_frame = pd.read_csv(layout_path) 

# Create arrays for each electrode
#read_file.create_hdf_arrays(hdf5_name, all_electrodes, 
#                            dig_in, emg_port, emg_channels)

# Read data files, and append to electrode arrays
if file_type == ['one file per channel']:
	#read_file.read_files_abu(hdf5_name, dig_in, electrode_layout_frame) 
	read_file.read_digins(hdf5_name, dig_in, dig_in_list)
	read_file.read_electrode_channels(hdf5_name, electrode_layout_frame)
	if len(emg_channels) > 0:
	    read_file.read_emg_channels(hdf5_name, electrode_layout_frame)
elif file_type == ['one file per signal type']:
	read_file.read_digins_single_file(hdf5_name, dig_in, dig_in_list)
	#This next line takes care of both electrodes and emgs
	read_file.read_electrode_emg_channels_single_file(hdf5_name, electrode_layout_frame, electrodes_list, num_recorded_samples, emg_channels)

# Write out template params file to directory if not present
#home_dir = os.getenv('HOME')
#blech_clust_path = os.path.join(home_dir,'Desktop','blech_clust')
print(blech_clust_path)
params_template_path = os.path.join(
        blech_clust_path,
        'params/sorting_params_template.json')
params_template = json.load(open(params_template_path,'r'))
# Info on taste digins and laser should be in exp_info file
all_params_dict = params_template.copy() 
all_params_dict['sampling_rate'] = sampling_rate

params_out_path = hdf5_name.split('.')[0] + '.params'
if not os.path.exists(params_out_path):
    print('No params file found...Creating new params file')
    with open(params_out_path, 'w') as params_file:
        json.dump(all_params_dict, params_file, indent = 4)
else:
    print("Params file already present...not writing a new one")

# Dump shell file(s) for running GNU parallel job on the user's blech_clust folder on the desktop
# First get number of CPUs - parallel be asked to run num_cpu-1 threads in parallel
num_cpu = multiprocessing.cpu_count()

electrode_bool = electrode_layout_frame.loc[
        electrode_layout_frame.electrode_ind.isin(all_electrodes)]
not_none_bool = electrode_bool.loc[~electrode_bool.CAR_group.isin(["none","None",'na'])]
bash_electrode_list = not_none_bool.electrode_ind.values
job_count = np.min((len(bash_electrode_list), int(num_cpu-2)))
# todo: Account for electrodes labelled none when writing parallel command
runner_path = os.path.join(blech_clust_path,'blech_clust_jetstream_parallel1.sh') 
f = open(os.path.join(blech_clust_path,'blech_clust_jetstream_parallel.sh'), 'w')
print(f"parallel -k -j {job_count} --noswap --load 100% --progress " +\
        "--memfree 4G --retry-failed "+\
        f"--joblog {dir_name}/results.log "+\
        f"bash {runner_path} "+\
        #f"::: {{{','.join([str(x) for x in bash_electrode_list])}}}", 
        f"::: {' '.join([str(x) for x in bash_electrode_list])}", 
        file = f)
f.close()

# Then produce the file that runs blech_process.py
f = open(os.path.join(blech_clust_path,'blech_clust_jetstream_parallel1.sh'), 'w')
print("export OMP_NUM_THREADS=1", file = f)
blech_process_path = os.path.join(blech_clust_path,'blech_process.py')
print(f"python {blech_process_path} $1", file=f)
f.close()

# Dump the directory name where blech_process has to cd
f = open(os.path.join(blech_clust_path,'blech.dir'), 'w')
print(dir_name, file=f)
f.close()

print('blech_clust.py complete \n')
print('*** Please check params file to make sure all is good ***\n')
