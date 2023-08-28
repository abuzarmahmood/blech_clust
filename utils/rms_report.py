"""
Given a recording, generate a report and visualization of the RMS values
for the data
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.signal as signal

def calc_rms(x):
	"""
	Calculate the RMS of a signal
	"""
	return np.sqrt(np.mean(np.square(x)))

# set first argument to be the recording directory
parser = argparse.ArgumentParser()
parser.add_argument("recording_dir", help="directory of recording")
args = parser.parse_args()

data_dir = args.recording_dir
data_dir = '/home/abuzarmahmood/Desktop/test_RMS_230825_140130'

# data_dir timestamp
data_dir_timestamp = os.path.split(data_dir)[-1].split('_')[-2:]
data_dir_timestamp = '_'.join(data_dir_timestamp)

# Output to Desktop
home_dir = os.path.expanduser('~')
desktop_dir = os.path.join(home_dir, 'Desktop')
output_base_dir = os.path.join(desktop_dir, 'RMS_report')
output_dir = os.path.join(output_base_dir, data_dir_timestamp) 
if not os.path.isdir(output_dir):
	os.makedirs(output_dir)

# Use info file for port list calculation
info_file = np.fromfile(data_dir + '/info.rhd', dtype=np.dtype('float32'))
sampling_rate = int(info_file[2])

# get the data
file_list = os.listdir(data_dir)
data_file_list = sorted([file for file in file_list if 'amp' in file])
data = np.stack([np.fromfile(x, dtype = np.dtype('int16')) \
		for x in [os.path.join(data_dir, file) for file in data_file_list]])

# Scaled data to microvolts
data = data * 0.195

# Bandpass filter data at >250Hz
b, a = signal.butter(3, 250/(sampling_rate/2), btype='highpass')
data = signal.filtfilt(b, a, data, axis=1)

# Time vec
time_vec = np.arange(data.shape[1])/sampling_rate

# Chop off first and last 1s
data = data[:,sampling_rate:-sampling_rate]
time_vec = time_vec[sampling_rate:-sampling_rate]

# get rms values
rms_vals = np.apply_along_axis(calc_rms, 1, data)
rms_vals = np.round(rms_vals, 2)

# Plot data as subplots
fig,ax = plt.subplots(len(data),1,
					  sharex=True, sharey=True,
					  figsize=(3,20))
for i in range(len(data)):
	ax[i].plot(time_vec, data[i,:], linewidth=0.1)
	#ax[i].plot(downsampled_time_vec, downsampled_data[i,:])
	ax[i].set_ylabel(f'{data_file_list[i]}', rotation=0, labelpad=20)
ax[-1].set_xlabel('Time (s)')
fig.suptitle('Data for {}'.format(os.path.basename(data_dir)))
fig.savefig(os.path.join(output_dir, 'data.png'),
			bbox_inches='tight')
plt.close(fig)
#plt.show()

# Downsample data
downsample_factor = 100
downsampled_data = data[:,::downsample_factor]
downsampled_time_vec = time_vec[::downsample_factor]

# Calculate correlations on downsampled data
corr_mat = np.corrcoef(downsampled_data)
# Remove diagonal
corr_mat = corr_mat - np.diag(np.diag(corr_mat))
# Plot correlation matrix
fig,ax = plt.subplots()
im = ax.imshow(corr_mat)
ax.set_title('Correlation matrix')
plt.colorbar(im, ax=ax, label='Correlation')
fig.savefig(os.path.join(output_dir, 'corr_mat.png'))
plt.close(fig)

# Generate report of rms values per channel
# And save to output_dir
with open(os.path.join(output_dir, 'rms_report.txt'), 'w') as f:
	f.write('RMS values for {}\n'.format(os.path.basename(data_dir)))
	for i in range(len(rms_vals)):
		f.write(f'{data_file_list[i]}: {rms_vals[i]} uV\n')


