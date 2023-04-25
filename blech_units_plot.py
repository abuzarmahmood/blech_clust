# Import stuff!
import numpy as np
import tables
import easygui
import sys
import os
import matplotlib.pyplot as plt
import shutil

# Import 3rd part code
from utils import blech_waveforms_datashader
from utils import memory_monitor as mm
from utils.blech_utils import imp_metadata

# Get name of directory with the data files
metadata_handler = imp_metadata(sys.argv)
dir_name = metadata_handler.dir_name
os.chdir(dir_name)
print(f'Processing : {dir_name}')

# Open the hdf5 file
hf5 = tables.open_file(metadata_handler.hdf5_name, 'r+')

# Get all the units from the hdf5 file
units = hf5.list_nodes('/sorted_units')

# Delete and remake a directory for storing the plots of the unit waveforms (if it exists)
try:
    shutil.rmtree("unit_waveforms_plots", ignore_errors = True)
except:
    pass
os.mkdir("unit_waveforms_plots")

# Now plot the waveforms from the units in this directory one by one
for unit in range(len(units)):
        waveforms = units[unit].waveforms[:]
        x = np.arange(waveforms.shape[1]) + 1
        fig, ax = blech_waveforms_datashader.\
                waveforms_datashader(waveforms, x, downsample=False)
        ax.set_xlabel('Sample (30 samples per ms)')
        ax.set_ylabel('Voltage (microvolts)')
        ax.set_title('Unit %i, total waveforms = %i' % (unit, waveforms.shape[0]) \
                + '\n' + 'Electrode: %i, Single Unit: %i, RSU: %i, FS: %i' %\
                (hf5.root.unit_descriptor[unit]['electrode_number'], \
                hf5.root.unit_descriptor[unit]['single_unit'], \
                hf5.root.unit_descriptor[unit]['regular_spiking'], \
                hf5.root.unit_descriptor[unit]['fast_spiking']))
        fig.savefig('./unit_waveforms_plots/Unit%02d.png' % (unit))
        plt.close("all")
        
        # Also plot the mean and SD for every unit - 
        # downsample the waveforms 10 times to remove effects of upsampling during de-jittering
        fig = plt.figure()
        plt.plot(x, np.mean(waveforms, axis = 0), linewidth = 4.0)
        plt.fill_between(
                x, 
                np.mean(waveforms, axis = 0) - np.std(waveforms, axis = 0), 
                np.mean(waveforms, axis = 0) + np.std(waveforms, axis = 0), 
                alpha = 0.4)
        plt.xlabel('Sample (30 samples per ms)')
        plt.ylabel('Voltage (microvolts)')
        plt.title('Unit %i, total waveforms = %i' % (unit, waveforms.shape[0]) + \
                '\n' + 'Electrode: %i, Single Unit: %i, RSU: %i, FS: %i' % \
                (hf5.root.unit_descriptor[unit]['electrode_number'], \
                hf5.root.unit_descriptor[unit]['single_unit'], \
                hf5.root.unit_descriptor[unit]['regular_spiking'], 
                hf5.root.unit_descriptor[unit]['fast_spiking']))
        fig.savefig('./unit_waveforms_plots/Unit%i_mean_sd.png' % (unit))
        plt.close("all")

hf5.close()

