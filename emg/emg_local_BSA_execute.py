# Runs a local BSA analysis (see emg_local_BSA.py) on one trial of EMG data. Runs on the HPC

# Import stuff
import numpy as np
import easygui
import os
import sys

# Read blech.dir, and cd to that directory. 
with open('BSA_run.dir', 'r') as f:
    dir_list = [x.strip() for x in f.readlines()]

# If there is more than one dir in BSA_run.dir,
# loop over both, as both sets will have the same number of trials
for dir_name in dir_list: 
    os.chdir(dir_name)

    # Read the data files
    emg_env = np.load('emg_env.npy')
    sig_trials = np.load('sig_trials.npy')

    # cd to emg_BSA_results
    os.chdir('emg_BSA_results')

    # Get SGE_TASK_ID # - this will determine the taste+trial to be looked at
    try:
        task = int(os.getenv('SGE_TASK_ID'))
    except:
        # Alternatively, if running on jetstream (or personal computer) 
        # using GNU parallel, get sys.argv[1]
        task = int(sys.argv[1])

    taste = int((task-1)/sig_trials.shape[-1])
    trial = int((task-1)%sig_trials.shape[-1])

    # Import R related stuff - use rpy2 for Python->R and pandas for R->Python
    # Needed for the next line to work on Anaconda. 
    # Also needed to do conda install -c r rpy2 at the command line
    import readline 
    import rpy2.robjects as ro
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    from rpy2.robjects.packages import importr
    # rpy.common got deprecated in newer versions of pandas. So we use rpy2 instead
    #import pandas.rpy.common as com
    from rpy2.robjects import r

    # Fire up BaSAR on R
    basar = importr('BaSAR')

    # Make the time array and assign it to t on R
    T = (np.arange(7000) + 1)/1000.0
    t_r = ro.r.matrix(T, nrow = 1, ncol = 7000)
    ro.r.assign('t_r', t_r)
    ro.r('t = c(t_r)')

    # Make arrays to store the posterior probabilities and frequencies of analysis
    p = np.zeros((7000, 20))
    omega = np.zeros(20)

    # Run BSA on trial 'trial' of taste 'taste' and assign the results to p and omega.
    input_data = emg_env[taste, trial, :]
    # Check that trial is non-zero, if it isn't, don't try to run BSA
    if input_data.sum() == 0:
        raise Exception('Input data is all zeros, BSA will not run')
        exit()

    Br = ro.r.matrix(input_data, nrow = 1, ncol = 7000)
    ro.r.assign('B', Br)
    ro.r('x = c(B[1,])')

    # x is the data, 
    # we scan periods from 0.1s (10 Hz) to 1s (1 Hz) in 20 steps. 
    # Window size is 300ms. 
    # There are no background functions (=0)
    ro.r('r_local = BaSAR.local(x, 0.1, 1, 20, t, 0, 300)') 
    #p_r = com.load_data('r_local')
    p_r = r['r_local']
    # r_local is returned as a length 2 object, 
    # with the first element being omega and the second being the 
    # posterior probabilities. These need to be recast as floats
    r_p = np.array(p_r[1]).astype('float')
    p[:, :] = r_p[:, :]
    omega[:] = np.array(p_r[0]).astype('float')/(2.0*np.pi) 

    # Save p and omega by taste and trial number
    #np.save('taste%i_trial%i_p.npy' % (taste, trial), p)
    #np.save('taste%i_trial%i_omega.npy' % (taste, trial), omega)
    np.save(f'taste{taste:02}_trial{trial:02}_p.npy', p)
    np.save(f'taste{taste:02}_trial{trial:02}_omega.npy', omega)
