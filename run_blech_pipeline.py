
"""
Creating a Prefect pipeline for running blech_clust up till post-processing
"""
import os
from subprocess import PIPE, Popen
from prefect import flow, task
from glob import glob
import json
import argparse

############################################################
parser = argparse.ArgumentParser(
    description='Run blech_clust pipeline on data in a directory')
parser.add_argument('dir', type=str,
                    help='Path to directory containing data to be processed')
parser.add_argument('-pre', action='store_true',
                    help='Run preprocessing steps')
parser.add_argument('-post', action='store_true',
                    help='Run postprocessing steps')
args = parser.parse_args()


def raise_error_if_error(process, stderr, stdout):
    print(stdout.decode('utf-8'))
    if process.returncode:
        decode_err = stderr.decode('utf-8')
        raise Exception(decode_err)


############################################################
# Define paths
############################################################
# Define paths
# TODO: Replace with call to blech_process_utils.path_handler
script_path = os.path.realpath(__file__)
blech_clust_dir = os.path.dirname(script_path)

# Read emg_env path
with open(os.path.join(blech_clust_dir, 'params', 'env_params.json')) as f:
    env_params = json.load(f)
emg_env_path = env_params['emg_env']

data_dir = args.dir

############################################################
# Data Prep Scripts
############################################################


def check_data_present():
    if os.path.isdir(data_dir):
        return True
    else:
        raise Exception('Data directory does not exist')

############################################################
# Common Scripts
############################################################


@task(log_prints=True)
def reset_blech_clust():
    script_name = './pipeline_testing/reset_blech_clust.py'
    process = Popen(["python", script_name],
                    stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process, stderr, stdout)


@task(log_prints=True)
def run_clean_slate(data_dir):
    script_name = 'blech_clean_slate.py'
    process = Popen(["python", script_name, data_dir],
                    stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process, stderr, stdout)


#@task(log_prints=True)
#def run_exp_info(data_dir):
#    script_name = 'blech_exp_info.py'
#    process = Popen(["python", script_name, data_dir],
#                    stdout=PIPE, stderr=PIPE)
#    stdout, stderr = process.communicate()
#    raise_error_if_error(process, stderr, stdout)


@task(log_prints=True)
def run_blech_clust(data_dir):
    script_name = 'blech_clust.py'
    process = Popen(["python", script_name, data_dir],
                    stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process, stderr, stdout)


@task(log_prints=True)
def make_arrays(data_dir):
    script_name = 'blech_make_arrays.py'
    process = Popen(["python", script_name, data_dir],
                    stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process, stderr, stdout)

############################################################
# Spike Only
############################################################


@task(log_prints=True)
def run_CAR(data_dir):
    script_name = 'blech_common_avg_reference.py'
    process = Popen(["python", script_name, data_dir],
                    stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process, stderr, stdout)


@task(log_prints=True)
def run_jetstream_bash(data_dir):
    script_name = 'blech_clust_jetstream_parallel.sh'
    process = Popen(["bash", script_name],
                    stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process, stderr, stdout)


@task(log_prints=True)
def select_clusters(data_dir):
    script_name = 'pipeline_testing/select_some_waveforms.py'
    process = Popen(["python", script_name, data_dir],
                    stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process, stderr, stdout)


@task(log_prints=True)
def post_process(data_dir):
    script_name = 'blech_post_process.py'
    plot_flag = '-p ' + 'False'
    dir_flag = '-d' + data_dir
    sorted_units_path = glob(os.path.join(data_dir, '*sorted_units.csv'))[0]
    file_flag = '-f' + sorted_units_path
    process = Popen(["python", script_name, plot_flag, dir_flag, file_flag],
                    stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process, stderr, stdout)


@task(log_prints=True)
def units_similarity(data_dir):
    script_name = 'blech_units_similarity.py'
    process = Popen(["python", script_name, data_dir],
                    stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process, stderr, stdout)


@task(log_prints=True)
def units_plot(data_dir):
    script_name = 'blech_units_plot.py'
    process = Popen(["python", script_name, data_dir],
                    stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process, stderr, stdout)


@task(log_prints=True)
def make_psth(data_dir):
    script_name = 'blech_make_psth.py'
    process = Popen(["python", script_name, data_dir],
                    stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process, stderr, stdout)


@task(log_prints=True)
def pal_iden_setup(data_dir):
    script_name = 'blech_palatability_identity_setup.py'
    process = Popen(["python", script_name, data_dir],
                    stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process, stderr, stdout)


@task(log_prints=True)
def overlay_psth(data_dir):
    script_name = 'blech_overlay_psth.py'
    process = Popen(["python", script_name, data_dir],
                    stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process, stderr, stdout)

############################################################
# Define Flows
############################################################


@flow(log_prints=True)
def blech_pre():
    check_data_present()
    os.chdir(blech_clust_dir)
    reset_blech_clust()
    run_clean_slate(data_dir)
    #run_exp_info(data_dir)
    run_blech_clust(data_dir)
    run_CAR(data_dir)
    run_jetstream_bash(data_dir)


@flow(log_prints=True)
def blech_post():
    check_data_present()
    os.chdir(blech_clust_dir)
    units_similarity(data_dir)
    units_plot(data_dir)
    make_arrays(data_dir)
    make_psth(data_dir)
    pal_iden_setup(data_dir)
    overlay_psth(data_dir)


############################################################
# Run Flows
############################################################
if __name__ == '__main__':
    if args.pre:
        blech_pre()
    elif args.post:
        blech_post()
    else:
        print('No flow selected')
