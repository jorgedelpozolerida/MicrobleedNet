import os
import subprocess
from multiprocessing import Pool


# Define the base directories for the three levels
level1_dirs = ['Scratch-Pretrained-FineTuned', 'TL-Pretrained-FineTuned']
level2_dirs = ['predict_cmb_crb', 'predict_cmb_dou', 'predict_cmb_valid']
level3_dirs = ['F1macro', 'valloss']

# Base directory paths
base_prediction_dir = '/storage/evo1/jorge/datasets/cmb/predictions_last'
synthseg_dir = '/storage/evo1/jorge/datasets/cmb/synthseg_masks_resampled/'
metadata_csv_path = '/storage/evo1/jorge/MicrobleedNet/data-misc/csv/CMB_metadata_all.csv'
savedir_base  = "/storage/evo1/jorge/datasets/cmb/evaluations_cmb"
base_groundtruth_dir = "/storage/evo1/jorge/datasets/cmb/{dataset}/Data"

def run_evaluation(command):
    """
    Function to run a subprocess with the given command.
    """
    print("Running command:", ' '.join(command))
    subprocess.run(command)
    print(f"Completed command: {' '.join(command)}")

def generate_commands():
    """
    Generator that yields command lists to be run in subprocesses.
    """
    for l1 in level1_dirs:
        for l2 in level2_dirs:
            for l3 in level3_dirs:
                # Construct the specific paths
                predictions_dir = os.path.join(base_prediction_dir, l1, l2, l3)
                savedir = os.path.join(savedir_base, l1, l2, l3)
                dataset = l2.replace('predict_', '')
                base_groundtruth_dir_dataset = base_groundtruth_dir.format(**{"dataset": dataset})
                
                # Ensure the savedir exists
                os.makedirs(savedir, exist_ok=True)

                # Construct the command to run
                cmd = [
                    'python', 'evaluate_CMBlevel.py',
                    '--savedir', savedir,
                    '--groundtruth_dir', base_groundtruth_dir_dataset,
                    '--predictions_dir', predictions_dir,
                    '--num_workers', '1',
                    '--cmb_metadata_csv', metadata_csv_path,
                    '--synthseg_dir', synthseg_dir,
                    # '--overwrite',
                    '--create_plots'
                ]
                yield cmd

if __name__ == '__main__':
    
    NUM_WORKERS = 5

    # PArse num workser arparse
    num_workers = min(NUM_WORKERS, os.cpu_count())
    
    # Setup a pool of workers equal to the number of available CPU cores
    with Pool(processes=num_workers) as pool:
        pool.map(run_evaluation, generate_commands())

    print("All evaluations completed.")
