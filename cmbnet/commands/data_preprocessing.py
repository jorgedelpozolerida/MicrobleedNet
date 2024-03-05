#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Script to pre-process datasets

Steps and order followed to process data:

# 1. Load dataset data while:
    - Cleaning + QC
    - Region growing if necessary
# 2. Resample and Standardize
# 3. Crop (using brain mask)
# 4. Concatenate (stack together into single file)

Datasets implemented:
- VALDO challenge

NOTE:
- Can work in paralell using many CPUs as specified
- Also retrieves and saves metadata for masks before and after processing

TODO:
- Save metadata nicely in the process
- Save plots in the process to ensure correct processing
- Registration?
- Implement for 3 extra datasets:
    - DOU
    - CEREBRIU
    - MONEMI



@author: jorgedelpozolerida
@date: 25/01/2024
"""
import os
import argparse
import traceback


import logging                                                                      
import numpy as np                                                                  
from tqdm import tqdm
import nibabel as nib
import multiprocessing as mp
import time 
from nilearn.image import resample_to_img, resample_img
import json
from datetime import datetime
from functools import partial
import sys
import pandas as pd
import shutil
import threading
import csv
import subprocess

# Define a lock for thread synchronization
csv_lock = threading.Lock()

# Utils
import cmbnet.preprocessing.loading as loading
import cmbnet.preprocessing.processing_steps as process_steps
import cmbnet.preprocessing.process_masks as process_masks
import cmbnet.utils.utils_general as utils_general
import cmbnet.visualization.utils_plotting as utils_plt

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

# Get the logger for the package
logger_nib = logging.getLogger('nibabel')

# Set the log level to CRITICAL to deactivate normal logging
logger_nib.setLevel(logging.CRITICAL)



##############################################################################
###################                 OTHERS                 ###################
##############################################################################

def numpy_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: numpy_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_list(item) for item in obj]
    else:
        return obj
    
def update_study_status(study_uid, new_status, msg, csv_log_filepath):
    '''
    Updates the status of a study in the CSV log file and adds a message.

    Args:
        study_uid (str): The unique identifier of the study.
        new_status (str): The new status ('completed' or 'failed').
        msg (str): The message associated with the status update.
        csv_log_filepath (str): Path to the CSV log file.
    '''
    # Prepare data to be appended to the CSV
    data = {'studyUID': [study_uid], 'status': [new_status], 'message': [msg]}
    df = pd.DataFrame(data, dtype=str)
    
    # Append data to the CSV file
    df.to_csv(csv_log_filepath, mode='a', header=False, index=False, sep=';', quoting=1, quotechar='"')


def delete_study_files(args, study):
    """ 
    Deletes files for study if failed
    """
    # Define paths to directories and files
    study_dir = os.path.join(args.data_dir_path, study)
    pre_plots_dir = os.path.join(args.plots_path, "pre")
    post_plots_dir = os.path.join(args.plots_path, "post")

    # Delete study folder and its contents recursively
    if os.path.exists(study_dir):
        try:
            shutil.rmtree(study_dir)
        except OSError as e:
            print(f"Error deleting study directory {study_dir}: {e}")

    # Delete pre and post plots
    for plots_dir in [pre_plots_dir, post_plots_dir]:
        if os.path.exists(plots_dir):
            for filename in os.listdir(plots_dir):
                if filename.startswith(f"{study}-CMB-") and filename.endswith(".png"):
                    file_path = os.path.join(plots_dir, filename)
                    try:
                        os.remove(file_path)
                    except OSError as e:
                        print(f"Error deleting file {file_path}: {e}")
                        # Log error if deletion fails


##############################################################################
###################                 MAIN                  ###################
##############################################################################

def process_study(args, subject, msg=''):
    """
    Process a given study (subject) by performing a series of operations including loading,
    resampling, cropping, and saving the MRIs and annotations.
    
    Args:
        args (dict): Parsed arguments coming from the parse_args() function.
        subject (str): The subject identifier.
        msg (str, optional): Log message. Defaults to ''.
    
    Returns:
        str: Updated log message after completing processing.
    """
    # Ensure necessary directories exist
    for sub_d in [args.mris_subdir, args.annotations_subdir, args.annotations_metadata_subdir]:
        dir_path = os.path.join(args.data_dir_path, subject, sub_d)
        utils_general.ensure_directory_exists(dir_path)

    # Load MRI scans and annotations with QC
    mris_raw, annotations_raw, nifti_paths, labels_metadata, im_specs_orig, prim_seq, msg = loading.load_mris_and_annotations(
        args, subject, msg, log_level="\t"
        )
    msg += f'\tUsing {prim_seq} as primary sequence\n'

    # Skull stripping
    mris_noskull, brain_masks, msg = process_steps.skull_strip(args.synthstrip_docker, subject, mris_raw,
                                                    utils_general.ensure_directory_exists(
                                                        os.path.join(args.cache_folder, "synthstrip")
                                                        ),
                                                    msg)
    
    # Crop images
    mris_cropped, annotations_cropped =  process_steps.crop_mris_and_annotations(
        mris_noskull, annotations_raw, brain_masks, prim_seq, msg, log_level="\t"
    )

    # Fix orientation and dtype
    mris_fixed, annotations_fixed, msg = process_steps.fix_orientation_and_dtype(mris_cropped, annotations_cropped, prim_seq, msg, log_level="\t")

    # Resample and standardize MRI data
    mris_resampled, annotations_resampled, msg = process_steps.resample_mris_and_annotations(
        args, mris_fixed, annotations_fixed, primary_sequence=prim_seq, isotropic=True, msg=msg
    )
    affine_after_resampling = mris_resampled[prim_seq].affine
    header_after_resampling = mris_resampled[prim_seq].header

    # Prune CMBs
    annotations_pruned, labels_metadata, msg = process_masks.prune_CMBs(args,
                                                                        annotations_raw, 
                                                                        annotations_resampled, 
                                                                        labels_metadata, 
                                                                        prim_seq,  
                                                                        msg,
                                                                        log_level="\t")

    # Clean CMB masks and generate plots
    msg += "\tCleaning final masks and checking new stats for annotations after transforms\n"
    processed_mask_nib, metadata, msg = process_masks.process_cmb_mask(annotations_pruned[prim_seq], msg, args.dataset_name)
    annotations_metadata_new = {prim_seq: metadata}
    
    final_anno_nib = processed_mask_nib
    final_mri_nib = mris_resampled[prim_seq]

    # Save processed images to disk
    nib.save(final_mri_nib, os.path.join(args.data_dir_path, subject, args.mris_subdir, f'{subject}.nii.gz'))
    nib.save(final_anno_nib, os.path.join(args.data_dir_path, subject, args.annotations_subdir, f'{subject}.nii.gz'))

    # Handle and save metadata
    metadata_out = {
        "subject": subject,
        "seq_type": prim_seq,
        **labels_metadata[prim_seq],
        "n_CMB_old": len(labels_metadata[prim_seq]["CMBs_old"]),
        "CMBs_new": annotations_metadata_new[prim_seq],
        "n_CMB_new": len(annotations_metadata_new[prim_seq]),
        "old_specs": im_specs_orig,
        "new_specs": loading.extract_im_specs(final_mri_nib)
    }
    metadata_filepath = os.path.join(args.data_dir_path, subject, args.annotations_metadata_subdir, f'{subject}_metadata.json')
    with open(metadata_filepath, "w") as file:
        json.dump(metadata_out, file, default=loading.convert_numpy, indent=4)
    msg += "\tCorrectly saved NIfTI images and metadata for study\n"

    # Generate and save CMB plots for debugging
    mask_with_CMS = np.zeros_like(final_anno_nib.get_fdata())
    for k_i, cm_i in annotations_metadata_new[prim_seq].items():
        cm = tuple(cm_i['CM'])
        mask_with_CMS[cm] = 1  # Mark the center of mass in the mask
    mask_with_CMS_im= nib.Nifti1Image(mask_with_CMS.astype(np.uint8), affine_after_resampling, header_after_resampling)

    utils_plt.generate_cmb_plots(subject,
                                mri_im=final_mri_nib,
                                raw_cmb=mask_with_CMS_im,
                                processed_cmb=final_anno_nib,
                                cmb_metadata=metadata_out['CMBs_new'],
                                plots_path=utils_general.ensure_directory_exists(os.path.join(args.plots_path, "post")))

    msg += "\tCorrectly generated and saved CMB plots for study\n"
    
    # CMBs num check
    n_CMB_new = int(metadata_out['n_CMB_new'])
    n_CMB_old = int(metadata_out['n_CMB_raw'])
    if n_CMB_old != n_CMB_new:
        msg += f"\t ISSUE: number of original CMBs differ before ({n_CMB_old}) and after ({n_CMB_new}) preprocessing\n"
    return msg

def process_single_study_worker(args, studies_pending: mp.Queue, studies_done: mp.Queue, processes_done: mp.Queue, worker_number: int):
    '''
    Worker function that processes a single study.
    
    Args:
        args (dict): Parsed arguments coming from the parse_args() function.
        studies_pending (mp.Queue): Queue containing studies that need to be processed.
        studies_done (mp.Queue): Queue containing studies that have been processed.
        processes_done (mp.Queue): Queue indicating when processes have finished.
        worker_number (int): Identifier for the worker.
    '''
    while not studies_pending.empty():
        # Try to extract a study from the queue
        try:
            study, i, n = studies_pending.get()
        except Exception as e:
            print(f"Worker {worker_number} - No more items to process or error: {str(e)}")
            break

        msg = f'Started processing {study}... (worker {worker_number})\n'
        
        try:
            # Attempt to process the study
            msg_process = process_study(args, study, '')
            msg += msg_process
            status = 'completed'
            msg += f'Finished processing of {study} (worker {worker_number})!\n\n'
        except Exception as e:
            # Handle exceptions during processing
            status = 'failed'
            msg += f'Failed to process {study}!\n\nException caught: \n{traceback.format_exc()}\n'

        # Log the outcome
        utils_general.write_to_log_file(msg, args.log_file_path)
        
        # Update the CSV log with the study's processing status
        update_study_status(study, status, msg, args.csv_log_filepath)
        
        # delete files if failed
        if status == 'failed':
            delete_study_files(args, study)

        # Indicate the study has been processed
        studies_done.put((study, status, msg))

    # Signal that this worker has completed its task
    processes_done.put(worker_number)

def process_all_studies(args, studies):
    '''
    Processes list of studies: performs data preprocessing for provided studies
    '''
    # initiate multiprocessing queues
    studies_pending = mp.Queue()
    studies_done = mp.Queue()
    processes_done = mp.Queue()

    # put all studies in queue
    for i, study in enumerate(studies, start=1):
        studies_pending.put((study, i, len(studies)))

    # initialize progress bar
    if args.progress_bar:
        progress_bar = tqdm(total=studies_pending.qsize())
        number_of_studies_done_so_far = 0

    # start processes
    processes = []
    if studies_pending.qsize() > 0:
        number_of_workers = min(args.num_workers, studies_pending.qsize())
        for i in range(number_of_workers):
            process = mp.Process(target=process_single_study_worker,
                                    args=(args, studies_pending, studies_done,
                                        processes_done, i))
            processes.append(process)
            process.start()

        while True:
            if args.progress_bar:
                number_of_studies_done_now = studies_done.qsize()
                difference = number_of_studies_done_now - number_of_studies_done_so_far
                if difference > 0:
                    progress_bar.update(difference)
                    number_of_studies_done_so_far = number_of_studies_done_now

            if processes_done.qsize() == number_of_workers:
                if args.progress_bar:
                    progress_bar.close()
                for process in processes:
                    process.terminate()
                break

            time.sleep(0.1)


def main(args):

    args.data_dir_path = os.path.join(args.output_dir, 'Data')
    args.mris_subdir = 'MRIs'
    args.annotations_subdir = 'Annotations'
    args.annotations_metadata_subdir = 'Annotations_metadata'
    args.plots_path = os.path.join(args.output_dir, 'plots')
    args.cache_folder = os.path.join(args.output_dir, 'tmp')

    if args.reprocess_file:
        assert args.processed_dir is not None
        assert os.path.exists(args.processed_dir)
        assert os.path.exists(args.reprocess_file)

    assert os.path.exists(args.synthstrip_docker)

    # Initialize log files
    current_time = datetime.now()
    current_datetime = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    args.log_file_path = os.path.join(args.output_dir, f'log_{current_datetime}.txt')
    args.csv_log_filepath = os.path.join(args.output_dir, f'log_{current_datetime}.csv')
    
    for dir_p in [args.output_dir, args.data_dir_path, args.plots_path, args.cache_folder]:
        utils_general.ensure_directory_exists(dir_p)

    # Create an empty CSV log file with headers
    with open(args.csv_log_filepath, 'w', newline='') as f:
        f.write("studyUID;status;message\n")  

    msg = "***********************************************\n" + \
        f"STARTING PROCESSING OF DATASET {args.dataset_name}\n" + \
        "***********************************************\n"

    # Get subject list
    subjects = loading.get_dataset_subjects(args.dataset_name, args.input_dir)
    
    
    # If args.remove_studies exclude from data processed
    if args.remove_studies and args.start_from_log is None:
        msg += f"STARTING DELETION OF STUDIES FOR DATASET {args.dataset_name}\n"
        if len(args.remove_studies) == 1 and args.remove_studies[0].endswith(".csv"):
            df_remove = pd.read_csv(args.remove_studies[0])
            remove_studies = df_remove['studyUID'].to_list()
        else:
            remove_studies = args.remove_studies
        if set(remove_studies).issubset(set(subjects)):
            utils_general.confirm_action(f"Will delete {len(remove_studies)} studies")
            for stud in remove_studies:
                delete_study_files(args, stud)
                msg += f"Succesfully deleted the following study: {stud}\n"
            utils_general.write_to_log_file(msg, args.log_file_path)
        else:
            missing_studies = set(remove_studies) - set(subjects)
            e_msg = f"ERROR: The following specified studies are not available in the dataset: {', '.join(missing_studies)}\n"
            msg += e_msg
            utils_general.write_to_log_file(msg, args.log_file_path)
            raise ValueError(e_msg)
        return
    
    msg +=  f"CSV log: {args.csv_log_filepath}\n" 

    # Overwrite with failed studies
    if args.start_from_log is not None:
        df_log = pd.read_csv(args.start_from_log, sep=";", dtype=str)
        df_log_fail = df_log[df_log['status'] == 'failed']
        unprocessed_studies = [s for s in subjects if s not in df_log['studyUID'].to_list()]
        # Combine failed and unprocessed studies
        subjects_used = df_log_fail['studyUID'].to_list() + unprocessed_studies
        msg += f"Collected a total of {len(subjects_used)} subjects (Failed: {len(df_log_fail)}, Unprocessed: {len(unprocessed_studies)}) out of {len(subjects)} from log file {args.start_from_log}\n"
        subjects = subjects_used
        
        if args.remove_studies:
            utils_general.confirm_action(f"Will delete {len(subjects)} studies if present already")
            for stud in subjects:
                delete_study_files(args, stud)
                msg += f"Succesfully deleted the following study: {stud}\n"

    # Overwrite with give studies
    if args.studies is not None:
        if len(args.studies) == 1 and args.studies[0].endswith(".csv"):
            df_filter = pd.read_csv(args.studies[0], dtype=str)
            filter_studies = df_filter['studyUID'].to_list()
        else:
            filter_studies = args.studies
        if set(filter_studies).issubset(set(subjects)):
            msg += f"Filtered studies from {len(subjects)} subjects to {len(filter_studies)}\n"
            subjects = filter_studies

        else:
            missing_studies = set(filter_studies) - set(subjects)
            e_msg = f"ERROR: The following specified studies are not available in the dataset: {', '.join(missing_studies)}"
            msg += e_msg
            utils_general.write_to_log_file(msg, args.log_file_path)
            raise ValueError(e_msg)
        
    msg += f"Processing {len(subjects)} studies\n\n"
    print(msg)
    utils_general.confirm_action()
    utils_general.write_to_log_file(msg, args.log_file_path)

    # Parallelizing using multiprocessing or not
    try:
        process_all_studies(args, subjects)
        df_results = pd.read_csv(args.csv_log_filepath, sep=";", dtype=str)
        df_results_fail = df_results[df_results['status'] == 'failed']
        final_msg = "***********************************************\n" + \
        f"FINISHED PROCESSING OF DATASET {args.dataset_name}\n" + \
        f"Succesful studies: {df_results.shape[0]}, Failed studies: {df_results_fail.shape[0]}\n" + \
        "***********************************************\n"
        utils_general.write_to_log_file(final_msg, args.log_file_path)

    except Exception:
        _logger.error('Exception caught in main: {}'.format(traceback.format_exc()))
        return 1
    return 0


def parse_args():
    '''
    Parses all script arguments.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--voxel_size', type=float, default=0.5,
                        help='Voxel size of isotropic space. default 0.5')
    parser.add_argument('--input_dir', type=str, default=None, required=True,
                        help='Path to the input directory of dataset')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Full path to the directory where processed files will be saved')
    parser.add_argument('--num_workers', type=int, default=5,
                            help='Number of workers running in parallel')
    parser.add_argument('--dataset_name', type=str, default=None, required=True, 
                        choices=['valdo', 'cerebriu', 'cerebriu-neg', 'momeni', 'momeni-synth', 'dou', 'rodeja'], 
                        help='Raw dataset name, to know what type of preprocessing is needed')
    parser.add_argument('--studies',  nargs='+', type=str, default=None, required=False,
                        help='Specific studies to process. If None, all processed')
    parser.add_argument('--remove_studies',  type=str, nargs='+',  default=None, required=False,
                        help='Full path to CSV with studyUID of studies to remove from processed data. If given, only this is done')
    parser.add_argument('--start_from_log', type=str, default=None, required=False,
                        help='Full path to the CSV log file where to rerun for failed cases')
    parser.add_argument('--progress_bar', type=bool, default=True,
                        help='Whether or not to show a progress bar')
    parser.add_argument('--reprocess_file', type=str, default=None, required=False,
                        help='Full path to the CSV with info for re-processing. If provided, the whole workflow of processing changes to REPROCESS')
    parser.add_argument('--processed_dir', type=str, default=None, required=False,
                        help='Path to the processed input directory of dataset')
    parser.add_argument('--synthstrip_docker', type=str, default="/datadrive_m2/jorge/synthstrip-docker",
                        help='Full path to docker image of Synthstrip')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
    
