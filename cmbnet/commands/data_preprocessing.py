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

# Define a lock for thread synchronization
csv_lock = threading.Lock()

# Utils
import cmbnet.preprocessing.loading as loading
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
###################            PROCESSING STEPS            ###################
##############################################################################

def resample(source_image, target_image, interpolation, is_annotation=False,
            isotropic=False, source_sequence=None, target_sequence=None, msg='', 
            log_level="\t\t"):
    '''
    Resamples source image to target image (no registration is performed).
    Args:
        source_image (nib.Nifti1Image): Source image being resampled.
        target_image (nib.Nifti1Image): Target image to which source is being resampled to.
        interpolation (str): Resampling method (one of nearest, linear and continuous).
        is_annotation (bool): Whether the source image is an annotation.
        isotropic (bool): Whether to resample to isotropic (uses only source image).
        source_sequence (str)(optional): Source sequence (for logging purposes only).
        target_sequence (str)(optional): Target sequence (for logging purposes only).
        msg (str)(optional): Log message.
    Returns:
        resampled_image (nib.Nifti1Image): Resampled source image.
        msg (str): Log message.
    '''
    if isotropic:
        msg += f'{log_level}Resampling {source_sequence} MRI to isotropic of voxel size {args.voxel_size} using {interpolation} interpolation...\n'
        msg += f'{log_level}\tShape before resampling: {source_image.shape}\n'

        desired_voxel_size = float(args.voxel_size)
        isotropic_affine = np.diag([desired_voxel_size, desired_voxel_size, desired_voxel_size])
        resampled_image = resample_img(source_image, target_affine=isotropic_affine,
                                        interpolation=interpolation,
                                        fill_value=np.min(source_image.get_fdata()),
                                        order='F')

    elif is_annotation:
        msg += f'{log_level}Resampling {source_sequence} annotation to {target_sequence} using {interpolation} interpolation...\n'
        msg += f'{log_level}\tShape before resampling: {source_image.shape}\n'

        if interpolation == 'nearest':
            resampled_image = resample_to_img(source_image, target_image,
                                                interpolation=interpolation,
                                                fill_value=0)

        elif interpolation == 'linear':
            resampled_image = np.zeros(target_image.shape, dtype=np.float32)

            unique_labels = np.rint(np.unique(source_image.get_fdata()))
            for unique_label in unique_labels:

                annotation_binary = nib.Nifti1Image(
                    (np.rint(source_image.get_fdata()) == unique_label).astype(np.float32),
                    source_image.affine, source_image.header)

                annotation_binary = resample_to_img(annotation_binary, target_image,
                                                    interpolation=interpolation, fill_value=0)

                resampled_image[annotation_binary.get_fdata() >= 0.5] = unique_label

            resampled_image = nib.Nifti1Image(resampled_image, affine=annotation_binary.affine,
                                                header=annotation_binary.header)

    else:
        msg += f'{log_level}Resampling {source_sequence} MRI to {target_sequence} using {interpolation} interpolation...\n'
        msg += f'{log_level}\tShape before resampling: {source_image.shape}\n'

        resampled_image = resample_to_img(source_image, target_image, interpolation=interpolation,
                                            fill_value=np.min(source_image.get_fdata()))

    msg += f'{log_level}Shape after resampling: {resampled_image.shape}\n'

    return resampled_image, msg

def resample_mris_and_annotations(mris, annotations, primary_sequence, isotropic, msg=''):
    '''
    Resamples MRIs and annotations to primary sequence space.
    Args:
        mris (dict): Dictionary of MRIs.
        annotations (dict): Dictionary of annotations.
        primary_sequence (str): Sequence to which other sequences are being resampled to.
        isotropic (bool): Whether to resample to isotropic (uses only source image).
        msg (str)(optional): Log message.
    Returns:
        mris (dict): Dictionary of resampled MRIs.
        annotations (dict): Dictionary of resampled annotations.
        msg (str): Log message.
    '''
    msg += '\tResampling MRIs and annotations maps...\n'

    start = time.time()

    if isotropic:
        mris[primary_sequence], msg = resample(source_image=mris[primary_sequence],
                                                target_image=None,
                                                interpolation='linear',
                                                isotropic=True,
                                                source_sequence=primary_sequence,
                                                target_sequence=primary_sequence, msg=msg)
    for sequence in mris:
        # resample MRI
        if sequence != primary_sequence:
            mris[sequence], msg = resample(source_image=mris[sequence],
                                            target_image=mris[primary_sequence],
                                            interpolation='continuous',
                                            source_sequence=sequence,
                                            target_sequence=primary_sequence, msg=msg)
        # resample annotation
        annotations[sequence], msg = resample(source_image=annotations[sequence],
                                                target_image=mris[primary_sequence],
                                                interpolation='nearest', # bcs binary mask
                                                # interpolation='linear',
                                                is_annotation=True,
                                                source_sequence=sequence,
                                                target_sequence=primary_sequence, msg=msg)

    end = time.time()
    msg += f'\t\tResampling of MRIs and annotations took {end - start} seconds!\n\n'

    return mris, annotations, msg


def crop_and_concatenate(mris, annotations, primary_sequence, save_sequence_order, msg=''):
    '''
    Crops and concatenates MRIs and annotations to non-zero region.
    Args:
        mris (nib.Nifti1Image): Input MRIs.
        annotations (nib.Nifti1Image): Input annotations.
        primary_sequence (str): Sequence to which other sequences are being resampled to.
        save_sequence_order ([str]): Save sequence order.
        msg (str)(optional): Log message.
    Returns:
        cropped_mris (np.ndarray): Cropped MRIs array.
        cropped_annotations (np.ndarray): Cropped annotations array.
        msg (str): Log message.
    '''
    msg += '\tCropping and concatenating MRIs and annotations...\n'

    start = time.time()

    # get brain mask from primary sequence
    mask = process_masks.get_brain_mask(image=mris[primary_sequence])

    x, y, z = np.where(mask == 1)
    coordinates = {'x': [np.min(x), np.max(x)], 'y': [np.min(y), np.max(y)],
                    'z': [np.min(z), np.max(z)]}

    # concatenate MRIs and annotations
    mris_array, annotations_array = [], []

    for sequence in save_sequence_order:
        mris_array.append(mris[sequence].get_fdata()[..., None])
        annotations_array.append(annotations[sequence].get_fdata()[..., None])

    mris_array = np.concatenate(mris_array, axis=-1)
    annotations_array = np.concatenate(annotations_array, axis=-1)

    msg += f'\t\tMRIs shape after concatenation: {mris_array.shape}\n'
    msg += f'\t\tAnnotations shape after concatenation: {annotations_array.shape}\n'

    # crop MRIs and annotations by applying brain mask
    cropped_mris = mris_array[coordinates['x'][0]:coordinates['x'][1],
                                coordinates['y'][0]:coordinates['y'][1],
                                coordinates['z'][0]:coordinates['z'][1], :]

    cropped_annotations = annotations_array[coordinates['x'][0]:coordinates['x'][1],
                                            coordinates['y'][0]:coordinates['y'][1],
                                            coordinates['z'][0]:coordinates['z'][1], :]

    msg += f'\t\tMRIs shape after cropping: {cropped_mris.shape}\n'
    msg += f'\t\tAnnotations shape after cropping: {cropped_annotations.shape}\n'

    end = time.time()
    msg += f'\t\tCropping and concatenation of MRIs and annotations took {end - start} seconds!\n\n'

    return cropped_mris, cropped_annotations, msg

def combine_annotations(annotations, priorities, msg=''):
    '''
    Combines multi-channel annotations to single-channel according to label priotiries.
    Args:
        annotations (np.array): Annotations array.
        priorities ([int]): Label priorities.
        msg (str)(optional): Log message.
    Returns:
        combined_annotations (np.array): Combined annotations array.
        msg (optional): Log message.
    '''
    
    # TODO: if with future datasets several labels, combine here. 
    
    # For now let's just take first channel (T2S)
    combined_annotations = annotations[:, :, :, 0]
    
    return combined_annotations, msg

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
    df = pd.DataFrame(data)
    
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

    # Load MRI and annotations with quality control
    mris, annotations, labels_metadata, im_specs_orig, prim_seq, msg = loading.load_mris_and_annotations(
        args, subject, msg, log_level="\t"
    )
    msg += f'\tUsing {prim_seq} as primary sequence\n'

    # Resample and standardize MRI data
    mris_resampled, annotations_resampled, msg = resample_mris_and_annotations(
        mris, annotations, primary_sequence=prim_seq, isotropic=True, msg=msg
    )

    # Save affine and header after resampling for later use
    affine_after_resampling = mris_resampled[prim_seq].affine
    header_after_resampling = mris_resampled[prim_seq].header

    # Crop and concatenate sequences
    save_seq_order = [prim_seq] + [seq for seq in mris_resampled if seq != prim_seq]
    msg += f'\tConcatenating MRIs in the following order: {save_seq_order}\n'
    mris_array, annotations_array, msg = crop_and_concatenate(
        mris_resampled, annotations_resampled, primary_sequence=prim_seq, save_sequence_order=save_seq_order, msg=msg
    )

    # Squeeze arrays to remove any unnecessary fourth dimension
    mris_array = mris_array.squeeze(axis=-1)
    annotations_array = annotations_array.squeeze(axis=-1)
    msg += f'\tSqueezed MRIs and Annotations (keeping {save_seq_order[0]})\n'
    msg += f'\t\tMRIs shape after cropping: {mris_array.shape}\n'
    msg += f'\t\tAnnotations shape after cropping: {annotations_array.shape}\n'

    # Convert processed data to Nifti1Image format for saving
    mris_image = nib.Nifti1Image(mris_array.astype(np.float32), affine_after_resampling, header_after_resampling)
    annotations_image_pre = nib.Nifti1Image(annotations_array.astype(np.uint8), affine_after_resampling, header_after_resampling)

    # Clean CMB masks and generate plots
    msg += "\tCleaning final masks and checking new stats for annotations after transforms\n"
    processed_mask, metadata, msg = process_masks.process_cmb_mask(annotations_image_pre, msg)
    annotations_metadata_new = {prim_seq: metadata}
    annotations_image = nib.Nifti1Image(processed_mask.get_fdata().astype(np.uint8), affine_after_resampling, header_after_resampling)

    # Save processed images to disk
    nib.save(mris_image, os.path.join(args.data_dir_path, subject, args.mris_subdir, f'{subject}.nii.gz'))
    nib.save(annotations_image, os.path.join(args.data_dir_path, subject, args.annotations_subdir, f'{subject}.nii.gz'))

    # Handle and save metadata
    metadata_out = {
        "subject": subject,
        "seq_type": prim_seq,
        **labels_metadata[prim_seq],
        "n_CMB_old": len(labels_metadata[prim_seq]["CMBs_old"]),
        "CMBs_new": annotations_metadata_new[prim_seq],
        "n_CMB_new": len(annotations_metadata_new[prim_seq]),
        "old_specs": im_specs_orig,
        "new_specs": loading.extract_im_specs(mris_image)
    }
    metadata_filepath = os.path.join(args.data_dir_path, subject, args.annotations_metadata_subdir, f'{subject}_metadata.json')
    with open(metadata_filepath, "w") as file:
        json.dump(metadata_out, file, default=loading.convert_numpy, indent=4)
    msg += "\tCorrectly saved NIfTI images and metadata for study\n"

    # Generate and save CMB plots for debugging
    mask_with_CMS = np.zeros_like(annotations_image.get_fdata())
    for k_i, cm_i in annotations_metadata_new[prim_seq].items():
        cm = tuple(cm_i['CM'])
        mask_with_CMS[cm] = 1  # Mark the center of mass in the mask
    mask_with_CMS_im= nib.Nifti1Image(mask_with_CMS.astype(np.uint8), affine_after_resampling, header_after_resampling)

    utils_plt.generate_cmb_plots(subject,
                                mri_im=mris_image,
                                raw_cmb=mask_with_CMS_im,
                                processed_cmb=annotations_image,
                                cmb_metadata=metadata_out['CMBs_new'],
                                plots_path=utils_general.ensure_directory_exists(os.path.join(args.plots_path, "post")))

    msg += "\tCorrectly generated and saved CMB plots for study\n"
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

    # Initialize log files
    current_time = datetime.now()
    current_datetime = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    args.log_file_path = os.path.join(args.output_dir, f'log_{current_datetime}.txt')
    args.csv_log_filepath = os.path.join(args.output_dir, f'log_{current_datetime}.csv')
    
    for dir_p in [args.output_dir, args.data_dir_path, args.plots_path]:
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
    if args.remove_studies:
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
        df_log = pd.read_csv(args.start_from_log, sep=";")
        df_log_fail = df_log[df_log['status'] == 'failed']
        subjects = df_log_fail['studyUID'].to_list()
        msg += f"Collected a total of {len(subjects)} subjects out of {df_log.shape[0]} from log file {args.start_from_log}\n"


    # Overwrite with give studies
    if args.studies is not None:
        if len(args.studies) == 1 and args.studies[0].endswith(".csv"):
            df_filter = pd.read_csv(args.studies[0])
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
        df_results = pd.read_csv(args.csv_log_filepath, sep=";")
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
                        choices=['valdo', 'cerebriu', 'momeni', 'momeni-synth', 'dou', 'rodeja'], 
                        help='Raw dataset name, to know what type of preprocessing is needed')
    parser.add_argument('--studies',  nargs='+', type=str, default=None, required=False,
                        help='Specific studies to process. If None, all processed')
    parser.add_argument('--remove_studies',  type=str, nargs='+',  default=None, required=False,
                        help='Full path to CSV with studyUID of studies to remove from processed data. If given, only this is done')
    parser.add_argument('--start_from_log', type=str, default=None, required=False,
                        help='Full path to the CSV log file where to rerun for failed cases')
    parser.add_argument('--progress_bar', type=bool, default=True,
                        help='Whether or not to show a progress bar')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
    
