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
- Registration?
- Implement for 3 extra datasets:
    * CRBR
    * 20-SWI
    * 72-SWI


@author: jorgedelpozolerida
@date: 25/01/2024
"""
import os
import argparse
import traceback


import logging                                                                      
import numpy as np                                                                  
import pandas as pd                                                                 
from tqdm import tqdm
import nibabel as nib
import multiprocessing
import time 
from nilearn.image import resample_to_img, resample_img
from scipy.ndimage import generate_binary_structure, binary_closing, binary_dilation
from scipy.ndimage import label as nd_label
import json
from skimage.measure import label
from skimage.filters import threshold_otsu
from datetime import datetime
from functools import partial
import sys

# Utils
import utils.utils_datasets as utils_datasets
import utils.utils_processing as utils_process

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

# Get the logger for the package
logger_nib = logging.getLogger('nibabel')

# Set the log level to CRITICAL to deactivate normal logging
logger_nib.setLevel(logging.CRITICAL)

def ensure_directory_exists(dir_path):
    """ Create directory if non-existent """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def write_to_log_file(msg, log_file_path):
    '''
    Writes message to the log file.
    Args:
        msg (str): Message to be written to log file.
        log_file_path (str): Path to log file.
    '''
    current_time = datetime.now()
    with open(log_file_path, 'a+') as f:
        f.write(f'\n{current_time}\n{msg}')

def check_for_duplicates(lst):
    seen = set()
    for element in lst:
        if element in seen:
            print(f"Duplicate found: {element}")
            raise ValueError(f"Duplicate element found: {element}")
        seen.add(element)

def get_dataset_subjects(args):
    """ 
    Returns studies fomr dataset making sure some QCs
    """
    if args.dataset_name == "VALDO":
        assert "VALDO" in args.input_dir
        subjects = [d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]
    elif args.dataset_name == "cerebriu":
        assert "CEREBRIU" in args.input_dir
        subjects1 = os.listdir(os.path.join(args.input_dir))
        subjects2 = os.listdir(os.path.join(args.input_dir))
        if set(subjects1) == set(subjects2):
            subjects = subjects1
        else:
            raise ValueError(f"Not all subjects contain annotations, check data")
    elif args.dataset_name == "momeni":
        assert "momeni" in args.input_dir
        raise NotImplementedError
    elif args.dataset_name == "momeni-synth":
        assert "momeni" in args.input_dir
        raise NotImplementedError
    elif args.dataset_name == "dou":
        assert "cmb-3dcnn-data" in args.input_dir
        raise NotImplementedError
    else:
        raise NotImplemented
    
    check_for_duplicates(subjects)

    return subjects
    

def get_largest_cc(segmentation):
    """
    Gets the largest connected component in image.
    Args:
        segmentation (np.ndarray): Image with blobs.
    Returns:
        largest_cc (np.ndarray): A binary image containing nothing but the largest
                                    connected component.
    """
    labels = label(segmentation)
    bincount = np.array(np.bincount(labels.flat))
    ind_large = np.argmax(bincount)  # Background is initially largest
    bincount[ind_large] = 0  # Remove background
    ind_large = np.argmax(bincount)  # This should now be largest connected component
    largest_cc = labels == ind_large

    return np.double(largest_cc)


def load_mris_and_annotations(args, subject, msg=''):
    '''
    Loads MRI scans and their corresponding annotations for a given subject 
    from a specific dataset and performs orientation fix.    
    
    Args:
        args (object): Contains configuration parameters, including input directory and dataset name.
        subject (str): Identifier of the subject whose MRI scans and annotations are to be loaded.
        msg (str, optional): A string for logging purposes. Default is an empty string.
        
    Returns:
        mris (dict): Dictionary where keys are sequence names (e.g., "T1", "T2") and values are 
                        the corresponding MRI scans loaded as nibabel.Nifti1Image objects.
        annotations (dict): Dictionary where keys are sequence names and values are the corresponding 
                            annotations loaded as nibabel.Nifti1Image objects.
        labels_metadata (dict): Dictionary containing metadata related to labels (annotations).
        prim_seq (str): Primry sequence used for this subject
        msg (str): Updated log message.

    '''
    msg += '\tLoading MRI scans and annotations...\n'


    if args.dataset_name == "VALDO":
        sequences_raw, labels_raw, labels_metadata,  msg = utils_datasets.load_VALDO_data(args, subject, msg)
    if args.dataset_name == "cerebriu":
        sequences_raw, labels_raw, labels_metadata,  msg = utils_datasets.load_CEREBRIU_data(args, subject, msg)
    else:
        # Implement here for other datasets
        raise NotImplementedError
    
    start = time.time()

    mris = {}
    annotations = {}

    # Fill MRIs dict
    for sequence_name in sequences_raw:
        mris[sequence_name] = sequences_raw[sequence_name]
        msg += f'\t\tFound {sequence_name} MRI sequence of shape {mris[sequence_name].shape}\n'

        # fix orientation and data type
        mris[sequence_name] = nib.as_closest_canonical(mris[sequence_name])
        mris[sequence_name].set_data_dtype(np.float32) 


    # Handle primary sequence
    options = args.primary_sequence.split('|')
    if len(options) == 1:
        prim_seq = options[0] if options[0] in mris.keys() else next(iter(mris.keys()), None)
    else:
        prim_seq = next((option for option in options if option in mris.keys()), next(iter(mris.keys()), None))

    # Fill annotations dict
    for sequence_name in sequences_raw:
        if sequence_name in labels_raw.keys():
            annotations[sequence_name] = labels_raw[sequence_name]
            msg += f'\t\tFound {sequence_name} annotation of shape {annotations[sequence_name].shape}\n'
        else:
            annotations[sequence_name] = nib.Nifti1Image(np.zeros(shape=mris[prim_seq].shape),
                                                    affine=mris[prim_seq].affine,
                                                    header=mris[prim_seq].header)
            msg += f'\t\tMissing {sequence_name} annotation, filling with 0s\n'

        # fix orientation adn data type
        annotations[sequence_name] = nib.as_closest_canonical(annotations[sequence_name])
        annotations[sequence_name].set_data_dtype(np.uint8)


    end = time.time()
    msg += f'\t\tLoading of MRIs and annotations took {end - start} seconds!\n\n'

    return mris, annotations, labels_metadata, prim_seq, msg

def resample(source_image, target_image, interpolation, is_annotation=False,
            isotropic=False, source_sequence=None, target_sequence=None, msg=''):
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
        msg += f'\tResampling {source_sequence} MRI to isotropic of voxel size {args.voxel_size} using {interpolation} interpolation...\n'
        msg += f'\t\tShape before resampling: {source_image.shape}\n'

        desired_voxel_size = float(args.voxel_size)
        isotropic_affine = np.diag([desired_voxel_size, desired_voxel_size, desired_voxel_size])
        resampled_image = resample_img(source_image, target_affine=isotropic_affine,
                                        interpolation=interpolation,
                                        fill_value=np.min(source_image.get_fdata()),
                                        order='F')

    elif is_annotation:
        msg += f'\tResampling {source_sequence} annotation to {target_sequence} using {interpolation} interpolation...\n'
        msg += f'\t\tShape before resampling: {source_image.shape}\n'

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
        msg += f'\tResampling {source_sequence} MRI to {target_sequence} using {interpolation} interpolation...\n'
        msg += f'\t\tShape before resampling: {source_image.shape}\n'

        resampled_image = resample_to_img(source_image, target_image, interpolation=interpolation,
                                            fill_value=np.min(source_image.get_fdata()))

    msg += f'\t\tShape after resampling: {resampled_image.shape}\n'

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

def get_brain_mask(image):
    """
    Computes brain mask using Otsu's thresholding and morphological operations.
    Args:
        image (nib.Nifti1Image): Primary sequence image.
    Returns:
        mask (np.ndarray): Computed brain mask.
    """
    # TODO: investigate if this a good fit for brain mask in all cases. Play around
    image_data = image.get_fdata()
    
    # Otsu's thresholding
    threshold = threshold_otsu(image_data)
    mask = image_data > threshold

    # Apply morphological operations
    struct = generate_binary_structure(3, 2)  # this defines the connectivity
    mask = binary_closing(mask, structure=struct)
    mask = get_largest_cc(mask)
    mask = binary_dilation(mask, iterations=5, structure=struct)

    return mask


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
    mask = get_brain_mask(image=mris[primary_sequence])

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

def process_study(args, subject, msg=''):
    """
    Process a given study (subject) by performing a series of operations 
    including loading, resampling, cropping, and saving the MRIs and annotations.
    
    Args:
        args (dict): Parsed arguments coming from parse_args() function.
        subject (str): The subject identifier.
        msg (str, optional): Log message. Defaults to ''.
    
    Returns:
        None. The function writes processed data to disk and updates the log message.
    """
    
    # Initialize
    start = time.time()
    msg = f'Started processing {subject}...\n\n'
    
    # Create dirs
    for sub_d in [args.mris_subdir, args.annotations_subdir, args.annotations_metadata_subdir]:
        ensure_directory_exists(os.path.join(args.data_dir_path, subject, sub_d))
    
    
    try:

        # 1. Perform QC while loading data
        mris, annotations, labels_metadata, prim_seq, msg = load_mris_and_annotations(args, subject, msg)
        msg += f'\tUsing {prim_seq} as primary sequence\n'

        # 2. Resample and Standardize
        mris, annotations, msg = resample_mris_and_annotations(mris, annotations, 
                                                                primary_sequence=prim_seq, 
                                                                isotropic=True, 
                                                                msg=msg)
        
        # Save affine after resampling
        affine_after_resampling = mris[prim_seq].affine
        header_after_resampling = mris[prim_seq].header
        
        # 3. Crop and Concatenate
        save_seq_order = [prim_seq] + [seq for seq in mris.keys() if seq != prim_seq]
        msg += f'\tCocatenating MRIs in the following order: {save_seq_order}\n'

        mris_array, annotations_array, msg = crop_and_concatenate(
            mris, annotations, primary_sequence=prim_seq, 
            save_sequence_order=save_seq_order, msg=msg)
        
        # 4. Combine annotations
        annotations_array, msg = combine_annotations(annotations_array, None, msg)
        
        # Convert to Nifti1Image
        mris_image = nib.Nifti1Image(mris_array.astype(np.float32), affine_after_resampling, header_after_resampling)
        annotations_image = nib.Nifti1Image(annotations_array.astype(np.uint8), affine_after_resampling, header_after_resampling)
        
        # Check Annotations Stats
        msg += "\tChecking new stats for annotations after transforms\n"
        _, metadata, msg = utils_process.process_cmb_mask(annotations_image, msg)
        annotations_metadata_new = {prim_seq: metadata}


        # Save to Disk
        nib.save(mris_image, os.path.join(args.data_dir_path, subject, args.mris_subdir, subject + '.nii.gz'))
        nib.save(annotations_image, os.path.join(args.data_dir_path, subject, args.annotations_subdir, subject + '.nii.gz'))
        
        # Convert numpy arrays to lists
        labels_metadata_listed = numpy_to_list(labels_metadata)
        annotations_metadata_new_listed = numpy_to_list(annotations_metadata_new)

        # Save Metadata for CMBs using JSON format
        with open(os.path.join(args.data_dir_path, subject, args.annotations_metadata_subdir, f'{subject}_raw.json'), "w") as file:
            json.dump(labels_metadata_listed, file, indent=4)
        with open(os.path.join(args.data_dir_path, subject, args.annotations_metadata_subdir, f'{subject}_processed.json'), "w") as file:
            json.dump(annotations_metadata_new_listed, file, indent=4)
    
    except Exception:
            
        msg += f'Failed to process {subject}!\n\nException caught: {traceback.format_exc()}'
    
    # Finalize
    end = time.time()
    msg += f'Finished processing of {subject} in {end - start} seconds!\n\n'
    write_to_log_file(msg, args.log_file_path)


def main(args):

    args.data_dir_path = os.path.join(args.output_dir, 'Data')
    args.mris_subdir = 'MRIs'
    args.annotations_subdir = 'Annotations'
    args.annotations_metadata_subdir = 'Annotations_metadata'
    
    for dir_p in [args.output_dir, args.output_dir, args.data_dir_path]:
        ensure_directory_exists(dir_p)

    current_time = datetime.now()
    current_datetime = current_time.strftime("%d%m%Y_%H%M%S")
    args.log_file_path = os.path.join(args.output_dir, f'log_{current_datetime}.txt')

    # Get subject list
    subjects = get_dataset_subjects(args)

    # Determine number of worker processes
    available_cpu_count = multiprocessing.cpu_count()
    num_workers = min(args.num_workers, available_cpu_count)
    
    # Parallelizing using multiprocessing
    with multiprocessing.Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap(partial(process_study, args), subjects), total=len(subjects)))

def parse_args():
    '''
    Parses all script arguments.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--primary_sequence', type=str, default='T2S|SWI',
                        help='Primary sequence (to which the rest will be conformed to). Default T2S.')
    parser.add_argument('--voxel_size', type=float, default=0.5,
                        help='Voxel size of isotropic space. default 0.5')
    parser.add_argument('--input_dir', type=str, default=None, required=True,
                        help='Path to the input directory of dataset')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Full path to the directory where processed files will be saved')
    parser.add_argument('--num_workers', type=int, default=5,
                            help='Number of workers running in parallel')
    parser.add_argument('--dataset_name', type=str, default=None, required=True, choices=['VALDO', 'cerebriu','momeni', 'momeni-synth', 'dou'], 
                        help='Raw dataset name, to know what preprocessign to do')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
    
