#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Script with utility functions to help cleaning CMB datasets


{Long Description of Script}


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

import json
from skimage.measure import label
from skimage.filters import threshold_otsu
from datetime import datetime
from functools import partial
import glob
import sys
from typing import Tuple, Dict, List, Any

current_dir_path = os.path.dirname(os.path.abspath(__file__))
parent_dir_path = os.path.abspath(os.path.join(current_dir_path, os.pardir))
sys.path.append(parent_dir_path)

import utils.utils_processing as utils_process
import utils.utils_general as utils_general


logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


##############################################################################
###################                 GENERAL                ###################
##############################################################################

def check_for_duplicates(lst):
    seen = set()
    for element in lst:
        if element in seen:
            print(f"Duplicate found: {element}")
            raise ValueError(f"Duplicate element found: {element}")
        seen.add(element)

def get_dataset_subjects(dataset_name, input_dir):
    """ 
    Returns studies fomr dataset making sure some QCs
    """
    if dataset_name == "VALDO":
        assert "VALDO" in input_dir
        subjects = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    elif dataset_name == "cerebriu":
        assert "CEREBRIU" in input_dir
        subjects1 = os.listdir(os.path.join(input_dir))
        subjects2 = os.listdir(os.path.join(input_dir))
        if set(subjects1) == set(subjects2):
            subjects = subjects1
        else:
            raise ValueError(f"Not all subjects contain annotations, check data")
    elif dataset_name == "momeni":
        assert "momeni" in input_dir
        raise NotImplementedError
    elif dataset_name == "momeni-synth":
        assert "momeni" in input_dir
        raise NotImplementedError
    elif dataset_name == "dou":
        assert "cmb-3dcnn-data" in input_dir
        raise NotImplementedError
    else:
        raise NotImplemented
    
    check_for_duplicates(subjects)

    return subjects

def get_files_metadata_from_processed(data_dir, subjects_selected=None):
    """ 
    Args:
    
    data_dir:  .../Data/ dir of processed dataset
    subjects (optional): list of subjects to get metadata from
    
    Retrieves for all or selected subjects in a processed dataset directory all
    relevant data related to mri and annotations files among others
    """
    all_subjects = os.listdir(data_dir)
    if subjects_selected is not None:
        all_subjects = [s for s in all_subjects if s in subjects_selected]
    
    all_metadata = []
    
    for sub in all_subjects:
        metadata_dict = utils_general.read_json_to_dict(os.path.join(data_dir, sub, "Annotations_metadata", f"{sub}_raw.json"))
        metadata_dict_keys = list(metadata_dict.keys())
        
        all_metadata.append(
            {
                "id": sub,
                "anno_path": os.path.join(data_dir, sub, "Annotations", f"{sub}.nii.gz"),
                "mri_path": os.path.join(data_dir, sub, "MRIs", f"{sub}.nii.gz"),
                "seq_type": metadata_dict_keys[0],
                "raw_metadata_path": os.path.join(data_dir, sub, "Annotations_metadata", f"{sub}_raw.json"),
                "processed_metadata_path": os.path.join(data_dir, sub, "Annotations_metadata", f"{sub}_processed.json")
            }
        )
    
    
    return all_metadata


##############################################################################
###################                   VALDO                ###################
##############################################################################

def process_VALDO_mri(mri_im, msg=''):
    """
    Process a VALDO MRI image to handle NaNs by replacing them with the background value.

    Args:
    - mri_im (nibabel.Nifti1Image): The nibabel object of the MRI.

    Returns:
    - nibabel.Nifti1Image: Processed MRI as a nibabel object.
    - str: Updated log message.
    """
    
    # Extract data from nibabel object
    data = mri_im.get_fdata().copy()
    
    # Identify NaNs
    nan_mask = np.isnan(data)
    num_nans = np.sum(nan_mask)
    perc_nans = num_nans/len(data.flatten())*100
    
    if num_nans > 0:
        # Compute the background value using small patches from the edges
        edge_patches = [data[:10, :10, :5], data[-10:, :10, :5], 
                        data[:10, -10:, :5], data[-10:, -10:, :5], 
                        data[:10, :10, -5:], data[-10:, :10, -5:], 
                        data[:10, -10:, -5:], data[-10:, -10:, -5:]]
        background_value = np.nanmedian(np.concatenate(edge_patches))
        if np.isnan(background_value):
            background_value = 0
            msg += f'\t\tForced background value to 0 as region selected is full of nan\n'
        # Replace NaNs with the background value
        data[nan_mask] = background_value

        msg += f'\t\tFound {round(perc_nans, 2)}% of NaNs and replaced with background value: {background_value}\n'
    
    # Convert processed data back to Nifti1Image
    processed_mri_im = nib.Nifti1Image(data, mri_im.affine, mri_im.header)

    return processed_mri_im, msg


def perform_VALDO_QC(mris, annotations, msg):
    """
    Perform Quality Control (QC) specific to the VALDO dataset on MRI sequences and labels.

    Args:
        args (Namespace): Arguments passed to the main function.
        subject (str): The subject identifier.
        mris (dict): Dictionary of MRI sequences.
        annotations (dict): Dictionary of labels.
        msg (str): Log message.

    Returns:
        mris_qc (dict): Dictionary of QC'ed MRI sequences.
        annotations_qc (dict): Dictionary of QC'ed labels.
        annotations_metadata (dict): Metadata associated with the QC'ed labels.
        msg (str): Updated log message.
    """

    mris_qc, annotations_qc, annotations_metadata = {}, {}, {}

    # Quality Control of Labels
    for anno_sequence, anno_im in annotations.items():
        annotations_qc[anno_sequence], metadata, msg = utils_process.process_cmb_mask(anno_im, msg)
        annotations_metadata[anno_sequence] = metadata

    # Quality Control of MRI Sequences
    for mri_sequence, mri_im in mris.items():
        mris_qc[mri_sequence], msg = process_VALDO_mri(mri_im, msg)
    
    return mris_qc, annotations_qc, annotations_metadata, msg


def load_VALDO_data(args, subject, msg):
    """
    Load MRI sequences and labels specific to the VALDO dataset. PErforms QC in the process.

    Args:
        args (Namespace): Command-line arguments or other configuration.
        subject (str): The subject identifier.
        msg (str): Log message.

    Returns:
        sequences_qc (dict): Dictionary of QC'ed MRI sequences.
        labels_qc (dict): Dictionary of QC'ed labels.
        labels_metadata (dict): Metadata associated with the labels.
        msg (str): Updated log message.
    """
    subject_old_dir = os.path.join(args.input_dir, subject)

    # 1. Load Raw MRI Sequences
    sequences_raw = {
        "T1": nib.load(os.path.join(subject_old_dir, f"{subject}_space-T2S_desc-masked_T1.nii.gz")),
        "T2": nib.load(os.path.join(subject_old_dir, f"{subject}_space-T2S_desc-masked_T2.nii.gz")),
        "T2S": nib.load(os.path.join(subject_old_dir, f"{subject}_space-T2S_desc-masked_T2S.nii.gz"))
    }
    
    # 2. Load Raw Labels (Annotations are made in T2S space for VALDO dataset)
    labels_raw = {
        "T2S": nib.load(os.path.join(subject_old_dir, f"{subject}_space-T2S_CMB.nii.gz"))
    }
    
    # 3. Perform Quality Control (QC) on Loaded Data
    sequences_qc, labels_qc, labels_metadata, msg = perform_VALDO_QC(sequences_raw, labels_raw, msg)
    
    return sequences_qc, labels_qc, labels_metadata, msg




##############################################################################
###################                   CEREBRIU             ###################
##############################################################################

def load_CEREBRIU_raw(input_dir, study):
    """
    Load raw MRI and segmentation data for a given CEREBRIU study.

    Args:
        input_dir (str): Directory containing the study subfolders.
        study (str): Specific study identifier.

    Returns:
        Tuple[Dict, Dict, str, str]: Tuple containing dictionaries of raw MRI sequences and labels,
                                        sequence type, and subfolder name.

    Raises:
        ValueError: If no files or multiple files are found where only one is expected.
    """
    mri_dir = os.path.join(input_dir, study, "images")
    cmb_dir = os.path.join(input_dir, study, "segmentations")

    # Find the CMB file in segmentations folder
    cmb_files = glob.glob(os.path.join(cmb_dir, "*.nii.gz"))
    if len(cmb_files) == 0:
        raise ValueError("No CMB files found")
    elif len(cmb_files) > 1:
        raise ValueError(f"Multiple CMB files found in {cmb_dir}")

    # Get the CMB file and determine corresponding MRI subfolder
    cmb_file = cmb_files[0]
    cmb_filename = os.path.basename(cmb_file).split('.')[0]  # Filename without extension

    # Find corresponding MRI file
    mri_subfolder_path = os.path.join(mri_dir, cmb_filename)
    if not os.path.isdir(mri_subfolder_path):
        raise ValueError(f"No corresponding MRI subfolder found for {cmb_filename}")

    mri_files = glob.glob(os.path.join(mri_subfolder_path, "*.nii.gz"))
    if len(mri_files) == 0:
        raise ValueError(f"No MRI files found in {mri_subfolder_path}")
    elif len(mri_files) > 1:
        raise ValueError(f"Multiple MRI files found in {mri_subfolder_path}")

    # Load Raw MRI Sequences and Labels
    seq_type = cmb_filename.split("_")[0]
    sequences_raw = {seq_type: nib.load(mri_files[0])}
    labels_raw = {seq_type: nib.load(cmb_file)}

    return sequences_raw, labels_raw, seq_type, cmb_filename


def process_CEREBRIU_cmb(label_im: nib.Nifti1Image, 
                            labelid: int, 
                            mri_im: nib.Nifti1Image, 
                            size_threshold: int,
                            max_dist_voxels: int, 
                            msg: str, 
                            multiple: bool = False,
                            show_progress: bool = False) -> Tuple[nib.Nifti1Image, nib.Nifti1Image, List[Tuple[int]], List[Dict], str]:
    """
    Processes Cerebriu CMB data using region growing and other operations.

    Args:
        label_im (nib.Nifti1Image): NIfTI image containing label data.
        labelid (int): The label identifier to filter the mask.
        mri_im (nib.Nifti1Image): NIfTI image of the MRI scan.
        size_threshold (int): The size threshold used for region growing.
        max_dist_voxels (int): The voxel distance threshold for region growing
        msg (str): Log message string.
        multiple (bool): Flag indicating if multiple processing is required.
        show_progress (bool): Flag indicating if tqdm progress bar desired.

    Returns:
        Tuple[.., .., , List[Tuple[int]], List[Dict], str]: 
        - Processed mask as np array
        - Raw mask as np array
        - List of seeds used for region growing
        - Metadata for each CMB regarding region growing
        - Updated log message

    Raises:
        RuntimeError: If overlap is detected between individual processed masks.
    """
    mask_data = label_im.get_fdata()
    mri_data = mri_im.get_fdata()

    mask_filt = mask_data == int(labelid)

    voxel_size = label_im.header.get_zooms()  # Extract voxel size from the image
    mask_filt_single_list = utils_process.isolate_single_CMBs(mask_filt, voxel_size)

    rg_metadata = []
    seeds_calculated = []
    final_processed_mask = np.zeros_like(mask_data, dtype=bool)
    msg += f'\t\t\tNumber of CMBs found in label id {str(labelid)}: {str(len(mask_filt_single_list))}.\n'

    # Processing loop with optional progress bar
    iterator = range(len(mask_filt_single_list))
    if show_progress:
        iterator = tqdm(iterator, total=len(mask_filt_single_list), desc="Processing CMBs")

    for i in iterator:
        cmb_single_mask = mask_filt_single_list[i]
        seeds = [tuple(seed) for seed in np.array(np.where(cmb_single_mask)).T]
        seeds_calculated.extend(seeds)

        processed_mask, metadata = utils_process.region_growing_with_auto_tolerance(
            volume=mri_data,
            seeds=seeds,    
            size_threshold=size_threshold,
            max_dist_voxels=max_dist_voxels,
            tolerance_range=(0, 150, 0.5),
            connectivity=6,
            show_progress=show_progress,
            # intensity_mode="point"
            # intensity_mode="average"
        )

        # Check for overlap
        if np.any(final_processed_mask & processed_mask):
            raise RuntimeError("Overlap detected between individual processed masks")

        final_processed_mask = final_processed_mask | processed_mask
        rg_metadata.append(metadata)
        msg += f"\t\t\tProcessed CMB {i}. n_seeds={len(seeds)}, new_size={np.sum(processed_mask)}\n"

    if not multiple and len(mask_filt_single_list) > 1:
        msg += f"\t\t\tWARNING: Expected single CMBs and detected several.\n"

    return final_processed_mask, mask_filt, seeds_calculated, rg_metadata, msg

def process_cerebriu_anno(args: Any, 
                            subject: str, 
                            label_im: nib.Nifti1Image, 
                            mri_im: nib.Nifti1Image, 
                            seq_folder: str, 
                            msg: str) -> Tuple[nib.Nifti1Image, Dict, str]:
    """
    Process annotations for a CEREBRIU dataset subject.

    Args:
        args (Any): Configuration parameters including input directory.
        subject (str): Subject identifier.
        label_im (nib.Nifti1Image): Label image.
        mri_im (nib.Nifti1Image): MRI image.
        seq_folder (str): Sequence folder name.
        msg (str): Log message.

    Returns:
        Tuple[nib.Nifti1Image, Dict, str]: Processed annotation image, metadata, and log message.
    """
    tasks_dict = utils_general.read_json_to_dict(os.path.join(args.input_dir, subject, "tasks.json"))
    task_item = next((it for it in tasks_dict if it['name'] == subject), None)
    series_data = next((seq for seq in task_item['series'] if seq['name'] == seq_folder), None) if task_item else None

    if not series_data:
        raise ValueError("Series data not found for the specified sequence folder.")

    extracted_data = {
        "segmentMap": series_data.get("segmentMap", {}),
        "landmarks3d": series_data.get("landmarks3d", []),
        "sequence_meta": series_data.get("classifications", []),
        "study_meta": task_item.get("classification", [])
    }

    assert extracted_data["segmentMap"], "segmentMap is empty"

    # Thresholds ---
    size_th = utils_process.calculate_size_threshold(label_im)
    label_mask_all = np.zeros_like(label_im.get_fdata(), dtype=bool)

    max_dist_mm = 9 # mm
    voxel_size = np.mean(label_im.header.get_zooms())  # Adjust as needed
    max_dist_voxels = max_dist_mm / voxel_size
    msg += f"\t\tComputed maximum distance allowed in voxels: {max_dist_voxels}.\n"

    for labelid, mask_dict in extracted_data["segmentMap"].items():
        multiple = mask_dict['attributes'].get('Multiple', False)
        msg += f"\t\tProcessing label {labelid} with {'multiple' if multiple else 'single'} CMB annotations.\n"
        
        label_mask, raw_mask, seeds, cmb_metadata, msg = process_CEREBRIU_cmb(
            label_im, labelid, mri_im, size_th, max_dist_voxels, msg, multiple)

        # Check for overlap
        if np.any(label_mask_all & label_mask):
            raise RuntimeError("Overlap detected between different CMB annotated masks")

        label_mask_all |= label_mask

    annotation_processed_nib = nib.Nifti1Image(label_mask_all.astype(np.int16), label_im.affine, label_im.header)
    processed_mask_nib, metadata, msg = utils_process.process_cmb_mask(annotation_processed_nib, msg, log_level="\t\t")

    return processed_mask_nib, metadata, msg


def process_cerebriu_mri(args, subject, mri_im, seq_folder, msg):

    return mri_im, msg

def perform_CEREBRIU_QC(args, subject, mris, annotations, seq_folder, msg):
    """
    Perform Quality Control (QC) specific to the VALDO dataset on MRI sequences and labels.

    Args:
        args (Namespace): Arguments passed to the main function.
        subject (str): The subject identifier.
        mris (dict): Dictionary of MRI sequences.
        annotations (dict): Dictionary of labels.
        msg (str): Log message.

    Returns:
        mris_qc (dict): Dictionary of QC'ed MRI sequences.
        annotations_qc (dict): Dictionary of QC'ed labels.
        annotations_metadata (dict): Metadata associated with the QC'ed labels.
        msg (str): Updated log message.
    """

    mris_qc, annotations_qc, annotations_metadata = {}, {}, {}

    # Quality Control of MRI Sequences
    for mri_sequence, mri_im in mris.items():
        mris_qc[mri_sequence], msg = process_cerebriu_mri(args, subject, mri_im, seq_folder, msg)
    
    # Quality Control of Labels
    for anno_sequence, anno_im in annotations.items():
        annotations_qc[anno_sequence], metadata, msg = process_cerebriu_anno(args, subject, anno_im, mris_qc[anno_sequence], seq_folder, msg)
        annotations_metadata[anno_sequence] = metadata

    
    return mris_qc, annotations_qc, annotations_metadata, msg
    
    
    

    return mris_qc, annotations_qc, annotations_metadata, msg

def load_CEREBRIU_data(args, subject, msg):
    """
    Load MRI sequences and labels specific to the CEREBRIU dataset. Performs QC in the process.

    Args:
        args (Namespace): Command-line arguments or other configuration.
        subject (str): The subject identifier.
        msg (str): Log message.

    Returns:
        sequences_qc (dict): Dictionary of QC'ed MRI sequences.
        labels_qc (dict): Dictionary of QC'ed labels.
        labels_metadata (dict): Metadata associated with the labels.
        msg (str): Updated log message.
    """

    # 1. Load raw data
    sequences_raw, labels_raw, sequence_type, seq_folder = load_CEREBRIU_raw(args.input_dir, subject)
    msg += f'\t\tSequence type: {sequence_type}\n'

    # 2. Perform Quality Control (QC) and Data Cleaning
    sequences_qc, labels_qc, labels_metadata, msg = perform_CEREBRIU_QC(args, subject, sequences_raw, labels_raw, seq_folder, msg)
    
    
    
    return sequences_qc, labels_qc, labels_metadata, msg