#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Module with functions for internal CEREBRIU dataset


@author: jorgedelpozolerida
@date: 13/02/2024
"""
import os
import argparse
import traceback

import logging                                                                      
import numpy as np                                                                  
import pandas as pd                                                                 
from tqdm import tqdm
import nibabel as nib
from scipy.io import loadmat
import glob
import sys
from typing import Tuple, Dict, List, Any

import cmbnet.preprocessing.process_masks as process_masks
import cmbnet.utils.utils_general as utils_general


logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

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
                            show_progress: bool = False,
                            log_level = '\t\t\t'
                            ) -> Tuple[nib.Nifti1Image, nib.Nifti1Image, List[Tuple[int]], List[Dict], str]:
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
    mask_filt_single_list = process_masks.isolate_single_CMBs(mask_filt, voxel_size)

    rg_metadata = []
    seeds_calculated = []
    final_processed_mask = np.zeros_like(mask_data, dtype=bool)
    msg += f'{log_level}Number of CMBs found in label id {str(labelid)}: {str(len(mask_filt_single_list))}.\n'

    # Processing loop with optional progress bar
    iterator = range(len(mask_filt_single_list))
    if show_progress:
        iterator = tqdm(iterator, total=len(mask_filt_single_list), desc="Processing CMBs")

    for i in iterator:
        cmb_single_mask = mask_filt_single_list[i]
        seeds = [tuple(seed) for seed in np.array(np.where(cmb_single_mask)).T]
        seeds_calculated.extend(seeds)

        processed_mask, metadata, msg = process_masks.region_growing_with_auto_tolerance(
            volume=mri_data,
            seeds=seeds,    
            size_threshold=size_threshold,
            max_dist_voxels=max_dist_voxels,
            tolerance_values=(0, 150, 0.5),
            connectivity=6,
            show_progress=show_progress,
            msg=msg,
            log_level=f"{log_level}\t",
            intensity_mode="point"
        )

        # Check for overlap
        if np.any(final_processed_mask & processed_mask):
            raise RuntimeError("Overlap detected between individual processed masks")

        final_processed_mask = final_processed_mask | processed_mask
        rg_metadata.append(metadata)
        msg += f"{log_level}Processed CMB {i}. n_seeds={len(seeds)}, new_size={np.sum(processed_mask)}\n"

    if not multiple and len(mask_filt_single_list) > 1:
        msg += f"{log_level}WARNING: Expected single CMBs and detected several.\n"

    return final_processed_mask, mask_filt, seeds_calculated, rg_metadata, msg

def process_cerebriu_anno(args: Any, 
                            subject: str, 
                            label_im: nib.Nifti1Image, 
                            mri_im: nib.Nifti1Image, 
                            seq_folder: str, 
                            msg: str,
                            log_level= "\t\t"
                            ) -> Tuple[nib.Nifti1Image, Dict, str]:
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

    # Compute size threshold and maximum distance in voxels
    size_th, max_dist_voxels = process_masks.calculate_size_and_distance_thresholds(mri_im, max_dist_mm=10)
    msg = f"{log_level}Thresholds for RegionGrowing --> Max. distance ={max_dist_voxels}, Max Size={size_th}\n"

    label_mask_all = np.zeros_like(label_im.get_fdata(), dtype=bool)

    for labelid, mask_dict in extracted_data["segmentMap"].items():
        multiple = mask_dict['attributes'].get('Multiple', False)
        msg += f"{log_level}Processing label {labelid} with {'multiple' if multiple else 'single'} CMB annotations.\n"
        
        label_mask, raw_mask, seeds, cmb_metadata, msg = process_CEREBRIU_cmb(
            label_im, labelid, mri_im, size_th, max_dist_voxels, msg, multiple)

        # Check for overlap
        if np.any(label_mask_all & label_mask):
            raise RuntimeError("Overlap detected between different CMB annotated masks")

        label_mask_all |= label_mask

    annotation_processed_nib = nib.Nifti1Image(label_mask_all.astype(np.int16), label_im.affine, label_im.header)
    processed_mask_nib, metadata, msg = process_masks.process_cmb_mask(annotation_processed_nib, msg, log_level="\t\t")

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

    # 2. Perform Quality Control (QC) and Data Cleaning
    sequences_qc, labels_qc, labels_metadata, msg = perform_CEREBRIU_QC(args, subject, sequences_raw, labels_raw, seq_folder, msg)
    
    
    
    return sequences_qc, labels_qc, labels_metadata, sequence_type, msg