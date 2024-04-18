#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Module with functions for Rodeja dataset

paper: https://arxiv.org/abs/2301.09322


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
import json
from typing import Tuple, Dict, List, Any

from scipy.ndimage import center_of_mass

import cmbnet.preprocessing.process_masks as process_masks
import cmbnet.utils.utils_general as utils_general
import cmbnet.utils.utils_plotting as utils_plt
import cmbnet.preprocessing.loading as loading


logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings("ignore", message="The provided image has no sform in its header. Please check the provided file. Results may not be as expected.")

##############################################################################
###################              RODEJA/HERSLEV           ###################
##############################################################################

def find_macthing_files(pattern, search_path):
    """
    Find a single file matching a pattern in the given directory or its subdirectories.

    Args:
        pattern (str): The pattern to search for.
        search_path (str): The path to search in.

    Returns:
        str: The path to the found file.

    Raises:
        ValueError: If no files or more than one file are found.
    """
    matches = []
    for root, dirs, files in os.walk(search_path):
        matches.extend(os.path.join(root, file) for file in files if pattern in file)
    return matches

def load_RODEJA_raw(input_dir: str, study: str, msg: str, log_level: str) -> Tuple[Dict[str, nib.Nifti1Image], Dict[str, nib.Nifti1Image], str, List[dict], str]:
    """
    Load raw MRI and segmentation data for a given RODEJA study, including centers of mass for microbleeds.
    """
    seq_type = "SWI"
    
    annotations_dir = os.path.join(input_dir, "cmb_annotations", "Annotations")
    mri_dirs = [os.path.join(input_dir, "cmb_annotations", sub, "images") for sub in ["cph_annotated", "cph_annotated_mip"]]

    # Load Raw Labels (Segmentation Masks)
    labels_raw = {}
    label_filepaths = find_macthing_files(f"{study}mask.nii", annotations_dir)
    assert len(label_filepaths) == 1, f"More or no label files found: {label_filepaths}"
    im_anno = nib.load(label_filepaths[0])
    im_anno_modified, cmb_info, msg = process_rawmask_rodeja(im_anno, msg)
    labels_raw[seq_type] = im_anno_modified

    assert labels_raw, f"{log_level}Label not found for subject: {study}"

    sequences_raw = {}
    # Load Raw MRI Sequences
    mri_filepaths = []
    for mri_dir in mri_dirs:
        mri_filepaths += find_macthing_files(f"{study}.nii.gz", mri_dir)
    
    assert len(mri_filepaths) == 1, f"More or no label files found: {mri_filepaths}"
    sequences_raw[seq_type] = nib.load(mri_filepaths[0])

    assert sequences_raw, f"{log_level}MRI sequence not found for subject: {study}"

    nifti_paths = {
        seq_type: mri_filepaths[0],
        "CMB": label_filepaths[0]
    }
    
    assert sequences_raw[seq_type].shape == labels_raw[seq_type].shape, f"Different shapes, revise data"

    return sequences_raw, labels_raw, nifti_paths, seq_type, cmb_info, msg


def process_rawmask_rodeja(im_anno, msg):
    """
    Process a NIfTI mask file to identify CMBs, calculate their size and center of mass,
    and convert the mask into a binary format.
    
    Args:
        im_anno (nib.Nifti1Image): nibabel nifty image
    
    Returns:
        binary_im_anno (nib.Nifti1Image): Binary mask as a NIfTI image.
        cmb_info (list): List of dictionaries with 'label', 'size', and 'center_of_mass' for each CMB.
        msg
    """
    mask_data = im_anno.get_fdata()
    msg += f"\t\tCombining labels from original masks\n"

    # Identify unique labels in the mask, excluding the background
    unique_labels = np.unique(mask_data)
    unique_labels = unique_labels[unique_labels != 0]  # Exclude background

    cmb_info = {}
    com_list = []
    binary_mask_data = np.zeros_like(mask_data, dtype=bool)

    for i, lbl in enumerate(unique_labels):
        cmb_mask = (mask_data == lbl)  # Create a mask for the current label
        cmb_size = np.sum(cmb_mask)  # Calculate the size (count of voxels for the current label)
        cmb_com = center_of_mass(cmb_mask)  # Calculate the center of mass for the current label
        cmb_com = int(cmb_com[0]), int(cmb_com[1]), int(cmb_com[2])
        com_list.append(cmb_com)
        cmb_info[i] = {
            'label': lbl,
            'size': cmb_size,
            'CM': cmb_com
        }
        # Ensure there's no overlap with previously processed masks
        if np.any(np.logical_and(binary_mask_data, cmb_mask)):

            msg += f"\t\t\tCAUTION: Overlap detected at {cmb_com}\n" + \
                    f"\t\t\t         Previosly visited CMBs: {com_list[:i]}\n"
        # Update the final mask and metadata
        binary_mask_data |= cmb_mask


    # Create a new NIfTI image for the binary mask
    binary_im_anno = nib.Nifti1Image(binary_mask_data, im_anno.affine, im_anno.header)

    return binary_im_anno, cmb_info, msg

def reprocess_RODEJA_study(study, processed_dir, mapping_file, dataset,
                    mri_im: nib.Nifti1Image, 
                    cmb_info: list, msg: str,
                    log_level="\t\t"):
    """
    Re-Process annotations for dataset subject by creating a sphere.
    Radius is derived from:
    - Manually set radius in "manual_fixes.csv" file 
    OR
    - Using Region Growing estimated radius from first run
    """
    # Get study metadata
    json_file = os.path.join(processed_dir, "Data", study, "Annotations_metadata", f"{study}_metadata.json")
    with open(json_file, 'r') as file:
        metadata_dict = json.load(file)

    # Get mapping list
    msg += f"{log_level}---- Study found in manual fixes mapping CSV ----\n"

    # Compute size threshold and maximum distance in voxels
    size_th, max_dist_voxels = process_masks.calculate_size_and_distance_thresholds(mri_im, max_dist_mm=10)

    # Initialize the final processed mask
    final_processed_mask = np.zeros_like(mri_im.get_fdata(), dtype=bool)
    old_metadata = {}  # To collect metadata from original masks

    # Process each CMB based on its center of mass
    visited_coms = []
    for i, com_meta in cmb_info.items():
        
        com = com_meta['CM']
        size = com_meta['size']
        radius_orig = (3 * int(size) / (4 * np.pi))**(1/3)
        msg += f"{log_level}\tCMB-{i}\n"

        # Get from previous execution
        radius = radius_orig * 0.3
        msg += f"{log_level}\t\tWill use original radius of {radius_orig}, reduced to {radius}\n"

        processed_mask, metadata, msg = process_masks.grow_3D_sphere(
            example_im=mri_im,  # nibabel object
            seeds=[com],
            rad=radius,
            size_threshold=size_th,
            max_dist_voxels=max_dist_voxels,
            log_level=f"{log_level}\t\t",
            msg=msg
        )
        processed_mask = processed_mask.astype(bool)
        radius_new = (3 * int(metadata['n_pixels']) / (4 * np.pi))**(1/3)
        met_sph = metadata
        msg += f"{log_level}\t\tSphere created with radius {radius_new}mm, size={np.sum(processed_mask)}\n"

        if np.any(np.logical_and(final_processed_mask, processed_mask)):
            msg += f"{log_level}\t\tCAUTION: Overlap detected at {com}\n" + \
                    f"{log_level}\t\t         Previously visited CMBs: {visited_coms[:int(i)]}\n"

        final_processed_mask |= processed_mask
        old_metadata[i] = {
            "CM": com,
            "size": np.sum(processed_mask),
            "radius": round(radius, ndigits=2),
            "raw_metadata": com_meta,
            "sphere": met_sph
        }
        visited_coms.append(com)

    metadata_out = old_metadata
    annotation_processed_nib = nib.Nifti1Image(final_processed_mask.astype(np.int16), mri_im.affine, mri_im.header)

    return annotation_processed_nib, metadata_out, msg

def process_RODEJA_mri(mri_im, msg='', log_level='\t\t'):
    """
    Process a RODEJA MRI image to handle NaNs by replacing them with the background value.

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
            msg += f'{log_level}Forced background value to 0 as region selected is full of nan\n'
        # Replace NaNs with the background value
        data[nan_mask] = background_value

        msg += f'{log_level}Found {round(perc_nans, 2)}% of NaNs and replaced with background value: {background_value}\n'
    
    # Convert processed data back to Nifti1Image
    processed_mri_im = nib.Nifti1Image(data, mri_im.affine, mri_im.header)

    return processed_mri_im, msg

def perform_RODEJA_QC(args, subject, cmb_info, mris, annotations, msg):
    """
    Perform Quality Control (QC) specific to the RODEJA dataset on MRI sequences and labels.

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
    
    map_df = loading.get_sphere_df(args.reprocess_file, subject, args.dataset_name)
    subjects4reprocess = map_df['studyUID'].to_list()
    
    # Quality Control of MRI Sequences
    for mri_sequence, mri_im in mris.items():
        mris_qc[mri_sequence], msg = process_RODEJA_mri(mri_im, msg)

    # Quality Control of Labels
    for anno_sequence, anno_im in annotations.items():
        if args.reprocess_file is not None and subject in subjects4reprocess:
                annotations_qc[anno_sequence], metadata, msg = reprocess_RODEJA_study(
                    study=subject, processed_dir=args.processed_dir, mapping_file=args.reprocess_file,
                    dataset=args.dataset_name, 
                    mri_im=mris_qc[anno_sequence], 
                    cmb_info=cmb_info, 
                    msg=msg)
                annotations_metadata[anno_sequence] = metadata
        else:
            annotations_qc[anno_sequence], metadata, msg = process_masks.process_cmb_mask(anno_im, msg)
            annotations_metadata[anno_sequence] = metadata
        
    # Prepare metadta in correct format
    metadata_out = {
        "healthy": "no" if annotations_metadata.get("SWI") else "yes",
        "CMBs_old": annotations_metadata.get("SWI", {}),
    }

    return mris_qc, annotations_qc, metadata_out, msg

def load_RODEJA_data(args, subject, msg):
    """
    Load MRI sequences and labels specific to the RODEJA dataset. Performs QC in the process.

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
    # 1. Load Raw Annotations and MRI Sequences
    sequences_raw, labels_raw, nifti_paths, seq_type, cmb_info, msg = load_RODEJA_raw(args.input_dir, subject,
                                                                            msg, log_level="\t\t")

    # 2. Perform Quality Control (QC) on Loaded Data
    sequences_qc, labels_qc, labels_metadata, msg = perform_RODEJA_QC(args, subject, cmb_info, sequences_raw, labels_raw, msg)

    new_n_CMB = len(labels_metadata['CMBs_old'])
    old_n_CMB = len(cmb_info)
    if old_n_CMB != new_n_CMB:
        msg += f"\t\tCAUTION: there were originally {old_n_CMB} CMB labels and now {new_n_CMB} CCs detected\n"
    
    labels_metadata.update({"n_CMB_raw": old_n_CMB, "CMB_raw": cmb_info})

    # 4. Save plots for debugging
    utils_plt.generate_cmb_plots(
        subject, sequences_raw[seq_type], labels_raw[seq_type], 
        labels_qc[seq_type], labels_metadata['CMBs_old'], 
        plots_path=utils_general.ensure_directory_exists(os.path.join(args.plots_path, "pre")),
        zoom_size=100
    )
    metadata_out = {seq_type: labels_metadata}

    return sequences_qc, labels_qc, nifti_paths, metadata_out, seq_type, msg