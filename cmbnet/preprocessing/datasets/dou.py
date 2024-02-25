#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Module with functions for Dou dataset

paper: https://www.cse.cuhk.edu.hk/~qdou/cmb-3dcnn/cmb-3dcnn.html

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
import cmbnet.visualization.utils_plotting as utils_plt


logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

##############################################################################
###################                   DOU                  ###################
##############################################################################


def load_DOU_raw(input_dir: str, study: str) -> Tuple[Dict[str, nib.Nifti1Image], Dict[str, nib.Nifti1Image], str]:
    """
    Load raw MRI and segmentation data for a given DOU study, including centers of mass for microbleeds.

    Args:
        input_dir (str): Directory containing the study subfolders.
        study (str): Specific study identifier.

    Returns:
        Tuple containing dictionaries of raw MRI sequences and labels, sequence type, and list with centers of mass.
    """
    mri_dir = os.path.join(input_dir, "nii")
    cmb_dir = os.path.join(input_dir, "ground_truth")

    cmb_file = os.path.join(cmb_dir, f"{study}.mat")
    mri_file = os.path.join(mri_dir, f"{study}.nii")

    # Load the MRI data
    mri_nib = nib.load(mri_file)
    mri_data = mri_nib.get_fdata()

    # Initialize an empty mask with the same dimensions as the MRI data
    cmb_mask = np.zeros(mri_data.shape)

    # Load the centers of mass from the .mat file
    mat_data = loadmat(cmb_file)
    centers_of_mass = mat_data['cen']  # Assuming 'cen' is the key for centers of mass

    # Mark each center of mass in the cmb_mask
    com_list = []
    for center in centers_of_mass:
        com = [center[0], center[1], center[2]]
        new_center = tuple(int(c)-1 for c in com) # correct indexing
        cmb_mask[new_center] = 1
        com_list.append(tuple(new_center))

    # Create a NIfTI image from the cmb_mask
    cmb_nib = nib.Nifti1Image(cmb_mask, affine=mri_nib.affine, header=mri_nib.header)

    # Load Raw MRI Sequences and Labels
    seq_type = "SWI"
    sequences_raw = {seq_type: mri_nib}
    labels_raw = {seq_type: cmb_nib}

    return sequences_raw, labels_raw, seq_type, com_list



def process_DOU_anno(mri_im: nib.Nifti1Image, com_list: list, msg: str, log_level='\t\t') -> Tuple[nib.Nifti1Image, Dict, str]:
    """
    Process annotations for a DOU dataset subject by applying region growing algorithm on Cerebral Microbleeds (CMBs) 
    based on their center of mass (COM).

    Args:
        mri_im (nib.Nifti1Image): Input MRI image as a Nifti1Image object.
        com_list (list): List of center of mass coordinates for each CMB.
        msg (str): Initial log message to which this function will append.

    Returns:
        Tuple[nib.Nifti1Image, Dict, str]: A tuple containing the processed annotation as a Nifti1Image, 
                                            metadata about the processing, and an updated log message.
    """
    # Compute size threshold and maximum distance in voxels
    size_th, max_dist_voxels = process_masks.calculate_size_and_distance_thresholds(mri_im, max_dist_mm=10)

    # Initialize the final processed mask
    final_processed_mask = np.zeros_like(mri_im.get_fdata(), dtype=bool)
    rg_metadata = {}  # To collect metadata from region growing
    msg += f"{log_level}Applying Region Growing with max_distance={max_dist_voxels}, max_size={size_th}\n\n"

    # Process each CMB based on its center of mass
    for i, com in enumerate(com_list):
        msg += f"{log_level}\tCMB-{i}\n"
        seeds = [com]
        best_n_pixels = 1
        counter = 0
        # Iterate over combinations of parameters
        for connectivity in [6, 26]:
            for intensity_mode in ['point', 'running_average']:
                for diff_mode in ['relative', 'normal']:
                    if diff_mode == "relative":
                        range_temp = np.concatenate((np.arange(1e-3, 20, 0.05), np.arange(20, 100, 1), np.arange(100, 10000, 100)))
                    else:
                        range_temp = np.arange(1e-3, 100, 0.05)
                    msg_temp = ""
                    processed_mask, metadata, msg_temp = process_masks.region_growing_with_auto_tolerance(
                        volume=mri_im.get_fdata(),
                        seeds=seeds,
                        size_threshold=size_th,
                        max_dist_voxels=max_dist_voxels,
                        tolerance_values=range_temp,
                        connectivity=connectivity,
                        show_progress=True,
                        intensity_mode=intensity_mode,
                        diff_mode=diff_mode,
                        log_level=f"{log_level}\t\t",
                        msg=msg_temp
                    )
                    n_pixels = metadata['n_pixels']
                    # Update best results if this combination yielded more pixels
                    if (n_pixels > best_n_pixels) or counter == 0:
                        best_n_pixels = n_pixels
                        bestconnectivity = connectivity
                        best_processed_mask = processed_mask
                        best_metadata = metadata
                        best_msg = msg_temp
                        best_intensity_mode = intensity_mode
                        best_diff_mode = diff_mode
                    counter += 1

        # Construct a final message summarizing the optimization result
        msg += best_msg
        msg += f"{log_level}\t\tOptimization chose: '{bestconnectivity}-conn', " \
                    f"'{best_intensity_mode}', '{best_diff_mode}', " \
                    f"size={best_n_pixels}.\n"
        # Ensure there's no overlap with previously processed masks
        if np.any(final_processed_mask & best_processed_mask):
            msg += f"{log_level}\t\tCAUTION: Overlap detected at {com}\n" + \
                    f"{log_level}\t\t         Previosly visited CMBs: {com_list[:i]}\n"
        # Update the final mask and metadata
        final_processed_mask |= best_processed_mask

        # radius
        radius = (3 * int(best_metadata['n_pixels']) / (4 * np.pi))**(1/3)

        # save metadata for CMB i
        rg_metadata[i] = {
            "CM": com,
            "size": best_metadata['n_pixels'],
            "radius": round(radius, ndigits=2),
            "region_growing": {
                "distance_th": max_dist_voxels,
                "size_th": size_th,
                'sphericity_ind': best_metadata['sphericity_ind'], 
                "selected_tolerance": best_metadata['tolerance_selected'],
                "n_tolerances": best_metadata['tolerances_inspected'], 
                "elbow_i": best_metadata['elbow_i'], 
                "elbow2end_tol": best_metadata['elbow2end_tol'],
                'connectivity': bestconnectivity,
                "intensity_mode": best_intensity_mode,
                "diff_mode": best_diff_mode 
            }
        }
    # Save if healthy or not
    metadata_out = {
        "healthy": "no" if com_list else "yes",
        "CMBs_old": rg_metadata
    }
    annotation_processed_nib = nib.Nifti1Image(final_processed_mask.astype(np.int16), mri_im.affine, mri_im.header)
    
    return annotation_processed_nib, metadata_out, msg


def process_DOU_mri(mri_im, msg='', log_level='\t\t'):
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
            msg += f'{log_level}Forced background value to 0 as region selected is full of nan\n'
        # Replace NaNs with the background value
        data[nan_mask] = background_value

        msg += f'{log_level}Found {round(perc_nans, 2)}% of NaNs and replaced with background value: {background_value}\n'
    
    # Convert processed data back to Nifti1Image
    processed_mri_im = nib.Nifti1Image(data, mri_im.affine, mri_im.header)

    return processed_mri_im, msg

def perform_DOU_QC(args, subject, mris, annotations, com_list, msg):
    """
    Perform Quality Control (QC) specific to the VALDO dataset on MRI sequences and labels.

    Args:
        args (Namespace): Arguments passed to the main function.
        subject (str): The subject identifier.
        mris (dict): Dictionary of MRI sequences.
        annotations (dict): Dictionary of labels.
        com_list
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
        mris_qc[mri_sequence], msg = process_DOU_mri(mri_im, msg)
    
    # Quality Control of Labels
    for anno_sequence, anno_im in annotations.items():
        annotations_qc[anno_sequence], metadata, msg = process_DOU_anno(mris_qc[anno_sequence], com_list, msg)
        annotations_metadata[anno_sequence] = metadata

    return mris_qc, annotations_qc, annotations_metadata, msg


def load_DOU_data(args, subject, msg):
    """
    Load MRI sequences and labels specific to the DOU dataset. Performs QC in the process.

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
    sequences_raw, labels_raw, sequence_type, com_list = load_DOU_raw(args.input_dir, subject)

    # 2. Perform Quality Control and Data Cleaning
    sequences_qc, labels_qc, labels_metadata, msg = perform_DOU_QC(args, subject, sequences_raw, labels_raw, com_list, msg)
    
    
    # 3. Save plots for debugging
    utils_plt.generate_cmb_plots(
        subject, sequences_raw[sequence_type], labels_raw[sequence_type], 
        labels_qc[sequence_type], labels_metadata[sequence_type]['CMBs_old'], 
        plots_path=utils_general.ensure_directory_exists(os.path.join(args.plots_path, "pre")),
        zoom_size=100
    )
    return sequences_qc, labels_qc, labels_metadata, sequence_type, msg