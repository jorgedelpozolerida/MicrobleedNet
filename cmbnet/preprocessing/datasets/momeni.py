#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Module with functions for Momeni dataset

paper: https://www.sciencedirect.com/science/article/pii/S0169260721002029?via%3Dihub

TODO:
- implement momeni synth

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
###################          Momeni (real)                 ###################
##############################################################################


def extract_microbleed_coordinates_from_excel(excel_path, filename):
    """
    Extracts microbleed coordinates from an Excel file based on a matching filename in the first column.
    Processes only the first matching row if multiple are found, ignoring the rest.

    Args:
        excel_path (str): The path to the Excel file.
        filename (str): The filename to match against the first column for identifying the correct row.

    Returns:
        list of tuples: A list where each tuple contains the (x, y, z) coordinates of a microbleed, ensured to be integers.
    """
    df = pd.read_excel(excel_path)
    matching_rows = df[df[df.columns[0]] == filename]
    
    # Return an empty list if no matching row is found
    if matching_rows.shape[0] == 0:
        return []

    # Use only the first matching row, ignoring any additional matches
    matching_row = matching_rows.iloc[0]
    microbleed_coordinates = []
    
    # Iterate over the row's columns in groups of three to extract coordinates, starting from the second column
    for i in range(1, len(matching_row), 3):
        # Extract the group of three columns as coordinates
        x, y, z = matching_row.iloc[i:i+3].values
        
        # Check if the coordinates are integers or convertible to integers
        try:
            x, y, z = int(x), int(y), int(z)
            microbleed_coordinates.append((x, y, z))
        except (ValueError, TypeError):
            # Skip this group of coordinates if any of them isn't an integer or convertible to an integer
            continue

    return microbleed_coordinates


def load_MOMENI_raw(input_dir: str, study: str) -> Tuple[Dict[str, nib.Nifti1Image], Dict[str, nib.Nifti1Image], str, list]:
    """
    Load raw MRI and segmentation data for a given Momeni study, including centers of mass for microbleeds.
    """

    mri_dir = os.path.join(input_dir, "data", "PublicDataShare_2020", "rCMB_DefiniteSubject")
    mri_file = os.path.join(mri_dir, f"{study}.nii.gz")
    mri_nib = nib.load(mri_file)

    cmb_metadata_excel = os.path.join(input_dir, "data", "PublicDataShare_2020", "rCMBInformationInfo.xlsx")
    centers_of_mass = extract_microbleed_coordinates_from_excel(cmb_metadata_excel, f"{study}.nii.gz")

    com_list = []
    cmb_mask = np.zeros_like(mri_nib.get_fdata())
    for center in centers_of_mass:
        new_center = tuple(int(c)-1 for c in center) # correct indexing
        cmb_mask[new_center] = 1
        com_list.append(new_center)
    cmb_nib = nib.Nifti1Image(cmb_mask, affine=mri_nib.affine, header=mri_nib.header)

    seq_type = "SWI"
    sequences_raw = {seq_type: mri_nib}
    labels_raw = {seq_type: cmb_nib}

    return sequences_raw, labels_raw, seq_type, com_list

def process_MOMENI_anno(mri_im: nib.Nifti1Image, com_list: list, msg: str, connectivity=6, log_level="\t\t") -> Tuple[nib.Nifti1Image, Dict, str]:
    """
    Process annotations for a Momeni dataset subject by applying region growing algorithm on CMBs based on their COM.
    """
    # Compute size threshold and maximum distance in voxels
    size_th, max_dist_voxels = process_masks.calculate_size_and_distance_thresholds(mri_im, max_dist_mm=10)
    msg = f"{log_level}Thresholds for RegionGrowing --> Max. distance ={max_dist_voxels}, Max Size={size_th}\n"

    # Initialize the final processed mask
    final_processed_mask = np.zeros_like(mri_im.get_fdata(), dtype=bool)
    rg_metadata = {}  # To collect metadata from region growing
    msg += f"{log_level}Processing CMB annotations \n"

    # Process each CMB based on its center of mass
    for i, com in enumerate(com_list):
        seeds = [com]

        best_n_pixels = 1
        best_processed_mask = None
        best_metadata = None
        best_msg = ""

        # Iterate over combinations of parameters
        for connectivity in [6, 26]:
            for intensity_mode in ['point', 'running_average']:
                for diff_mode in ['relative', 'normal']:
                    if diff_mode == "relative":
                        range_temp = np.concatenate((np.arange(0, 20, 0.05), np.arange(20, 100, 1), np.arange(100, 10000, 100)))
                    else:
                        range_temp = np.arange(0, 100, 0.05)
                    msg_temp = ""
                    processed_mask, metadata, msg_temp = process_masks.region_growing_with_auto_tolerance(
                        volume=mri_im.get_fdata(),
                        seeds=seeds,
                        size_threshold=size_th,
                        max_dist_voxels=None,
                        tolerance_values=range_temp,
                        connectivity=connectivity,
                        show_progress=False,
                        intensity_mode=intensity_mode,
                        diff_mode=diff_mode,
                        log_level=f"{log_level}\t",
                        msg=msg_temp
                    )
                    n_pixels = metadata['n_pixels']
                    # Update best results if this combination yielded more pixels
                    if n_pixels > best_n_pixels:
                        best_n_pixels = n_pixels
                        bestconnectivity = connectivity
                        best_processed_mask = processed_mask
                        best_metadata = metadata
                        best_msg = msg_temp
                        best_intensity_mode = intensity_mode
                        best_diff_mode = diff_mode

        # Construct a final message summarizing the optimization result
        best_msg += f"{log_level}Optimization selected connectivity={bestconnectivity}, " \
                    f"intensity_mode={best_intensity_mode}, diff_mode={best_diff_mode} " \
                    f"with n_pixels={best_n_pixels}."
        msg += best_msg

        # Ensure there's no overlap with previously processed masks
        if np.any(final_processed_mask & best_processed_mask):
            raise RuntimeError("Overlap detected between individual processed masks.")

        # Update the final mask and metadata
        final_processed_mask |= best_processed_mask

        # save metadata for CMB i
        rg_metadata[i] = {
            "size": best_metadata['n_pixels'],
            "CM": com,
            "region_growing": {
                "selected_tolerance": best_metadata['tolerance_selected'],
                "n_tolerances": best_metadata['tolerances_inspected'], 
                "elbow_i": best_metadata['elbow_i'], 
                "elbow2end_tol": best_metadata['elbow2end_tol'],
                'connectivity': bestconnectivity,
                "intensity_mode": best_intensity_mode,
                "diff_mode": best_diff_mode ,
                "distance_th": max_dist_voxels,
                "size_th": size_th,

            }
        }
    # Save if healthy or not
    metadata_out = {
        "healthy": "no" if com_list else "yes",
        "CMBs_old": rg_metadata,
    }
    annotation_processed_nib = nib.Nifti1Image(final_processed_mask.astype(np.int16), mri_im.affine, mri_im.header)

    return annotation_processed_nib, metadata_out, msg


def process_MOMENI_mri(mri_im, msg='', log_level='\t\t'):
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


def perform_MOMENI_QC(subject, mris, annotations, com_list, msg):
    """
    Perform Quality Control (QC) specific to the Momeni dataset on MRI sequences and labels.

    Args:
        args: Configuration or parameters passed for QC.
        subject (str): The subject identifier.
        mris (dict): Dictionary of MRI sequences.
        annotations (dict): Dictionary of labels.
        com_list (list): List of center of mass coordinates for each CMB.
        msg (str): Log message to be updated.

    Returns:
        mris_qc (dict): Dictionary of QC'ed MRI sequences.
        annotations_qc (dict): Dictionary of QC'ed labels.
        annotations_metadata (dict): Metadata associated with the CMBs
        msg (str): Updated log message.
    """
    mris_qc, annotations_qc, annotations_metadata = {}, {}, {}

    for mri_sequence, mri_im in mris.items():
        mris_qc[mri_sequence], msg = process_MOMENI_mri(mri_im, msg)
    
    for anno_sequence, anno_im in annotations.items():
        annotations_qc[anno_sequence], metadata, msg = process_MOMENI_anno(anno_im, com_list, msg)
        annotations_metadata[anno_sequence] = metadata
    
    return mris_qc, annotations_qc, annotations_metadata, msg


def load_MOMENI_data(args, subject, msg):
    """
    Load MRI sequences and labels specific to the Momeni dataset. Performs QC in the process.

    Args:
        args: Configuration or parameters for loading data.
        subject (str): The subject identifier.
        msg (str): Log message to be updated.

    Returns:
        sequences_qc (dict): Dictionary of QC'ed MRI sequences.
        labels_qc (dict): Dictionary of QC'ed labels.
        labels_metadata (dict): Metadata associated with the labels.
        msg (str): Updated log message.
    """
    # 1. Load raw data
    sequences_raw, labels_raw, sequence_type, com_list = load_MOMENI_raw(args.input_dir, subject)

    # 2. Perform Quality Control and Data Cleaning
    sequences_qc, labels_qc, labels_metadata, msg = perform_MOMENI_QC(subject, sequences_raw, labels_raw, com_list, msg)

    # 3. Save plots for debugging
    for i, CM in enumerate(com_list):
        filename_temp = os.path.join( args.plots_path, f"{subject}_CMB_{i}.png")                  
        metadata_str = f"""\
        sub: {subject}
        {sequence_type}, shape: {sequences_raw[sequence_type].shape}
        CMBloc: {CM}   CMBsize: {labels_metadata[sequence_type]["CMBs_old"][i]['size']}
        "connectivity": {labels_metadata[sequence_type]["CMBs_old"][i]['region_growing']['connectivity']}, "intensity_mode": {labels_metadata[sequence_type]["CMBs_old"][i]['region_growing']['intensity_mode']}, 
        "diff_mode": {labels_metadata[sequence_type]["CMBs_old"][i]['region_growing']['diff_mode']}, 
        "distance_th": {labels_metadata[sequence_type]["CMBs_old"][i]['region_growing']['distance_th']},"size_th": {labels_metadata[sequence_type]["CMBs_old"][i]['region_growing']['size_th']}"""
        utils_plt.plot_processed_mask(sequences_raw[sequence_type], labels_raw[sequence_type], labels_qc[sequence_type], CM, 100, metadata_str=metadata_str, save_path=filename_temp)
    return sequences_qc, labels_qc, labels_metadata, sequence_type, msg





##############################################################################
##################          Momeni (synth)                 ###################
##############################################################################

def load_MOMENIsynth_data():
    raise NotImplementedError