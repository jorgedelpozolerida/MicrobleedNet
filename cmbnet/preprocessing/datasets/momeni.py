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
        cmb_mask[center] = 1
        com_list.append(center)
    cmb_nib = nib.Nifti1Image(cmb_mask, affine=mri_nib.affine, header=mri_nib.header)

    seq_type = "SWI"
    sequences_raw = {seq_type: mri_nib}
    labels_raw = {seq_type: cmb_nib}

    return sequences_raw, labels_raw, seq_type, com_list

def process_MOMENI_anno(mri_im: nib.Nifti1Image, com_list: list, msg: str, connectivity=6, log_level="\t\t") -> Tuple[nib.Nifti1Image, Dict, str]:
    """
    Process annotations for a Momeni dataset subject by applying region growing algorithm on CMBs based on their COM.
    """
    size_th, max_dist_voxels = process_masks.calculate_size_and_distance_thresholds(mri_im, max_dist_mm=10)
    msg = f"{log_level}Thresholds for RegionGrowing --> Max. distance ={max_dist_voxels}, Max Size={size_th}\n"

    final_processed_mask = np.zeros_like(mri_im.get_fdata(), dtype=bool)
    rg_metadata = []
    msg += f"{log_level}Processing CMB annotations \n"

    for i, com in enumerate(com_list):
        seeds = [com]
        processed_mask, metadata, msg = process_masks.region_growing_with_auto_tolerance(
            volume=mri_im.get_fdata(),
            seeds=seeds,
            size_threshold=size_th,
            max_dist_voxels=max_dist_voxels,
            tolerance_range=(0, 100, 0.05),
            connectivity=connectivity,
            show_progress=True,
            intensity_mode="point",
            diff_mode="normal",
            log_level=f"{log_level}\t",
            msg=msg
        )

        if np.any(final_processed_mask & processed_mask):
            raise RuntimeError("Overlap detected between individual processed masks.")

        final_processed_mask |= processed_mask
        rg_metadata.append(metadata)
        msg += f"{log_level}Processed CMB {i}. center of mass={com}, new_size={np.sum(processed_mask)}\n"

    annotation_processed_nib = nib.Nifti1Image(final_processed_mask.astype(np.int16), mri_im.affine, mri_im.header)

    return annotation_processed_nib, rg_metadata, msg


def process_MOMENI_mri(subject, mri_im, msg):
    """
    Process MRI sequences specific to the Momeni dataset. Placeholder for actual processing logic.

    Args:
        args: Configuration or parameters passed for processing.
        subject (str): The subject identifier.
        mri_im (nibabel.Nifti1Image): The MRI image to process.
        msg (str): Log message to be updated.

    Returns:
        mri_im (nibabel.Nifti1Image): Processed MRI image.
        msg (str): Updated log message.
    """
    # Placeholder for processing logic, adjust as necessary for Momeni dataset specifics
    return mri_im, msg


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
        annotations_metadata (dict): Metadata associated with the QC'ed labels.
        msg (str): Updated log message.
    """
    mris_qc, annotations_qc, annotations_metadata = {}, {}, {}

    for mri_sequence, mri_im in mris.items():
        mris_qc[mri_sequence], msg = process_MOMENI_mri(subject, mri_im, msg)
    
    for anno_sequence, anno_im in annotations.items():
        annotations_qc[anno_sequence], metadata, msg = process_MOMENI_anno(anno_im, com_list, msg)
        annotations_metadata[anno_sequence] = metadata

    return mris_qc, annotations_qc, annotations_metadata, msg


def load_MOMENI_data(args, input_dir, subject, msg):
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
    sequences_raw, labels_raw, sequence_type, com_list = load_MOMENI_raw(input_dir, subject)

    sequences_qc, labels_qc, labels_metadata, msg = perform_MOMENI_QC(subject, sequences_raw, labels_raw, com_list, msg)
    
    return sequences_qc, labels_qc, labels_metadata, sequence_type, msg


##############################################################################
##################          Momeni (synth)                 ###################
##############################################################################

def load_MOMENIsynth_data():
    raise NotImplementedError