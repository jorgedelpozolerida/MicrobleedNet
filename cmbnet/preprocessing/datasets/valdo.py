#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Module with functions for VALDO dataset

paper: 

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

import cmbnet.preprocessing.process_masks as utils_process
import cmbnet.utils.utils_general as utils_general


logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)



##############################################################################
###################                   VALDO                ###################
##############################################################################

def process_VALDO_mri(mri_im, msg='', log_level='\t\t'):
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
    
    return sequences_qc, labels_qc, labels_metadata, "T2S", msg
