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
import json

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
import cmbnet.utils.utils_plotting as utils_plt


logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

import warnings

warnings.filterwarnings("ignore", message="All-NaN slice encountered")



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


def perform_VALDO_QC(args, subject, mris, annotations, msg):
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
        mris_qc[mri_sequence], msg = process_VALDO_mri(mri_im, msg)
        
    # # Quality Control of Labels
    # for anno_sequence, anno_im in annotations.items():
    #     annotations_qc[anno_sequence], metadata, msg = process_masks.process_cmb_mask(anno_im, msg)
    #     annotations_metadata[anno_sequence] = metadata

    # Quality Control of Labels
    for anno_sequence, anno_im in annotations.items():
        
        if args.reprocess_file is None:
            annotations_qc[anno_sequence], metadata, msg = process_masks.process_cmb_mask(anno_im, msg)
            annotations_metadata[anno_sequence] = metadata
            # Prepare metadta in correct format
            metadata_out = {
                "healthy": "no" if annotations_metadata.get("T2S") else "yes",
                "CMBs_old": annotations_metadata.get("T2S", {}),
            }

        else:
            json_file = os.path.join(args.processed_dir, "Data", subject, "Annotations_metadata", f"{subject}_metadata.json")
            with open(json_file, 'r') as file:
                metadata_dict = json.load(file)
            com_list = [tuple(int(i) for i in cc["CM"]) for cc in metadata_dict['CMBs_old'].values()]

            annotations_qc[anno_sequence], metadata, msg = process_masks.reprocess_study(
                study=subject, processed_dir=args.processed_dir, mapping_file=args.reprocess_file,
                dataset=args.dataset_name, 
                mri_im=mris_qc[anno_sequence], com_list=com_list, msg=msg)
            annotations_metadata[anno_sequence] = metadata
            metadata_out = metadata

    return mris_qc, annotations_qc, metadata_out, msg


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
    t2s_path = os.path.join(subject_old_dir, f"{subject}_space-T2S_desc-masked_T2S.nii.gz")
    sequences_raw = {
        # "T1": nib.load(os.path.join(subject_old_dir, f"{subject}_space-T2S_desc-masked_T1.nii.gz")),
        # "T2": nib.load(os.path.join(subject_old_dir, f"{subject}_space-T2S_desc-masked_T2.nii.gz")),
        "T2S": nib.load(t2s_path)
    }
    
    # 2. Load Raw Labels (Annotations are made in T2S space for VALDO dataset)
    cmb_path = os.path.join(subject_old_dir, f"{subject}_space-T2S_CMB.nii.gz")
    labels_raw = {
        "T2S": nib.load(cmb_path)
    }
    
    # 3. Perform Quality Control (QC) on Loaded Data
    sequences_qc, labels_qc, labels_metadata, msg = perform_VALDO_QC(args, subject, sequences_raw, labels_raw, msg)
    
    # 4. Save plots for debugging
    utils_plt.generate_cmb_plots(
        subject, sequences_raw['T2S'], labels_raw['T2S'], 
        labels_qc['T2S'], labels_metadata['CMBs_old'], 
        plots_path=utils_general.ensure_directory_exists(os.path.join(args.plots_path, "pre")),
        zoom_size=100
    )
    nifti_paths = {
        "T2S": t2s_path,
        "CMB": cmb_path
    }
    new_n_CMB = len(labels_metadata['CMBs_old'])
    labels_metadata.update({"n_CMB_raw": new_n_CMB,"CMB_raw": []})

    metadata_out = {"T2S": labels_metadata }
    return sequences_qc, labels_qc, nifti_paths, metadata_out, "T2S", msg
