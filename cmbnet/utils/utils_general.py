#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Script containing general utility functions


{Long Description of Script}


@author: jorgedelpozolerida
@date: 31/01/2024
"""

import os
import sys
import glob
import argparse
import traceback


import logging
import numpy as np
import pandas as pd
from datetime import datetime as dt
import json
import nibabel as nib
import ast


logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


###############################################################################
# General
###############################################################################

def ensure_directory_exists(dir_path, verbose=False):
    """ Create directory if non-existent """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        if verbose:
            print(f"Created the following dir: \n{dir_path}")
    return dir_path

def write_to_log_file(msg, log_file_path, printmsg=False):
    '''
    Writes message to the log file.
    Args:
        msg (str): Message to be written to log file.
        log_file_path (str): Path to log file.
    '''
    current_time = dt.now()
    with open(log_file_path, 'a+') as f:
        f.write(f'\n{current_time}\n{msg}')
    if printmsg:
        print(msg)
        
def confirm_action(message=""):
    """Prompt the user for confirmation before proceeding."""
    while True:
        answer = input(f'Do you want to proceed? [Y/n]: ')
        if not answer or answer[0].lower() == 'y':
            return answer
        elif answer[0].lower() == 'n':
            print('You did not approve. Exiting...')
            sys.exit(1)
        else:
            print('Invalid input. Please enter Y or n.')


def read_json_to_dict(file_path):
    """
    Reads a JSON file and converts it into a Python dictionary.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The JSON file content as a Python dictionary.
    """
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error reading the JSON file: {e}")
        return None
    
    
def create_nifti(data, affine, header, is_annotation=False):
    '''
    Creates Nifti1Image given data array, header and affine matrix.
    Args:
        data (np.ndarray): Input data array.
        affine (np.ndarray): Affine matrix.
        header (nib.Nifti1Header): Header.
        is_annotation (bool): Whether image is an annotation or not.
    Returns:
        image (nib.Nifti1Image): Created Nifti1Image.
    '''
    if is_annotation:
        image = nib.Nifti1Image(data.astype(np.uint8), affine=affine, header=header)
        image.set_data_dtype(np.uint8)

    else:
        image = nib.Nifti1Image(data.astype(np.float32), affine=affine, header=header)
        image.set_data_dtype(np.float32)

    return image



###############################################################################
# Data Loading and Manipulation
###############################################################################

def get_metadata_from_processed_final(data_dir, sub):

    metadata_dict = read_json_to_dict(
        os.path.join(data_dir, sub, "Annotations_metadata", f"{sub}_metadata.json")
    )
    metadata_dict_keys = list(metadata_dict.keys())
    return         {
            "id": sub,
            "anno_path": os.path.join(
                data_dir, sub, "Annotations", f"{sub}.nii.gz"
            ),
            "mri_path": os.path.join(data_dir, sub, "MRIs", f"{sub}.nii.gz"),
            "seq_type": metadata_dict_keys[0],
            "raw_metadata_path": os.path.join(
                data_dir, sub, "Annotations_metadata", f"{sub}_raw.json"
            ),
            "processed_metadata_path": os.path.join(
                data_dir, sub, "Annotations_metadata", f"{sub}_processed.json"
            ),
        }



def get_metadata_from_cmb_format(data_dir, sub_id):
    """
    Get all metadata from study using its subject id
    """

    fullpath_processing_metadata = os.path.join(
        data_dir, sub_id, "processing_metadata", f"{sub_id}.json"
    )
    processing_metadata_dict = read_json_to_dict(
        fullpath_processing_metadata
    )

    return {
        "id": sub_id,
        "anno_path": os.path.join(data_dir, sub_id, "Annotations", f"{sub_id}.nii.gz"),
        "mri_path": os.path.join(data_dir, sub_id, "MRIs", f"{sub_id}.nii.gz"),
        "processing_metadata_path": fullpath_processing_metadata, 
        **processing_metadata_dict,
    }

def load_clearml_predictions(pred_dir):
    
    subjects = os.listdir(pred_dir)
    metadata = []
    for sub in subjects:
        pred_files = glob.glob(
            os.path.join(pred_dir, sub, f"**/{sub}_PRED.nii.gz"), recursive=True
        )
        if len(pred_files) == 0:
            raise ValueError(
                f"No prediction files found for {sub}, check your data"
            )
        elif len(pred_files) > 1:
            raise ValueError(
                f"Multiple prediction files found for {sub}, check your data"
            )
        assert os.path.exists(pred_files[0])
        metadata.append({"id": sub, "pred_path": pred_files[0]})
        
    return metadata

def add_groundtruth_metadata(groundtruth_dir, gt_dir_struct, metadata):
    """
    Adds ground truth metadata to dict for subjects present.
    This function should be adapted to varying folder structures.
    """
    subjects_selected = [s_item["id"] for s_item in metadata]

    if gt_dir_struct == "processed_final":
        load_func = get_metadata_from_processed_final

    elif gt_dir_struct == "cmb_format":
        load_func = get_metadata_from_cmb_format
    else:
        raise NotImplementedError

    gt_metadata = {}

    for sub in subjects_selected:
        sub_meta = load_func(groundtruth_dir, sub)
        gt_metadata[sub] = sub_meta
    for meta_item in metadata:
        matching_item = gt_metadata[meta_item["id"]]
        meta_item.update({"gt_path": matching_item["anno_path"], **matching_item})

    return metadata


def add_CMB_metadata(CMB_metadata_df, metadata):

    for study_dict in metadata:
        sub_id = study_dict["id"]
        CMB_dict = study_dict['CMBs_new']
        for cmb_id, cmb_dict in CMB_dict.items():
            com = np.array(cmb_dict['CM'], dtype=np.int32) # Center of mass
            cmb_row = CMB_metadata_df[
                (CMB_metadata_df['seriesUID'] == sub_id) & (CMB_metadata_df['cmb_id'].astype(int) == int(cmb_id))
            ] 
            if cmb_row.empty:
                raise ValueError(f"CMB {cmb_id} not found for subject {sub_id}")
            cmb_row = cmb_row.to_dict(orient='records')[0]
            assert all(com == cmb_row['CM']), f"CM not mathcing for {sub_id} - {cmb_id}"
            cmb_row['CM'] = tuple(map(int, com))
            cmb_dict.update(cmb_row)
    return metadata