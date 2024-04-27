#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Script containing general utility functions


{Long Description of Script}


@author: jorgedelpozolerida
@date: 31/01/2024
"""

import os
import sys
import argparse
import traceback


import logging
import numpy as np
import pandas as pd
from datetime import datetime as dt
import json
import nibabel as nib


logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def ensure_directory_exists(dir_path, verbose=False):
    """ Create directory if non-existent """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        if verbose:
            print(f"Created the following dir: \n{dir_path}")
    return dir_path

def write_to_log_file(msg, log_file_path):
    '''
    Writes message to the log file.
    Args:
        msg (str): Message to be written to log file.
        log_file_path (str): Path to log file.
    '''
    current_time = dt.now()
    with open(log_file_path, 'a+') as f:
        f.write(f'\n{current_time}\n{msg}')
        
def confirm_action(message=""):
    """Prompt the user for confirmation before proceeding."""
    confirm = input(f"{message}\nDo you want to continue (yes/no): ")
    if confirm.lower() != 'yes':
        print("Action cancelled by user.")
        sys.exit(0)


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