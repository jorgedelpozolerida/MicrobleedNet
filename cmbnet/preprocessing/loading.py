#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Script with utility functions to help loading and cleaning CMB datasets


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
import time 
from scipy.io import loadmat
import glob
import sys
from typing import Tuple, Dict, List, Any

current_dir_path = os.path.dirname(os.path.abspath(__file__))
parent_dir_path = os.path.abspath(os.path.join(current_dir_path, os.pardir))
sys.path.append(parent_dir_path)

import cmbnet.preprocessing.datasets as dat_load
import cmbnet.preprocessing.process_masks as process_masks
import cmbnet.utils.utils_general as utils_general


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
    Returns list of ALL studies from raw dataset directory
    
    Note: performs osme QCs in the process
    """
    if dataset_name == "valdo":
        assert "VALDO" in input_dir
        subjects = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

    elif dataset_name == "cerebriu":
        assert "CEREBRIU" in input_dir
        subjects = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

    elif dataset_name == "momeni":
        assert "MOMENI" in input_dir
        mri_dir = os.path.join(input_dir, "data", "PublicDataShare_2020", "rCMB_DefiniteSubject")
        mri_dir_healthy = os.path.join(input_dir, "data", "PublicDataShare_2020", "NoCMBSubject")
        subjects_cmb = [d.split(".")[0] for d in os.listdir(mri_dir)]
        subjects_h = [d.split(".")[0] for d in os.listdir(mri_dir_healthy)]
        subjects = subjects_cmb + subjects_h

    elif dataset_name == "momeni-synth":
        assert "MOMENI" in input_dir
        mri_dir = os.path.join(input_dir, "data", "PublicDataShare_2020", "sCMB_DefiniteSubject")
        mri_dir_healthy = os.path.join(input_dir, "data", "PublicDataShare_2020", "sCMB_NoCMBSubject")
        subjects_cmb = [d.split(".")[0] for d in os.listdir(mri_dir)]
        subjects_h = [d.split(".")[0] for d in os.listdir(mri_dir_healthy)]
        subjects = subjects_cmb + subjects_h

    elif dataset_name == "dou":
        assert "DOU" in input_dir
        subjects_mri = [d.split('.')[0] for d in os.listdir(os.path.join(input_dir, "nii"))]
        subjects_gt = [d.split('.')[0] for d in os.listdir(os.path.join(input_dir, "ground_truth"))]
        if set(subjects_mri) == set(subjects_gt):
            subjects = subjects_mri
        else:
            raise ValueError(f"Not all subjects contain annotations, check data")
    elif dataset_name == "rodeja":
        assert "RODEJA" in input_dir
        raise NotImplementedError
    else:
        raise NotImplementedError
    
    check_for_duplicates(subjects)

    return subjects


def load_mris_and_annotations(args, subject, msg='', log_level='\t\t'):
    '''
    Loads MRI scans and their corresponding annotations for a given subject 
    from a specific dataset performing data cleaning and orientation fix.    
    
    Args:
        args (object): Contains configuration parameters, including input directory and dataset name.
        subject (str): Identifier of the subject whose MRI scans and annotations are to be loaded.
        msg (str, optional): A string for logging purposes. Default is an empty string.
        
    Returns:
        mris (dict): Dictionary where keys are sequence names (e.g., "T1", "T2") and values are 
                        the corresponding MRI scans loaded as nibabel.Nifti1Image objects.
        annotations (dict): Dictionary where keys are sequence names and values are the corresponding 
                            annotations loaded as nibabel.Nifti1Image objects.
        labels_metadata (dict): Dictionary containing metadata related to labels (annotations).
        prim_seq (str): Primry sequence used for this subject
        msg (str): Updated log message.

    '''
    msg += f'{log_level}Loading MRI scans and annotations...\n'


    if args.dataset_name == "valdo":
        sequences_raw, labels_raw, labels_metadata, prim_seq, msg = dat_load.load_VALDO_data(args, subject, msg)
    elif args.dataset_name == "cerebriu":
        sequences_raw, labels_raw, labels_metadata, prim_seq, msg = dat_load.load_CEREBRIU_data(args, subject, msg)
    elif args.dataset_name == "dou":
        sequences_raw, labels_raw, labels_metadata, prim_seq, msg = dat_load.load_DOU_data(args, subject, msg)
    elif args.dataset_name == "momeni":
        sequences_raw, labels_raw, labels_metadata, prim_seq, msg = dat_load.load_MOMENI_data(args, subject, msg)
    elif args.dataset_name == "momeni-synth":
        raise NotImplementedError
        sequences_raw, labels_raw, labels_metadata,  msg = load_MOMENIsynth_data(args, subject, msg)
    elif args.dataset_name == "rodeja":
        raise NotImplementedError
        sequences_raw, labels_raw, labels_metadata,  msg = load_RODEJA_data(args, subject, msg)
    else:
        # Implement here for other datasets
        raise NotImplementedError

    start = time.time()

    mris = {}
    annotations = {}

    # Fill MRIs dict
    for sequence_name in sequences_raw:
        mris[sequence_name] = sequences_raw[sequence_name]
        msg += f'{log_level}Found {sequence_name} MRI sequence of shape {mris[sequence_name].shape}\n'

        # fix orientation and data type
        mris[sequence_name] = nib.as_closest_canonical(mris[sequence_name])
        mris[sequence_name].set_data_dtype(np.float32) 

        orientation = nib.aff2axcodes(mris[sequence_name].affine)
        if orientation != ('R', 'A', 'S'):
            raise ValueError("Image does not have RAS orientation.")

    assert prim_seq in mris

    # Fill annotations dict
    for sequence_name in sequences_raw:
        if sequence_name in labels_raw.keys():
            annotations[sequence_name] = labels_raw[sequence_name]
            msg += f'{log_level}Found {sequence_name} annotation of shape {annotations[sequence_name].shape}\n'
        else:
            annotations[sequence_name] = nib.Nifti1Image(np.zeros(shape=mris[prim_seq].shape),
                                                    affine=mris[prim_seq].affine,
                                                    header=mris[prim_seq].header)
            msg += f'{log_level}Missing {sequence_name} annotation, filling with 0s\n'

        # fix orientation adn data type
        annotations[sequence_name] = nib.as_closest_canonical(annotations[sequence_name])
        annotations[sequence_name].set_data_dtype(np.uint8)

        orientation = nib.aff2axcodes(annotations[sequence_name].affine)

        if orientation != ('R', 'A', 'S'):
            raise ValueError("Image does not have RAS orientation.")

    end = time.time()
    msg += f'{log_level}Loading of MRIs and annotations took {end - start} seconds!\n\n'

    return mris, annotations, labels_metadata, prim_seq, msg

import numpy as np

import numpy as np

def convert_numpy(obj):
    if isinstance(obj, np.integer):
        if isinstance(obj, np.int64):
            return int(obj)
        else:
            return obj.item()
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.complexfloating):
        return complex(obj)
    elif isinstance(obj, (np.void, np.record)):
        return obj.tobytes().hex()
    elif isinstance(obj, np.datetime64):
        return np.datetime_as_string(obj)
    elif isinstance(obj, np.timedelta64):
        return np.timedelta64(obj, 'ms').astype('timedelta64[ms]').astype('int64')
    return obj





def extract_im_specs(img):
    return {
        'shape': img.shape,
        'voxel_dim': img.header.get_zooms(),
        'orientation': nib.aff2axcodes(img.affine),
        'data_type': img.header.get_data_dtype().name,
    }


###############################################################################
# Old
###############################################################################

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
















