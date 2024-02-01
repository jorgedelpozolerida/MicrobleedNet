#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Script to apply post-processing steps to models' predictions


NOTE:
- must be run with its own venv for SynthSeg

TODO:
- SynthSeg + fitering
- Mask cleaning
- Shape restrictions
- Model?

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
import glob
import json

import subprocess
import nibabel as nib
from nilearn.image import resample_to_img

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

# Utils
# import utils.utils_processing as utils_process
# import utils.utils_general as utils_general
# import utils.utils_evaluation as utils_eval

def apply_synthseg(input_path, output_path, synthseg_repo_path):
    # Construct the command
    command = ["python", f"{synthseg_repo_path}/scripts/commands/SynthSeg_predict.py",
                "--i", input_path, 
                "--o", output_path]

    # Log the command
    logging.info("Running command: " + ' '.join(command))

    # Run the command
    subprocess.run(command, check=True)

def load_clearml_predictions(args):
    """
    Loads subjects metadata following clearml folder structure
    
    Returns list with dictionaries containing "id" and "pred_path"
    """
    pred_dir = args.predictions_dir
    if args.pred_dir_struct == "clearml":
        subjects = os.listdir(pred_dir)
        metadata = []
        for sub in subjects:
            pred_files = glob.glob(os.path.join(pred_dir, sub, f"**/{sub}_PRED.nii.gz"), recursive = True)
            if len(pred_files) == 0:
                raise ValueError(f"No prediction files found for {sub}, check your data")
            elif len(pred_files) > 1:
                raise ValueError(f"Multiple prediction files found for {sub}, check your data")
            assert os.path.exists(pred_files[0])
            metadata.append(
                {
                    "id": sub,
                    "pred_path": pred_files[0]
                }
            )
        return metadata
    else:
        raise NotImplementedError
    
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
        metadata_dict = read_json_to_dict(os.path.join(data_dir, sub, "Annotations_metadata", f"{sub}_raw.json"))
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


def main(args):

    studies_metadata = get_files_metadata_from_processed(args.data_dir)
    pred_metadata = load_clearml_predictions(args)

    '''
    Each entry Looks like:
                {
                "id": sub,
                "anno_path": os.path.join(data_dir, sub, "Annotations", f"{sub}.nii.gz"),
                "mri_path": os.path.join(data_dir, sub, "MRIs", f"{sub}.nii.gz"),
                "seq_type": metadata_dict_keys[0],
                "raw_metadata_path": os.path.join(data_dir, sub, "Annotations_metadata", f"{sub}_raw.json"),
                "processed_metadata_path": os.path.join(data_dir, sub, "Annotations_metadata", f"{sub}_processed.json")
            } 
    '''
    

    for study_item in studies_metadata:
        studyuid = study_item['id']
        mri_path = study_item['mri_path']
        matching_pred_item = [i for i in pred_metadata if i['id'] == studyuid][0]
        pred_path = matching_pred_item['pred_path']
        output_path = os.path.join(args.output_dir, f"{studyuid}_synthseg.nii.gz")

        # Apply SynthSeg
        apply_synthseg(mri_path, args.output_dir, args.synthseg_repo_path)

        # Resample SynthSeg output
        mri_img = nib.load(mri_path)
        pred_img = nib.load(pred_path)
        synthseg_img = nib.load(output_path)
        resampled_img = resample_to_img(synthseg_img, mri_img, interpolation="nearest")

        resampled_path = os.path.join(args.output_dir, f"{studyuid}_resampled.nii.gz")
        nib.save(resampled_img, resampled_path)

        print(mri_img.shape, pred_img.shape, resampled_img.shape)

        sys.exit()

    
    # 1. Apply SynthSeg and filter out CMBs 
    # for each study in    studies_metadata
    # run something like  "python /home/cerebriu/data/RESEARCH/SynthSeg-master/scripts/commands/SynthSeg_predict.py 
    # --i f'{data_dir}/{study}/MRIs/{study}.nii.gz' 
    # --o output_dir
    
    # wait
    
    #     # load data created in outputdir with same name but ending in _synthseg.nii.gz

    
    # 2. Resample to mri using nearest neighbout interpolation form nilearn
    # resample synthseg mask to original mri dimensions 
    
    

    # 3. Comapre original mask with synthseg and create new masks only for brain regions

    return


def parse_args():
    '''
    Parses all script arguments.
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to the directory with processed data dir')
    parser.add_argument('--predictions_dir', type=str, default=None,
                        help='Path to the directory with predictions')
    parser.add_argument('--pred_dir_struct', type=str, default=None, choices= ['clearml'],
                        help='Type of structure for saved predictions')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Full path to the directory where results and logs will be saved')
    parser.add_argument('--synthseg_repo_path', type=str, default=None,
                        help='Full path to the SynthSeg repo')  
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)