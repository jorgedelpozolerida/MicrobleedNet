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
import scipy.ndimage as ndi
from collections import Counter


logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

# Utils
# import utils.utils_processing as utils_process
# import utils.utils_general as utils_general
# import utils.utils_evaluation as utils_eval
BRAIN_LABELS = set([
    2,  # left cerebral white matter
    3,  # left cerebral cortex
    7,  # left cerebellum white matter
    8,  # left cerebellum cortex
    10, # left thalamus
    11, # left caudate
    12, # left putamen
    13, # left pallidum
    17, # left hippocampus
    18, # left amygdala
    26, # left accumbens area
    28, # left ventral DC (Diencephalon)
    41, # right cerebral white matter
    42, # right cerebral cortex
    46, # right cerebellum white matter
    47, # right cerebellum cortex
    49, # right thalamus
    50, # right caudate
    51, # right putamen
    52, # right pallidum
    53, # right hippocampus
    54, # right amygdala
    58, # right accumbens area
    60  # right ventral DC (Diencephalon)
])


def apply_synthseg(args, input_path, output_path, synthseg_repo_path):
    # Construct the command
    command = ["python", f"{synthseg_repo_path}/scripts/commands/SynthSeg_predict.py",
                "--i", input_path, 
                "--o", output_path]

    if args.robust_synthseg:
        command.extend(["--robust"])

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


# Function to read SynthSeg labels
def read_synthseg_labels(file_path):
    labels_dict = {}
    with open(file_path, 'r') as file:
        # Skip header lines until we reach the line starting with 'labels'
        for line in file:
            if line.strip().lower().startswith('labels'):
                break

        # Process the label lines
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 2:
                label_num = int(parts[0])
                label_name = ' '.join(parts[1:])
                labels_dict[label_num] = label_name
    return labels_dict


def filter_mask_with_synthseg(args, studyuid, pred_img_data, resampled_img_data, ground_truth_img_data, brain_labels, non_brain_labels, synthseg_labels):
    _logger.info(f"Processing {studyuid} with custom SynthSeg filtering criteria")

    labeled_array, num_features = ndi.label(pred_img_data)
    labeled_array_gt, num_features_gt = ndi.label(ground_truth_img_data)

    new_mask = np.zeros_like(pred_img_data, dtype=np.int16)
    cortex_labels = set([3, 8, 42, 47])  # Cortex labels

    all_results = []
    for i in range(1, num_features + 1):
        component = (labeled_array == i)
        component_labels = resampled_img_data[component]

        # Count overlap with non-brain regions
        overlap_with_non_brain = np.count_nonzero(~np.isin(component_labels, list(brain_labels)))
        
        # Count overlap with brain regions
        overlap_with_brain = np.count_nonzero(np.isin(component_labels, list(brain_labels)))

        # Check overlap with ground truth
        overlap_with_ground_truth = np.any(ground_truth_img_data[component])

        # Calculate percentage overlap for each label type in the component
        overlap_percentages = {label: np.mean(component_labels == label) * 100 for label in np.unique(component_labels)}
        overlap_percentages_str = '\n\t\t\t'.join([f"{label} ({synthseg_labels.get(label, 'Unknown')}): {percent:.2f}%" for label, percent in overlap_percentages.items()])
        
        # Check for the overlap with cortex and non-brain regions
        cortex_and_non_brain_overlap = sum(overlap_percentages.get(label, 0) for label in cortex_labels | non_brain_labels)

        # Determine if the component is overlapping
        component_type = "overlap" if overlap_with_ground_truth else "-"
        indent = '    '  # Indentation (4 spaces)

        print(f"\tComponent {i} ({component_type}):")
        print(f"\t{indent}Overlaps:  {round(overlap_with_non_brain/np.sum(component)*100, 2)}-{round(overlap_with_brain/np.sum(component)*100, 2)}-{round(cortex_and_non_brain_overlap, 2)}")
        print(f"\t{indent}Overlap percentages with label types:\n\t\t\t{overlap_percentages_str}")
        
        # # Discard component if more than half overlaps with cortex and non-brain labels
        # if (cortex_and_non_brain_overlap >= 50 and (not overlap_with_brain>0) 
        #     or 
        #     int(cortex_and_non_brain_overlap)==100 ):
            
        #     result = "FN-1" if component_type == "overlap" else "TN-1"
        #     print(f"\t{indent}REMOVE-disc ({result}) -> more than 50% overlap with cortex and non-brain regions")

        # Log overlap details and percentages
        if overlap_with_non_brain > 0:
            overlap_labels = np.unique(component_labels)
            overlap_labels_str = ', '.join([synthseg_labels.get(label, str(label)) for label in overlap_labels])

            # if overlap_with_non_brain < np.sum(component) / 2:
            if overlap_with_brain>0:

                result = "TP-1" if component_type == "overlap" else "FP-1"
                print(f"\t{indent}GOOD-pass ({result}) -> CMB partially overlaps with non-brain regions")
                new_mask[component] = 1
            else:
                result = "FN-2" if component_type == "overlap" else "TN-2"
                print(f"\t{indent}REMOVE ({result}) -> CMB mostly overlaps with non-brain regions")
        else:
            result = "TP-2" if component_type == "overlap" else "FP-2"
            print(f"\t{indent}GOOD ({result}) -> CMB overlaps with brain regions only")
            new_mask[component] = 1

        all_results.append(result)
        
    # Count and log the number of connected components after filtering
    _, num_features_new = ndi.label(new_mask)
    print(all_results)
    print(f"OVERVIEW OF CMBs: {num_features} -----> {num_features_new} // {num_features_gt} ")
    _logger.info(f"Finished processing {studyuid}")

    return new_mask, all_results



def main(args):


    if not os.path.exists(args.processed_masks_dir):
        os.makedirs(args.processed_masks_dir)
        _logger.info(f"Processed masks directory created at {args.processed_masks_dir}")
    
    pred_metadata = load_clearml_predictions(args)
    studies_metadata = get_files_metadata_from_processed(args.data_dir, [s_item['id'] for s_item in pred_metadata])



    # Read SynthSeg labels
    labels_file = os.path.join(args.synthseg_repo_path, 'data', 'labels table.txt')
    synthseg_labels = read_synthseg_labels(labels_file)

    # Log brain and non-brain labels
    non_brain_labels = set(synthseg_labels.keys()) - BRAIN_LABELS
    brain_labels_str = '\n'.join([f"{label}: {synthseg_labels[label]}" for label in BRAIN_LABELS if label in synthseg_labels])
    non_brain_labels_str = '\n'.join([f"{label}: {synthseg_labels[label]}" for label in non_brain_labels if label in synthseg_labels])
    
    _logger.info(f"Brain labels used:\n{brain_labels_str}")
    _logger.info(f"Non-brain labels used:\n{non_brain_labels_str}")

    all_r = []
    for study_item in studies_metadata:
        studyuid = study_item['id']
        mri_path = study_item['mri_path']
        gt_path = study_item['anno_path']
        print('----------------------')
        print(f'Processing study {studyuid}')
        
        matching_pred_item = [i for i in pred_metadata if i['id'] == studyuid][0]
        pred_path = matching_pred_item['pred_path']
        output_path = os.path.join(args.output_dir, f"{studyuid}_synthseg.nii.gz")

        # Apply/Load SynthSeg
        if not os.path.exists(output_path) or (args.overwrite_synthseg):
            _logger.info(f"SynthSeg output not found. Will apply SynthSeg to MRI at {mri_path}")
            apply_synthseg(args, mri_path, args.output_dir, args.synthseg_repo_path)
            _logger.info(f"SynthSeg applied to {studyuid} and output saved to {output_path}")
        else:
            _logger.info(f"SynthSeg output found at {output_path}")

        # Create new processed mask if not exists or if overwrite
        new_mask_path = os.path.join(args.processed_masks_dir, f"{studyuid}_brain_mask.nii.gz")

        if not os.path.exists(new_mask_path) or (args.overwrite_newmasks):
            # Load the MRI, prediction, and SynthSeg images
            print(f"Loading MRI, prediction, and SynthSeg images for {studyuid}")
            mri_img = nib.load(mri_path)
            pred_img = nib.load(pred_path)
            gt_img = nib.load(gt_path)

            synthseg_img = nib.load(output_path)
            resampled_synthseg_img = resample_to_img(synthseg_img, mri_img, interpolation="nearest")

            # Get the data arrays and squeeze just in case
            _logger.info(f"Resampling SynthSeg image to match MRI for {studyuid}")
            pred_img_data = np.squeeze(pred_img.get_fdata())
            resampled_synthseg_data = np.squeeze(resampled_synthseg_img.get_fdata())

            # Create a new mask based on original predictions and SynthSeg brain regions
            _logger.info(f"Creating new brain mask for {studyuid}")
            new_mask, res = filter_mask_with_synthseg(args, studyuid, pred_img_data, resampled_synthseg_data, gt_img.get_fdata(), brain_labels=BRAIN_LABELS, non_brain_labels=non_brain_labels, synthseg_labels=synthseg_labels)

            # Save the new mask
            new_mask_img = nib.Nifti1Image(new_mask, mri_img.affine)
            nib.save(new_mask_img, new_mask_path)
            _logger.info(f"New brain mask for {studyuid} saved to {new_mask_path}")
            all_r.extend(res)

    print("*******************************")
    # Calculate the frequency of each element in the list
    element_frequencies = Counter(all_r)

    # Print the frequencies
    for element, frequency in element_frequencies.items():
        print(f"Element {element} occurs {frequency} times.")
    print("*******************************")
    
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
    parser.add_argument('--processed_masks_dir', type=str, required=True,
                        help='Directory to save processed masks')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Full path to the directory where results and logs will be saved')
    parser.add_argument('--synthseg_repo_path', type=str, default=None,
                        help='Full path to the SynthSeg repo') 
    parser.add_argument('--robust_synthseg',  default=False, action='store_true',
                        help='Add this flag if you want SynthSeg to be in roust mode')
    parser.add_argument('--overwrite_synthseg',  default=False, action='store_true',
                        help='Add this flag if you want previous SynthSeg runs to be overwritten')
    parser.add_argument('--overwrite_newmasks',  default=False, action='store_true',
                        help='Add this flag if you want previous post-processed masks to be overwritten')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)