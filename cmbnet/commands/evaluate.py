#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Script to evaluate predictions against ground truth

NOTE:
    - All studies in prediction dir must be in ground truth dir but not the other way around


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

import multiprocessing
from tqdm import tqdm
from functools import partial
import datetime

import os
import argparse
import traceback
import glob

import logging                                                                      
import numpy as np                                                                  
import pandas as pd                                                                 
from tqdm import tqdm
import nibabel as nib
import multiprocessing
import time 
from nilearn.image import resample_to_img, resample_img
from scipy.ndimage import generate_binary_structure, binary_closing, binary_dilation
from scipy.ndimage import label as nd_label
import json
from skimage.measure import label
from skimage.filters import threshold_otsu
from datetime import datetime
from functools import partial
import sys

# Utils
import cmbnet.utils.utils_datasets.utils_datasets as utils_datasets
import cmbnet.preprocessing.process_masks as utils_process
import utils.utils_general as utils_general
import utils.utils_evaluation as utils_eval

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

def evaluate_study(args, subject_metadata, msg):
    """
    Computes evaluation for study
    """
    gt_nib = nib.load(subject_metadata['gt_path'])
    pred_nib = nib.load(subject_metadata['pred_path'])
    results = {}
    for eval_method in args.evaluations:
        results_m = utils_eval.compute_individual_evaluation(gt_nib.get_fdata(), np.squeeze(pred_nib.get_fdata()), eval_method)
        results.update(results_m)
    return results


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
    elif args.pred_dir_struct == "post-process":
        metadata = [
            {
                    "id": f.split("_")[0],
                    "pred_path": os.path.join(pred_dir, f)
                } for f in os.listdir(pred_dir)
        ]
        return metadata
    else:
        raise NotImplementedError

def add_groundtruth_metadata(args, metadata):
    """
    Adds ground truth metadata to dict for subjects present. 
    This function should be adapted to varying folder structures.
    """
    if args.gt_dir_struct == "processed":
        gt_metadata = utils_datasets.get_files_metadata_from_processed(args.groundtruth_dir, [s_item['id'] for s_item in metadata])
        for meta_item in metadata:
            matching_item = [i for i in gt_metadata if i['id'] == meta_item['id']][0]
            meta_item.update({"gt_path": matching_item['anno_path'], "sequence": matching_item['seq_type']})
            
    elif args.gt_dir_struct == "post-processed":
        raise NotImplementedError
    else:
        raise NotImplementedError

    return metadata

def save_individual_eval(args, evaluation_results, studyuid):
    """
    Saves individual evaluation results to a file for later combination.
    """
    # Construct the file path
    file_path = os.path.join(args.output_dir, "temp" , f"{studyuid}_evaluation.json")

    # Combine the results and any additional information
    data_to_save = {
        "study_id": studyuid,
        "evaluation_results": evaluation_results
    }

    # Write the results to a file
    with open(file_path, 'w') as file:
        json.dump(data_to_save, file, indent=4)

    return file_path  # Optionally return the file path for reference


def process_study(args, subject_metadata, msg=''):
    """
    Evaluates a given study (subject) by comparing its ground truth with predicted mask
    """
    # Initialize
    start = time.time()
    studyuid = subject_metadata['id']
    msg = f'Started evaluating {studyuid}...\n\n'

    try:
        evaluation_results = evaluate_study(args, subject_metadata, msg)
        msg += f"\tEvaluation results: \n"
        for k, v in evaluation_results.items():
            msg += f"\t\t{k}:  {v}\n"
        file_path = save_individual_eval(args, evaluation_results, studyuid)
        msg += f"Results saved to {file_path}\n"
    except Exception:
        msg += f'Failed to process {studyuid}!\n\nException caught: {traceback.format_exc()}'
    
    # Finalize
    end = time.time()
    msg += f'Finished evaluation of {studyuid} in {end - start} seconds!\n\n'
    utils_general.write_to_log_file(msg, args.log_file_path)

def get_subjects_metadata(args):
    """ 
    Returns a list of dictionaries for studies present in predictions dir with
    "id", "gt_path" and "pred_path" keys.
    """
    
    id_and_preds_metadata = load_clearml_predictions(args)
    all_metadata = add_groundtruth_metadata(args, id_and_preds_metadata)

    return all_metadata 

def combine_evaluations(args):
    all_results = []

    # Read individual results
    for file in os.listdir(os.path.join(args.output_dir, "temp")):
        if file.endswith('_evaluation.json'):
            file_path = os.path.join(args.output_dir, "temp", file)
            with open(file_path, 'r') as f:
                data = json.load(f)
                data['evaluation_results']['study_id'] = data['study_id']
                all_results.append(data['evaluation_results'])

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    return df


def main(args):

    current_time = datetime.now()
    current_datetime = current_time.strftime("%d%m%Y_%H%M%S")
    utils_general.ensure_directory_exists(args.output_dir)
    args.log_file_path = os.path.join(args.output_dir, f'log_{current_datetime}.txt')

    # Get subject list
    subjects_metadata = get_subjects_metadata(args)

    msg = f"Succesfully found predictions and ground truth for a total of {len(subjects_metadata)} studies\n\n"
    _logger.info(msg)
    utils_general.write_to_log_file(msg, args.log_file_path)

    # Create necessary dirs
    utils_general.ensure_directory_exists(os.path.join(args.output_dir, "temp"))

    # Determine number of worker processes
    available_cpu_count = multiprocessing.cpu_count()
    num_workers = min(args.num_workers, available_cpu_count)

    # for sub_meta in tqdm(subjects_metadata):
    #     process_study(args, sub_meta, msg="")
    # Parallelizing using multiprocessing
    with multiprocessing.Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap(partial(process_study, args), subjects_metadata), total=len(subjects_metadata)))

    msg = f"Succesfully evaluated on all cases\n\n"
    utils_general.write_to_log_file(msg, args.log_file_path)

    # Combine results
    combined_df = combine_evaluations(args)
    combined_df_file = os.path.join(args.output_dir, "combined_evaluation_results.csv")
    combined_df.to_csv(combined_df_file, index=False)


    # Calculate metrics for each type
    detection_metrics = utils_eval.combine_evaluate_detection(combined_df)
    print(detection_metrics)
    utils_general.write_to_log_file(detection_metrics, args.log_file_path)

    classification_metrics = utils_eval.combine_evaluate_classification(combined_df)
    print(classification_metrics)
    utils_general.write_to_log_file(classification_metrics, args.log_file_path)

    # Calculate segmentation metrics
    segmentation_metrics = utils_eval.combine_evaluate_segmentation(combined_df)
    print(segmentation_metrics)
    utils_general.write_to_log_file(segmentation_metrics, args.log_file_path)

    # save these metrics to CSV files
    detection_metrics.to_csv(os.path.join(args.output_dir, 'detection_metrics.csv'), index=False)
    classification_metrics.to_csv(os.path.join(args.output_dir, 'classification_metrics.csv'), index=False)
    segmentation_metrics.to_csv(os.path.join(args.output_dir, 'segmentation_metrics.csv'), index=False)




def parse_args():
    '''
    Parses all script arguments.
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_dir', type=str, default=None,
                        help='Full path to the directory where results and logs will be saved')
    parser.add_argument('--groundtruth_dir', type=str, default=None,
                        help='Path to the directory with GT masks saved')
    parser.add_argument('--gt_dir_struct', type=str, default=None, choices= ['processed', 'post-processed'],
                        help='Type of structure for saved ground truth masks')
    parser.add_argument('--predictions_dir', type=str, default=None,
                        help='Path to the directory with predictions')
    parser.add_argument('--pred_dir_struct', type=str, default=None, choices= ['clearml', 'post-process'],
                        help='Type of structure for saved predictions')
    parser.add_argument('--evaluations', nargs='+', type=str,
                        default=['segmentation', 'classification', 'detection'],
                        help='Evaluation types to run.')
    parser.add_argument('--num_workers', type=int, default=5,
                            help='Number of workers running in parallel')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)