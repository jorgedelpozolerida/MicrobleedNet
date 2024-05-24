#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Generate data at CMB-level for predicted CMBs


Output of this file is meant to be post-processed in separate script to compute 
evaluation.

@author: jorgedelpozolerida
@date: 20/05/2024
"""


import os
import sys
import argparse
import traceback
import csv

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

import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import nibabel as nib
import multiprocessing
import time
import json
from datetime import datetime
from functools import partial
import sys
import ast

import pickle
import os

# Utils
import cmbnet.utils.utils_plotting as utils_plotting
import cmbnet.utils.utils_general as utils_general
import cmbnet.utils.utils_evaluation as utils_eval

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def evaluate_study_subject_level(args, subject_metadata, msg):
    """
    Computes evaluation for study
    """
    gt_nib = nib.load(subject_metadata["gt_path"])
    pred_nib = nib.load(subject_metadata["pred_path"])
    results = {}
    for eval_method in args.evaluations:
        results_m = utils_eval.compute_subject_level_evaluation(
            gt_nib.get_fdata(), np.squeeze(pred_nib.get_fdata()), eval_method
        )
        results.update(results_m)
    return results, msg


def evaluate_study_CMB_level(args, subject_metadata, cmb_metadata_study, msg):
    """
    Computes evaluation for study
    """
    mri_nib = nib.load(subject_metadata["mri_path"])
    gt_nib = nib.load(subject_metadata["gt_path"])
    pred_nib = nib.load(subject_metadata["pred_path"])
    synth_nib = nib.load(os.path.join(args.synthseg_dir, f"{subject_metadata['id']}_synthseg_resampled.nii.gz"))
    CMB_metadata = subject_metadata["CMBs_new"]

    assert len(cmb_metadata_study) == len(
        CMB_metadata
    ), f"Number of CMBs in metadata and in the mask do not match"

    # TODO: worflow for healthy scans somwhere

    predicted_CC_results, msg = utils_eval.get_predicted_CC_matches_and_metadata(
        mri_nib, gt_nib, pred_nib, synth_nib, cmb_metadata_study, msg
    )

    return predicted_CC_results, msg


def load_predictions(args):
    """
    Loads subjects metadata following clearml folder structure

    Returns list with dictionaries containing "id" and "pred_path"
    """
    pred_dir = args.predictions_dir
    if args.pred_dir_struct == "clearml":
        metadata = utils_general.load_clearml_predictions(pred_dir)
        return metadata
    elif args.pred_dir_struct == "post-process":
        metadata = [
            {"id": f.split("_")[0], "pred_path": os.path.join(pred_dir, f)}
            for f in os.listdir(pred_dir)
        ]
        return metadata
    else:
        raise NotImplementedError


def get_subjects_metadata(args):
    """
    Returns a list of dictionaries for studies present in predictions dir with
    "id", "gt_path" and "pred_path" keys.
    """

    id_and_preds_metadata = load_predictions(args)
    all_metadata = utils_general.add_groundtruth_metadata(
        args.groundtruth_dir, args.gt_dir_struct, id_and_preds_metadata
    )
    if args.cmb_metadata_csv is not None:
        cmb_metadata_df = pd.read_csv(args.cmb_metadata_csv)
        cmb_metadata_df = cmb_metadata_df[
            cmb_metadata_df["seriesUID"].isin([v["id"] for v in all_metadata])
        ]
        cmb_metadata_df["CM"] = (
            cmb_metadata_df["CM"]
            .apply(ast.literal_eval)
            .apply(lambda x: np.array(x, dtype=np.int32))
        )
        all_metadata = utils_general.add_CMB_metadata(cmb_metadata_df, all_metadata)
    return all_metadata


def add_groundtruth_metadata(args, metadata):
    """
    Adds ground truth metadata to dict for subjects present.
    This function should be adapted to varying folder structures.
    """
    subjects_selected = [s_item["id"] for s_item in metadata]

    if args.gt_dir_struct == "processed_final":
        load_func = utils_general.get_metadata_from_processed_final

    elif args.gt_dir_struct == "cmb_format":
        load_func = utils_general.get_metadata_from_cmb_format
    else:
        raise NotImplementedError

    gt_metadata = {}

    for sub in subjects_selected:
        sub_meta = load_func(args.groundtruth_dir, sub)
        gt_metadata[sub] = sub_meta
    for meta_item in metadata:
        matching_item = gt_metadata[meta_item["id"]]
        meta_item.update({"gt_path": matching_item["anno_path"], **matching_item})

    return metadata



def save_individual_eval(args, evaluation_results, file_path):
    """
    Saves individual evaluation results to a file for later combination using pickle serialization.
    """
  
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Use pickle to write the results to a file
    with open(file_path, "wb") as file:
        pickle.dump(evaluation_results, file)

    return file_path  # Optionally return the file path for reference



def generate_CMB_calls_plots(args, subject, subject_metadata, evaluation_results, zoom_size=100):

    plots_path = utils_general.ensure_directory_exists(
        os.path.join(args.plot_path, subject)
    )

    # Load data
    gt_nib = nib.load(subject_metadata["anno_path"])
    pred_nib = nib.load(subject_metadata["pred_path"])
    mri_im = nib.load(subject_metadata["mri_path"])

    # Plots GT and predictions for all existing CMBs
    gt_cms = []
    for cmb_num, cmb_dict in subject_metadata["CMBs_new"].items():
        cm = tuple(cmb_dict["CM"])
        gt_cms.append(cm)
        filename_temp = os.path.join(plots_path, f"CMB-{cmb_num}_{str(cm)}___{subject}.png")
        utils_plotting.plot_processed_mask_3x3(
            mri_im,
            gt_nib,
            pred_nib,
            cm,
            zoom_size,
            metadata_str="",
            save_path=filename_temp,
        )
        filename_temp = os.path.join(plots_path, f"CMB-{cmb_num}_{str(cm)}___{subject}_BRAIN.png")
        utils_plotting.plot_brain(
            mri_im,
            gt_nib,
            pred_nib,
            cm,
            zoom_size,
            metadata_str="",
            save_path=filename_temp,
        )
     
    matched_CMs = []
    for pred_item in evaluation_results:
        if pred_item['matched_GT_OverlapCMCounts'] is None:
            fp_cm = tuple(pred_item['pred_CM'])
            # FP
            filename_temp = os.path.join(plots_path, f"FP-{str(fp_cm)}___{subject}.png")
            utils_plotting.plot_processed_mask_3x3(
                mri_im,
                gt_nib,
                pred_nib,
                fp_cm,
                zoom_size,
                metadata_str="",
                save_path=filename_temp,
            )
            filename_temp = os.path.join(plots_path, f"FP-{str(fp_cm)}___{subject}_BRAIN.png")
            utils_plotting.plot_brain(
                mri_im,
                gt_nib,
                pred_nib,
                fp_cm,
                zoom_size,
                metadata_str="",
                save_path=filename_temp,
            )

        else:
            matched_CMs.append(tuple(pred_item['matched_GT_OverlapCMCounts']))       
    # FNs
    unmatched_gts = set(gt_cms) - set(matched_CMs)
    for com_FN in unmatched_gts:
        filename_temp = os.path.join(
            plots_path, f"FN___CMB-{com_FN}___{subject}.png"
        )
        utils_plotting.plot_processed_mask_3x3(
            mri_im,
            gt_nib,
            pred_nib,
            com_FN,
            zoom_size,
            metadata_str="",
            save_path=filename_temp,
        )

    # Plots false calls -------------------------

    

def save_evaluation_results_as_csv(evaluation_results, filename):
    """
    Saves the evaluation results in a CSV file, serializing complex structures to JSON strings.
    
    Args:
        evaluation_results (list of dict): The evaluation results to save.
        filename (str): Path to the file where the results should be saved.
    """
    with open(filename, mode='w', newline='') as file:
        fieldnames = evaluation_results[0].keys()  # Assumes all dicts have the same structure
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in evaluation_results:
            # Serialize each value that is a dictionary to a JSON string
            serialized_result = {key: json.dumps(value) if isinstance(value, dict) else value
                                 for key, value in result.items()}
            writer.writerow(serialized_result)

def process_study(args, subject_metadata, cmb_metadata, msg=""):
    """
    Evaluates a given study (subject) by comparing its ground truth with predicted mask
    """
    # Initialize
    start = time.time()
    studyuid = subject_metadata["id"]

    msg = f"Started evaluating {studyuid}...\n\n"

    # Construct the file path
    file_path = os.path.join(args.savedir, "temp", f"{studyuid}_evaluation.pkl")
    if os.path.exists(file_path) or args.overwrite:
        msg += f"\tResults already exist for {studyuid}!!!\n"
        # Finalize
        end = time.time()
        msg += f"Finished evaluation of {studyuid} in {end - start} seconds!\n\n"
        utils_general.write_to_log_file(msg, args.log_file_path)

        return

    try:
        cmb_metadata_study = cmb_metadata[cmb_metadata["seriesUID"] == studyuid]
        if not len(cmb_metadata_study) > 0:
            msg += f"\tNo CMB metadata found for {studyuid}\n"
        evaluation_results, msg = evaluate_study_CMB_level(
            args, subject_metadata, cmb_metadata_study, msg
        )
        save_individual_eval(args, evaluation_results, file_path)
        msg += f"Results saved to {file_path}\n"
        # TODO: investigate effect of this when cmb mode is on
        if args.create_plots:
            # Create plots
            generate_CMB_calls_plots(args, studyuid, subject_metadata, evaluation_results)

    except Exception:
        msg += f"Failed to process {studyuid}!\n\nException caught: {traceback.format_exc()}"

    # Finalize
    end = time.time()
    msg += f"Finished evaluation of {studyuid} in {end - start} seconds!\n\n"
    utils_general.write_to_log_file(msg, args.log_file_path)


def main(args):

    current_time = datetime.now()
    current_datetime = current_time.strftime("%d%m%Y_%H%M%S")
    utils_general.ensure_directory_exists(args.savedir)
    args.log_file_path = os.path.join(args.savedir, f"log_{current_datetime}.txt")
    args.plot_path = utils_general.ensure_directory_exists(
        os.path.join(args.savedir, "plots")
    )

    # Get subject list and metadata
    subjects_metadata = get_subjects_metadata(args)
    msg = f"Found predictions and ground truth for a total of {len(subjects_metadata)} studies\n\n"
    if args.studies is not None:
        subjects_metadata = [s for s in subjects_metadata if s["id"] in args.studies]
        msg += f"Selected {len(subjects_metadata)} studies to evaluate\n\n"

    # Get CMB metadata
    cmb_metadata = pd.read_csv(args.cmb_metadata_csv)
    cmb_metadata = cmb_metadata[
        cmb_metadata["seriesUID"].isin([v["id"] for v in subjects_metadata])
    ]

    _logger.info(msg)
    utils_general.write_to_log_file(msg, args.log_file_path)

    # Create necessary dirs
    utils_general.ensure_directory_exists(os.path.join(args.savedir, "temp"))

    # Determine number of worker processes
    available_cpu_count = multiprocessing.cpu_count()
    num_workers = min(args.num_workers, available_cpu_count)

    if num_workers == 1:
        for sub_meta in tqdm(subjects_metadata):
            process_study(args, sub_meta, cmb_metadata, msg="")
    else:
        # Parallelizing using multiprocessing
        with multiprocessing.Pool(processes=num_workers) as pool:
            list(
                tqdm(
                    pool.imap(
                        partial(process_study, args), subjects_metadata, cmb_metadata
                    ),
                    total=len(subjects_metadata),
                )
            )

    msg = f"Succesfully evaluated on all cases\n\n"
    utils_general.write_to_log_file(msg, args.log_file_path)


def parse_args():
    """
    Parses all script arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--savedir",
        type=str,
        default=None,
        help="Full path to the directory where results and logs will be saved",
    )
    parser.add_argument(
        "--groundtruth_dir",
        type=str,
        default=None,
        help="Path to the directory with GT masks saved",
    )
    parser.add_argument(
        "--gt_dir_struct",
        type=str,
        default="cmb_format",
        choices=["processed_final", "cmb_format"],
        help="Type of structure for saved ground truth masks",
    )
    parser.add_argument(
        "--predictions_dir",
        type=str,
        default=None,
        help="Path to the directory with predictions",
    )
    parser.add_argument(
        "--pred_dir_struct",
        type=str,
        default="clearml",
        choices=["clearml", "post-process"],
        help="Type of structure for saved predictions",
    )
    parser.add_argument(
        "--synthseg_dir",
        required=True,
        help="Directory where all SynthSeg RESAMPLED masks are saved.",
    )
    parser.add_argument(
        "--evaluations",
        nargs="+",
        type=str,
        default=["segmentation", "detection"],
        help="Evaluation types to run.",
    )
    parser.add_argument(
        "--studies",
        nargs="+",
        type=str,
        default=None,
        help="Specific studies to evaluate",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=5,
        help="Number of workers running in parallel",
    )
    parser.add_argument(
        "--cmb_metadata_csv",
        type=str,
        default=None,
        required=True,
        help="Full path to the CSV with CMB metadata with seriesUID-CM id pair",
    )
    # FLAGS
    parser.add_argument(
        "--create_plots",
        default=False,
        action="store_true",
        help="Add this flag if you want to create plots for CMBs",
    )
    parser.add_argument(
        "--overwrite",
        default=False,
        action="store_true",
        help="Add this flag if you want to overwrite existing results",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
