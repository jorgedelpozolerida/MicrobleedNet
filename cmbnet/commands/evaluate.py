#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Script to evaluate predictions against ground truth

NOTE:
    - All studies in prediction dir must be in ground truth dir but not the other way around


Three evaluations are made:
- classification (has some CMB or not)
- segmentation (pixel-wise evaluation)
- detection (connected components evaluation)


TODO:
- Distance based evaluation
- Allow to split performance by group

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

def evaluate_study_CMB_level(args, subject_metadata, msg):
    """
    Computes evaluation for study
    """
    gt_nib = nib.load(subject_metadata["gt_path"])
    pred_nib = nib.load(subject_metadata["pred_path"])
    CMB_metadata = subject_metadata["CMBs_new"]

    raise NotImplementedError
    # TODO: finish CMB-level eval
    results = utils_eval.compute_CMB_level_evaluation(
            gt_nib.get_fdata(), np.squeeze(pred_nib.get_fdata()), args.evaluations, CMB_metadata
        )
    print(f"Results for {subject_metadata['id']}:")
    print(results)
    return results, msg

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
    all_metadata = utils_general.add_groundtruth_metadata(args.groundtruth_dir, args.gt_dir_struct, id_and_preds_metadata)
    if args.cmb_metadata_csv is not None:
        cmb_metadata_df = pd.read_csv(args.cmb_metadata_csv)
        cmb_metadata_df= cmb_metadata_df[cmb_metadata_df['seriesUID'].isin([v['id'] for v in all_metadata])]
        cmb_metadata_df['CM'] = cmb_metadata_df['CM'].apply(ast.literal_eval).apply(lambda x: np.array(x, dtype=np.int32))
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


def save_individual_eval(args, evaluation_results, studyuid):
    """
    Saves individual evaluation results to a file for later combination.
    """
    # Construct the file path
    file_path = os.path.join(args.output_dir, "temp", f"{studyuid}_evaluation.json")

    # Combine the results and any additional information
    data_to_save = {"study_id": studyuid, "evaluation_results": evaluation_results}

    # Write the results to a file
    with open(file_path, "w") as file:
        json.dump(data_to_save, file, indent=4)

    return file_path  # Optionally return the file path for reference


def generate_CMB_calls_plots(args, subject, subject_metadata, zoom_size=100):

    plots_path = utils_general.ensure_directory_exists(
        os.path.join(args.plot_path, subject)
    )

    # Load data
    gt_nib = nib.load(subject_metadata["anno_path"])
    gt_nib_data = gt_nib.get_fdata().astype(np.uint8)
    pred_nib = nib.load(subject_metadata["pred_path"])
    pred_data = np.squeeze(pred_nib.get_fdata()).astype(np.uint8)
    mri_im = nib.load(subject_metadata["mri_path"])

    # Plots GT and predictions for all existing CMBs
    for cmb_num, cmb_dict in subject_metadata["CMBs_new"].items():
        cm = tuple(cmb_dict["CM"])
        filename_temp = os.path.join(plots_path, f"CMB-{cmb_num}___{subject}.png")
        utils_plotting.plot_processed_mask_3x3(
            mri_im,
            gt_nib,
            pred_nib,
            cm,
            zoom_size,
            metadata_str="",
            save_path=filename_temp,
        )
        filename_temp = os.path.join(plots_path, f"CMB-{cmb_num}___{subject}_BRAIN.png")
        utils_plotting.plot_brain(
            mri_im,
            gt_nib,
            pred_nib,
            cm,
            zoom_size,
            metadata_str="",
            save_path=filename_temp,
        )
        # Check for FN
        if gt_nib_data[cm] == 1 and pred_data[cm] == 0:
            filename_temp = os.path.join(
                plots_path, f"FN___CMB-{cmb_num}___{subject}.png"
            )
            utils_plotting.plot_processed_mask_3x3(
                mri_im,
                gt_nib,
                pred_nib,
                cm,
                zoom_size,
                metadata_str="",
                save_path=filename_temp,
            )

    # Plots false calls -------------------------

    # FP
    FP_list = utils_plotting.get_FP_coords(gt_nib.get_fdata(), pred_nib.get_fdata())
    for cmb_num, cm in enumerate(FP_list):
        filename_temp = os.path.join(plots_path, f"FP-{cmb_num}___{subject}.png")
        utils_plotting.plot_processed_mask_3x3(
            mri_im,
            gt_nib,
            pred_nib,
            cm,
            zoom_size,
            metadata_str="",
            save_path=filename_temp,
        )
        filename_temp = os.path.join(plots_path, f"FP-{cmb_num}___{subject}_BRAIN.png")
        utils_plotting.plot_brain(
            mri_im,
            gt_nib,
            pred_nib,
            cm,
            zoom_size,
            metadata_str="",
            save_path=filename_temp,
        )


def process_study(args, subject_metadata, msg=""):
    """
    Evaluates a given study (subject) by comparing its ground truth with predicted mask
    """
    # Initialize
    start = time.time()
    studyuid = subject_metadata["id"]
    
    
    msg = f"Started evaluating {studyuid}...\n\n"

    try:
        if args.eval_mode == "per-CMB":
            evaluation_results, msg = evaluate_study_CMB_level(args, subject_metadata, msg)
        elif args.eval_mode == "per-study":
            evaluation_results, msg = evaluate_study_subject_level(args, subject_metadata, msg)
        else:
            raise NotImplementedError
        msg += f"\tEvaluation results: \n"
        for k, v in evaluation_results.items():
            msg += f"\t\t{k}:  {v}\n"
        file_path = save_individual_eval(args, evaluation_results, studyuid)
        msg += f"Results saved to {file_path}\n"

        # TODO: investigate effect of this when cmb mode is on
        if args.create_plots and args.eval_mode == "per-study":
            # Create plots
            generate_CMB_calls_plots(args, studyuid, subject_metadata)

    except Exception:
        msg += f"Failed to process {studyuid}!\n\nException caught: {traceback.format_exc()}"

    # Finalize
    end = time.time()
    msg += f"Finished evaluation of {studyuid} in {end - start} seconds!\n\n"
    utils_general.write_to_log_file(msg, args.log_file_path)


def combine_evaluations(args):
    all_results = []

    # Read individual results
    for file in os.listdir(os.path.join(args.output_dir, "temp")):
        if file.endswith("_evaluation.json"):
            file_path = os.path.join(args.output_dir, "temp", file)
            with open(file_path, "r") as f:
                data = json.load(f)
                checkifresultskey = any(["results" in d for d in data["evaluation_results"].keys()])
                if checkifresultskey:
                    all_results.append({
                        "study_id": data["study_id"],
                        **{k.replace("_results", ""):val for k,val in data["evaluation_results"].items()}
                    })
                else:
                    for cmb_id, cmb_results in data["evaluation_results"].items():
                        print(cmb_results)
                        all_results.append({
                            "study_id": data["study_id"],
                            "cmb_id": cmb_id,
                            "CM": cmb_results['CM'],
                            **{k.replace("_results", ""):val for k,val in cmb_results.items() if k != "CM"}
                        })

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    return df


def main(args):

    current_time = datetime.now()
    current_datetime = current_time.strftime("%d%m%Y_%H%M%S")
    utils_general.ensure_directory_exists(args.output_dir)
    args.log_file_path = os.path.join(args.output_dir, f"log_{current_datetime}.txt")
    args.plot_path = utils_general.ensure_directory_exists(
        os.path.join(args.output_dir, "plots")
    )

    # Handle evaluation mode
    if args.cmb_metadata_csv is not None:
        assert args.cmb_metadata_columns is not None
        args.eval_mode = "per-CMB"
    else:
        args.eval_mode = "per-study"
    msg = f"EVALUATION MODE: {args.eval_mode}\n\n"


    # Get subject list and metadata
    subjects_metadata = get_subjects_metadata(args)
    msg += f"Found predictions and ground truth for a total of {len(subjects_metadata)} studies\n\n"
    if args.studies is not None:
        subjects_metadata = [s for s in subjects_metadata if s["id"] in args.studies]
        msg += f"Selected {len(subjects_metadata)} studies to evaluate\n\n"


    _logger.info(msg)
    utils_general.write_to_log_file(msg, args.log_file_path)

    if args.combined_results_csv is None:
        # Create necessary dirs
        utils_general.ensure_directory_exists(os.path.join(args.output_dir, "temp"))

        # Determine number of worker processes
        available_cpu_count = multiprocessing.cpu_count()
        num_workers = min(args.num_workers, available_cpu_count)

        if num_workers == 1:
            for sub_meta in tqdm(subjects_metadata):
                process_study(args, sub_meta, msg="")
        else:
            # Parallelizing using multiprocessing
            with multiprocessing.Pool(processes=num_workers) as pool:
                list(
                    tqdm(
                        pool.imap(partial(process_study, args), subjects_metadata),
                        total=len(subjects_metadata),
                    )
                )

        msg = f"Succesfully evaluated on all cases\n\n"
        utils_general.write_to_log_file(msg, args.log_file_path)

        # Combine results
        combined_df = combine_evaluations(args)
        combined_df_file = os.path.join(args.output_dir, "combined_evaluation_results.csv")
        combined_df.to_csv(combined_df_file, index=False)
    else:
        combined_df = pd.read_csv(args.combined_results_csv)
        combined_df['detection'] = combined_df['detection'].apply(ast.literal_eval)
        combined_df['segmentation'] = combined_df['segmentation'].apply(ast.literal_eval)
        msg = f"Loaded evaluations from pre-existing CSV:\n{args.combined_results_csv}\n\n"
        utils_general.write_to_log_file(msg, args.log_file_path)

    # Filter studies
    n_before = len(combined_df)
    selected_studies  = args.studies if args.studies is not None else combined_df['study_id'].to_list()
    combined_df = combined_df[combined_df['study_id'].isin(selected_studies)]
    msg = f"Filtered from {n_before} to {len(combined_df)} studies \n\n"
    utils_general.write_to_log_file(msg, args.log_file_path, True)

    # Study-level metadata split
    if args.metadata_csv is not None:
        metadata_df = pd.read_csv(args.metadata_csv)

    if "segmentation" in args.evaluations:
        macro_metrics, micro_metrics = utils_eval.combine_evaluate_segmentation(
            combined_df
        )

        # Print macroaveraged and microaveraged metrics
        print("\nSegmentation Metrics - MACRO:")
        print(macro_metrics)
        print("\nSegmentation Metrics - micro:")
        print(micro_metrics)

        # Write metrics to log file and save to CSV
        utils_general.write_to_log_file(macro_metrics, args.log_file_path)
        utils_general.write_to_log_file(micro_metrics, args.log_file_path)

        # Save these metrics to CSV files
        macro_metrics.to_csv(
            os.path.join(args.output_dir, "macro_segmentation_metrics.csv"), index=False
        )
        micro_metrics.to_csv(
            os.path.join(args.output_dir, "micro_segmentation_metrics.csv"), index=False
        )

    if "classification" in args.evaluations:
        classification_metrics = utils_eval.combine_evaluate_classification(combined_df)
        print("\nClassification metrics:")
        print(classification_metrics)
        utils_general.write_to_log_file(classification_metrics, args.log_file_path)
        # save these metrics to CSV files
        classification_metrics.to_csv(
            os.path.join(args.output_dir, "classification_metrics.csv"), index=False
        )

    if "detection" in args.evaluations:
        detection_macro_metrics, detection_micro_metrics, detection_totals_metrics = (
            utils_eval.combine_evaluate_detection(combined_df)
        )

        print("\nDetection metrics - MACRO:")
        print(detection_macro_metrics)
        print("\nDetection metrics - micro:")
        print(detection_micro_metrics)
        print("\nDetection metrics - TOTALS:")
        print(detection_totals_metrics)

        utils_general.write_to_log_file(detection_macro_metrics, args.log_file_path)
        utils_general.write_to_log_file(detection_micro_metrics, args.log_file_path)
        utils_general.write_to_log_file(detection_totals_metrics, args.log_file_path)

        # save these metrics to CSV files
        detection_macro_metrics.to_csv(
            os.path.join(args.output_dir, "detection_macro_metrics.csv"), index=False
        )
        detection_micro_metrics.to_csv(
            os.path.join(args.output_dir, "detection_micro_metrics.csv"), index=False
        )
        detection_totals_metrics.to_csv(
            os.path.join(args.output_dir, "detection_totals_metrics.csv"), index=False
        )

    print("Finished evaluation, find results in ", args.output_dir)


def parse_args():
    """
    Parses all script arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_dir",
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
        default=None,
        choices=["clearml", "post-process"],
        help="Type of structure for saved predictions",
    )
    parser.add_argument(
        "--evaluations",
        nargs="+",
        type=str,
        default=["segmentation", "classification", "detection"],
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
        "--combined_results_csv",
        type=str,
        default=None,
        help="Full path to the CSV with all results combined. If this provided evaluation results are not computed and taken from this file",
    )
    parser.add_argument(
        "--metadata_csv",
        type=str,
        default=None,
        help="Full path to the CSV with metadata to split evaluation by",
    )
    parser.add_argument(
        "--metadata_column",
        type=str,
        default=None,
        help="Column name from metadata CSV to use to split evaluation results",
    )
    parser.add_argument(
        "--cmb_metadata_csv",
        type=str,
        default=None,
        help="Full path to the CSV with CMB metadata to split evaluation by",
    )
    parser.add_argument(
        "--cmb_metadata_columns",
        nargs="+",
        type=str,
        default=None,
        help="Column name from CMB metadata CSV to use to split evaluation results",
    )
    # FLAGS
    parser.add_argument(
        "--create_plots",
        default=False,
        action="store_true",
        help="Add this flag if you want to create plots for CMBs",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
