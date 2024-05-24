#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Script to evaluate predictions against ground truth


This script uses output from evaluate_CMBlevel.py stored in specific temp/ folder 
as well as ground truth CMBs metadata to compute evaluation metrics. It provides
confidence intervals and std of metrics.



Three evaluations are possible:
- classification (has some CMB or not)
- segmentation (pixel-wise evaluation)
- detection -> at connected component level


Two ways of matching CMBs are possible:
- distance-based
- overlap-based


Performance can be split by metadata groups in input data at:
- study level
- CMB level


@author: jorgedelpozolerida
@date: 22/05/2024
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
import pickle

# Utils
import cmbnet.utils.utils_plotting as utils_plotting
import cmbnet.utils.utils_general as utils_general
import cmbnet.utils.utils_evaluation as utils_eval

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def get_predictions_df(pred_metadata_dir):
    """
    Load into dict all metadata from predictions
    """
    all_studies = [
        f.replace("_evaluation.pkl", "") for f in os.listdir(os.path.join(pred_metadata_dir, "temp"))
    ]
    all_data = []
    for study in all_studies:
        file_path = os.path.join(pred_metadata_dir, "temp", f"{study}_evaluation.pkl")
        with open(file_path, "rb") as file:
            eval_result = pickle.load(file)

        for pred_dict in eval_result:
            all_data.append({"seriesUID": study, **pred_dict})

    all_pred_df = pd.DataFrame(all_data)
    return all_pred_df


def evaluate_from_dataframes(args, all_studies_df, GT_metadata_all, pred_metadata_df):
    """
    Evaluate using data form dataframes
    """

    ##############################################################################
    # Detection metrics: uses TP, FP, FN at CMB-level
    # also dice score per TP
    ##############################################################################
    (
        detection_metrics,
        segmentation_metrics,
        study_results_detection,
        study_results_segmentation,
    ) = utils_eval.evaluate_detection_from_cmb_data(
        all_studies_df,
        GT_metadata_all,
        pred_metadata_df,
        match_col="matched_GT_DistancesToAllCMs",
    )
    ##############################################################################
    # Classification metrics: uses TP, FP, FN, TN at study-level
    ##############################################################################
    thresholds = [1, 2, 5]
    all_classifications = []
    for th in thresholds:
        classification_metrics = utils_eval.evaluate_classification_from_cmb_data(
            all_studies_df,
            GT_metadata_all,
            pred_metadata_df,
            threshold=th,
            match_col="matched_GT_DistancesToAllCMs",
        )
        all_classifications.append({"threshold": th, **classification_metrics})
    classification_metrics = pd.DataFrame(all_classifications)

    return detection_metrics, segmentation_metrics, classification_metrics, study_results_detection, study_results_segmentation


def main(args):

    # Load metadata dfs and clean
    all_studies_df = pd.read_csv(args.all_studies_csv)
    GT_metadata = pd.read_csv(args.gt_cmb_metadata_csv)
    GT_metadata_radiomics = pd.read_csv(args.gt_radiomics_metadata_csv)
    pred_metadata_df = get_predictions_df(args.cmb_pred_metadata_dir)

   # Merge into one dataframe all GT cmb metadata
    GT_metadata["CM"] = GT_metadata["CM"].apply(lambda x: tuple(ast.literal_eval(x)))
    GT_metadata_radiomics["CM"] = GT_metadata_radiomics["CM"].apply(
        lambda x: tuple(ast.literal_eval(x))
    )
    GT_metadata_all = pd.merge(
        GT_metadata, GT_metadata_radiomics, on=["seriesUID", "CM"], how="inner"
    )
    
    if args.dataset == ["cmb_valid"]:
        args.datasets = ["VALDO", "MOMENI", "RODEJA"]
        # args.studies  = os.listdir("/storage/evo1/jorge/datasets/cmb/cmb_valid/Data") # HARCODED

    else:
        args.datasets = args.dataset

    if args.studies is None:
        # filter on GT df as it has only evaluation studies
        args.studies = GT_metadata_all[GT_metadata_all["Dataset"].isin(args.datasets)][
            "seriesUID"
        ].unique()
        
    # First filtering 
    all_studies_df = all_studies_df[all_studies_df["seriesUID"].isin(args.studies)]
    GT_metadata = GT_metadata[GT_metadata["seriesUID"].isin(args.studies)]
    GT_metadata_radiomics = GT_metadata_radiomics[
        GT_metadata_radiomics["seriesUID"].isin(args.studies)
    ]
    pred_metadata_df = pred_metadata_df[pred_metadata_df["seriesUID"].isin(args.studies)]
  
    # Check validity of data
    # jorge = GT_metadata[~(GT_metadata['seriesUID'].isin(GT_metadata_radiomics['seriesUID']) & GT_metadata['CM'].isin(GT_metadata_radiomics['CM']))]
    # print(jorge)
    assert len(GT_metadata) == len(
        GT_metadata_radiomics
    ), f"Different number of studies in GT metadata {len(GT_metadata)} and radiomics metadata {len(GT_metadata_radiomics)}"
    assert (
        len(all_studies_df) >= GT_metadata["seriesUID"].nunique()
    ), "Different number of studies in all studies metadata and GT metadata"

    
    # GT_metadata_all.to_csv(
    #     os.path.join(args.output_dir, "GT_metadata_all.csv"), index=False
    # )
    # all_studies_df.to_csv(
    #     os.path.join(args.output_dir, "all_studies_df.csv"), index=False
    # )
    # pred_metadata_df.to_csv(
    #     os.path.join(args.output_dir, "pred_metadata_df.csv"), index=False
    # )

    all_dfs = {
        "GT_metadata": GT_metadata,
        "GT_metadata_radiomics": GT_metadata_radiomics,
        "all_studies_df": all_studies_df,
        "pred_metadata_df": pred_metadata_df,
    }

    if args.split_column:
        assert args.split_df, "Split column provided but no split dataframe"
        df_selected  = all_dfs[args.split_df]
        unique_categories = df_selected[args.split_column].unique()
    else:
        unique_categories = [None]

    # Assertions
    assert set(args.datasets).issubset(
        set(GT_metadata["Dataset"].unique())
    ), "Some datasets are not present in GT metadata"
    assert set(args.studies).issubset(
        set(GT_metadata["seriesUID"].unique())
    ), "Some studies are not present in GT metadata"

    for category in unique_categories:
        
        # Filter for that group
        if category is not None:
            studies_category = df_selected[df_selected[args.split_column] == category]["seriesUID"]
            args.studies = studies_category

        current_time = datetime.now()
        current_datetime = current_time.strftime("%d%m%Y_%H%M%S")
        utils_general.ensure_directory_exists(args.output_dir)
        args.log_file_path = os.path.join(args.output_dir, f"log_{current_datetime}.txt")
        msg = f"Starting evaluation at {current_datetime}\n\n"
        if category:
            msg += f"Evaluating category {category}, filtered in dataframe {args.split_df}\n\n"
        msg += f"Selected {len(args.studies)} studies from dataset {args.dataset} to evaluate\n\n"
        _logger.info(msg)
        utils_general.write_to_log_file(msg, args.log_file_path)

        # Filter all to datasets
        GT_metadata_all_filt = GT_metadata_all[GT_metadata_all["seriesUID"].isin(args.studies)]
        all_studies_df_filt = all_studies_df[all_studies_df["seriesUID"].isin(args.studies)]
        pred_metadata_df_filt = pred_metadata_df[
            pred_metadata_df["seriesUID"].isin(args.studies)
        ]

        # Evaluate ----------------------------------------------------------------
        detection_metrics, segmentation_metrics, classification_metrics, study_results_detection, study_results_detection = (
            evaluate_from_dataframes(
                args, all_studies_df_filt, GT_metadata_all_filt, pred_metadata_df_filt
            )
        )
        print("\nSegmentation Metrics:")
        print(segmentation_metrics)
        print("\nClassification metrics:")
        print(classification_metrics)
        print("\nDetection metrics:")
        print(detection_metrics)
        
        utils_general.write_to_log_file("\nSegmentation Metrics:\n", args.log_file_path)
        utils_general.write_to_log_file(segmentation_metrics.to_string(), args.log_file_path)
        utils_general.write_to_log_file("\nClassification metrics:\n", args.log_file_path)
        utils_general.write_to_log_file(classification_metrics.to_string(), args.log_file_path)
        utils_general.write_to_log_file("\nDetection metrics:\n", args.log_file_path)
        utils_general.write_to_log_file(detection_metrics.to_string(), args.log_file_path)
        
        suffix = f"_{category}" if category else ""

        # Save results
        segmentation_metrics.reset_index(names=['Metric']).to_csv(
            os.path.join(args.output_dir, f"segmentation_metrics{suffix}.csv"), index=False
        )
        classification_metrics.to_csv(
            os.path.join(args.output_dir, f"classification_metrics{suffix}.csv"), index=False
        )
        detection_metrics.reset_index(names=['Metric']).to_csv(
            os.path.join(args.output_dir, f"detection_metrics{suffix}.csv"), index=False
        )
        
        # Save study-level results with pickle
        with open(os.path.join(args.output_dir, f"study_results_detection{suffix}.pkl"), "wb") as file:
            pickle.dump(study_results_detection, file)
        with open(os.path.join(args.output_dir, f"study_results_segmentation{suffix}.pkl"), "wb") as file:
            pickle.dump(study_results_detection, file)

        utils_general.write_to_log_file(
            f"Results saved in {args.output_dir}", args.log_file_path
        )
        utils_general.write_to_log_file(f"Finished evaluation", args.log_file_path)

def parse_args():
    """
    Parses all script arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        required=True,
        help="Full path to the directory where results and logs will be saved",
    )
    parser.add_argument(
        "--cmb_pred_metadata_dir",
        type=str,
        default=None,
        required=True,
        help="Full path to the folder with predicetd CMB metadata computed by evaluate_CMBlevel.py",
    )
    parser.add_argument(
        "--gt_radiomics_metadata_csv",
        type=str,
        default=None,
        required=True,
        help="Path to the CSV with GT CMB radiomics metadata",
    )
    parser.add_argument(
        "--gt_cmb_metadata_csv",
        type=str,
        default=None,
        required=True,
        help="Path to the CSV with CMB metadata for GT",
    )
    parser.add_argument(
        "--all_studies_csv",
        type=str,
        default=None,
        required=True,
        help="Path to the CSV with all study-level metadata",
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
        "--dataset",
        nargs="+",
        type=str,
        default=None,
        required=True,
        help="Specific studies to evaluate",
    )
    parser.add_argument(
        "--split_column", type=str, help="Column to split the analysis by", default=None
    )
    parser.add_argument(
        "--split_df", type=str,
        choices=['GT_metadata', 'GT_metadata_radiomics', 'all_studies_df', 'pred_metadata_df'],
        default=None,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
