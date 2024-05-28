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
from cmbnet.utils.utils_evaluation import BRAIN_LABELS

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def get_predictions_df(pred_metadata_dir):
    """
    Load into dict all metadata from predictions
    """
    all_studies = [
        f.replace("_evaluation.pkl", "")
        for f in os.listdir(os.path.join(pred_metadata_dir, "temp"))
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
        all_cmbs_tracking
    ) = utils_eval.evaluate_detection_and_segment_from_cmb_data(
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

    return (
        detection_metrics,
        segmentation_metrics,
        classification_metrics,
        study_results_detection,
        study_results_segmentation,
        all_cmbs_tracking
    )


def load_and_prepare_data(args):
    all_studies_df = pd.read_csv(args.all_studies_csv)
    GT_metadata = pd.read_csv(args.gt_cmb_metadata_csv)
    GT_metadata_radiomics = pd.read_csv(args.gt_radiomics_metadata_csv)
    pred_metadata_df = get_predictions_df(args.cmb_pred_metadata_dir)

    # Compute extra metrics for pred
    pred_metadata_df['radius'] = pred_metadata_df['n_voxels'].apply(
        lambda x: ((x * (0.5**3)) / (4 / 3 * np.pi)) ** (1 / 3)
    )
    pred_metadata_df = utils_eval.add_location(pred_metadata_df, Isdfloaded=True) # add locations

    # Convert string representations of tuples to actual tuples
    GT_metadata["CM"] = GT_metadata["CM"].apply(lambda x: tuple(ast.literal_eval(x)))
    GT_metadata_radiomics["CM"] = GT_metadata_radiomics["CM"].apply(
        lambda x: tuple(ast.literal_eval(x))
    )

    GT_metadata_all = pd.merge(
        GT_metadata, GT_metadata_radiomics, on=["seriesUID", "CM"], how="inner"
    )
    GT_metadata_all = utils_eval.add_location(GT_metadata_all, Isdfloaded=False) # add locations
    
    
    

    return (
        all_studies_df,
        GT_metadata,
        GT_metadata_radiomics,
        pred_metadata_df,
        GT_metadata_all,
    )


def evaluate_group(
    output_dir,
    GT_metadata_all_filt,
    all_studies_df_filt,
    pred_metadata_df_filt,
    suffix="",
):
    # Logging
    current_time = datetime.now()
    current_datetime = current_time.strftime("%d%m%Y_%H%M%S")
    utils_general.ensure_directory_exists(output_dir)
    log_file_path = os.path.join(output_dir, f"log_{current_datetime}.txt")
    msg = f"Starting evaluation at {current_datetime}\n\n"
    _logger.info(msg)
    utils_general.write_to_log_file(msg, log_file_path)

    # Evaluate
    (
        detection_metrics,
        segmentation_metrics,
        classification_metrics,
        study_results_detection,
        study_results_segmentation,
        all_cmbs_tracking
    ) = evaluate_from_dataframes(
        args, all_studies_df_filt, GT_metadata_all_filt, pred_metadata_df_filt
    )

    print("\nSegmentation Metrics:")
    print(segmentation_metrics)
    print("\nClassification metrics:")
    print(classification_metrics)
    print("\nDetection metrics:")
    print(detection_metrics)

    utils_general.write_to_log_file("\nSegmentation Metrics:\n", log_file_path)
    utils_general.write_to_log_file(segmentation_metrics.to_string(), log_file_path)
    utils_general.write_to_log_file("\nClassification metrics:\n", log_file_path)
    utils_general.write_to_log_file(classification_metrics.to_string(), log_file_path)
    utils_general.write_to_log_file("\nDetection metrics:\n", log_file_path)
    utils_general.write_to_log_file(detection_metrics.to_string(), log_file_path)

    # Save results
    segmentation_metrics.reset_index(names=["Metric"]).to_csv(
        os.path.join(output_dir, f"segmentation_metrics{suffix}.csv"), index=False
    )
    classification_metrics.to_csv(
        os.path.join(output_dir, f"classification_metrics{suffix}.csv"), index=False
    )
    detection_metrics.reset_index(names=["Metric"]).to_csv(
        os.path.join(output_dir, f"detection_metrics{suffix}.csv"), index=False
    )

    # Save study-level results with pickle
    with open(
        os.path.join(output_dir, f"study_results_detection{suffix}.pkl"), "wb"
    ) as file:
        pickle.dump(study_results_detection, file)
    with open(
        os.path.join(output_dir, f"study_results_segmentation{suffix}.pkl"), "wb"
    ) as file:
        pickle.dump(study_results_segmentation, file)
        
    # Save cmb-level results with pickle
    with open(
        os.path.join(output_dir, f"all_cmbs_tracking{suffix}.pkl"), "wb"
    ) as file:
        pickle.dump(all_cmbs_tracking, file)

    utils_general.write_to_log_file(f"Results saved in {output_dir}", log_file_path)
    utils_general.write_to_log_file("Finished evaluation", log_file_path)


def main(args):

    # Load metadata dfs and clean
    (
        all_studies_df,
        GT_metadata,
        GT_metadata_radiomics,
        pred_metadata_df,
        GT_metadata_all,
    ) = load_and_prepare_data(args)

    if args.dataset == ["cmb_valid"]:
        args.datasets = ["VALDO", "MOMENI", "RODEJA"]
        cmb_valid_dir = "/storage/evo1/jorge/datasets/cmb/cmb_valid/Data"
        args.studies = [
            s
            for s in os.listdir(cmb_valid_dir)
            if os.path.isdir(os.path.join(cmb_valid_dir, s))
        ]  # HARCODED

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
    GT_metadata_all = GT_metadata_all[GT_metadata_all["seriesUID"].isin(args.studies)]
    pred_metadata_df = pred_metadata_df[
        pred_metadata_df["seriesUID"].isin(args.studies)
    ]

    # POST-PROCESSING 1: remove predicitons out-of-brain clearly
    pred_metadata_df = pred_metadata_df[
        pred_metadata_df['max_brain_key'].isin(BRAIN_LABELS) # filter out non-brain labels, for cases where no other brain label could be found
    ]

    # Check validity of data
    assert len(GT_metadata) == len(
        GT_metadata_radiomics
    ), f"Different number of studies in GT metadata {len(GT_metadata)} and radiomics metadata {len(GT_metadata_radiomics)}"
    assert (
        len(all_studies_df) >= GT_metadata["seriesUID"].nunique()
    ), "Different number of studies in all studies metadata and GT metadata"

    # Assertions
    assert set(args.datasets).issubset(
        set(GT_metadata["Dataset"].unique())
    ), "Some datasets are not present in GT metadata"

    missing_studies = [
        s for s in args.studies if s not in all_studies_df["seriesUID"].unique()
    ]
    print(f"Missing studies in all studies metadata: {missing_studies}")
    assert set(args.studies).issubset(
        set(all_studies_df["seriesUID"].unique())
    ), "Some studies are not present in GT metadata"

    ##############################################################################
    # Evaluations (comment out what not desired)
    ##############################################################################

    # All
    evaluate_group(
        args.output_dir,
        GT_metadata_all_filt=GT_metadata_all,
        all_studies_df_filt=all_studies_df,
        pred_metadata_df_filt=pred_metadata_df,
        suffix="",
    )

    # By Locations
    for loc in [
    'Cortex / grey-white junction ', 'Subcortical white matter', 
        'Basal ganglia grey matter', 'Thalamus', 'Brainstem', 'Cerebellum', 
    ]:
        print("---------------------------------------------------------------")
        print(f"Location: {loc}")
        print("---------------------------------------------------------------")

        utils_general.confirm_action()
        GT_metadata_all_location = GT_metadata_all[
                (
                    GT_metadata_all['BOMBS_label'] == loc
                )
            ]
        pred_metadata_df_location = pred_metadata_df[
                (
                pred_metadata_df['BOMBS_label'] == loc
                )
        ]
        print(f"GT metadata size before : {len(GT_metadata_all)}")
        print(f"GT metadata size after : {len(GT_metadata_all_location)}")
        print(f"Pred metadata size before : {len(pred_metadata_df)}")
        print(f"Pred metadata size after : {len(pred_metadata_df_location)}")
        
        evaluate_group(
            os.path.join(args.output_dir, loc.replace(" ", "").replace("/", "OR")),
            GT_metadata_all_filt=GT_metadata_all_location,
            all_studies_df_filt=all_studies_df,
            pred_metadata_df_filt=pred_metadata_df_location,
            suffix="",
        )

    # # Remove Anatomically impossible microbleeds, either too small or too big
    # GT_metadata_all_minimum_size = GT_metadata_all[
    #     (
    #         # (GT_metadata_all["shape_MeshVolume"] < 4.3)
    #         # | (GT_metadata_all["shape_Maximum3DDiameter"] > 10)
    #         # # remove nulls as these are tiny and thus have no radiomics data
    #         # | (GT_metadata_all["shape_MeshVolume"].isnull())
    #         # | (GT_metadata_all["shape_Maximum3DDiameter"].isnull())
    #         GT_metadata_all['radius'].between(0.5, 5)
    #     )
    # ]
    # pred_metadata_df_minimum_size = pred_metadata_df[
    #     (
    #         # (pred_metadata_df["shape_MeshVolume"] < 4.3)
    #         # | (pred_metadata_df["shape_Maximum3DDiameter"] > 10)
    #         # | (pred_metadata_df["shape_MeshVolume"].isnull())
    #         # | (pred_metadata_df["shape_Maximum3DDiameter"].isnull())
    #         pred_metadata_df['radius'].between(0.5, 5)
    #     )
    # ]
    # print(f"GT metadata size before : {len(GT_metadata_all)}")
    # print(f"GT metadata size after : {len(GT_metadata_all_minimum_size)}")
    # print(f"Pred metadata size before : {len(pred_metadata_df)}")
    # print(f"Pred metadata size after : {len(pred_metadata_df_minimum_size)}")

    # evaluate_group(
    #     os.path.join(args.output_dir, "minimum_size"),
    #     GT_metadata_all_filt=GT_metadata_all_minimum_size,
    #     all_studies_df_filt=all_studies_df,
    #     pred_metadata_df_filt=pred_metadata_df_minimum_size,
    #     suffix="",
    # )

    # GT_metadata_all_minimum_size = GT_metadata_all[
    #         (
    #             # (GT_metadata_all["shape_MeshVolume"] < 4.3)
    #             # | (GT_metadata_all["shape_Maximum3DDiameter"] > 10)
    #             # # remove nulls as these are tiny and thus have no radiomics data
    #             # | (GT_metadata_all["shape_MeshVolume"].isnull())
    #             # | (GT_metadata_all["shape_Maximum3DDiameter"].isnull())
    #             GT_metadata_all['radius'].between(0.5, 5)
    #         )
    #     ]
    # pred_metadata_df_minimum_size = pred_metadata_df[
    #         (
    #             # (pred_metadata_df["shape_MeshVolume"] < 4.3)
    #             # | (pred_metadata_df["shape_Maximum3DDiameter"] > 10)
    #             # | (pred_metadata_df["shape_MeshVolume"].isnull())
    #             # | (pred_metadata_df["shape_Maximum3DDiameter"].isnull())
    #             pred_metadata_df['radius'].between(0.5, 5)
    #         )
    # ]
    
    
    
    
    
    
    
    
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
