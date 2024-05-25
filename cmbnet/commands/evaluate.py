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

BRAIN_LABELS = set(
    [
        2,  # left cerebral white matter
        3,  # left cerebral cortex
        7,  # left cerebellum white matter
        8,  # left cerebellum cortex
        10,  # left thalamus
        11,  # left caudate
        12,  # left putamen
        13,  # left pallidum
        17,  # left hippocampus
        18,  # left amygdala
        26,  # left accumbens area
        28,  # left ventral DC (Diencephalon)
        41,  # right cerebral white matter
        42,  # right cerebral cortex
        46,  # right cerebellum white matter
        47,  # right cerebellum cortex
        49,  # right thalamus
        50,  # right caudate
        51,  # right putamen
        52,  # right pallidum
        53,  # right hippocampus
        54,  # right amygdala
        58,  # right accumbens area
        60,  # right ventral DC (Diencephalon)
    ]
)


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

    return (
        detection_metrics,
        segmentation_metrics,
        classification_metrics,
        study_results_detection,
        study_results_segmentation,
        all_cmbs_tracking
    )
    
def read_synthseg_labels(file_path):
    labels_dict = {}
    with open(file_path, "r") as file:
        # Skip header lines until we reach the line starting with 'labels'
        for line in file:
            if line.strip().lower().startswith("labels"):
                break

        # Process the label lines
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 2:
                label_num = int(parts[0])
                label_name = " ".join(parts[1:])
                labels_dict[label_num] = label_name
    return labels_dict
    
def add_location(df_location, pred=True):
    synth_labels  = read_synthseg_labels("/storage/evo1/jorge/MicrobleedNet/data-misc/cmb_analysis/labels table.txt")
    
    synthseg_mappings = pd.read_csv("/storage/evo1/jorge/MicrobleedNet/data-misc/cmb_analysis/synth_labels_mappedSilvia.csv")
    def get_max_key(x: pd.Series):
        
        x = {int(k):v for k, v in x.items()}
        
        # Check if there is any non-brain region
        x_filt = {k: v for k, v in x.items() if k in BRAIN_LABELS}
        if len(x_filt) > 0:
            x = x_filt
        if 0 in x and len(x) > 1:
            x.pop(0)
        if 24 in x and len(x) > 1:
            x.pop(24)

        # ignore the key with the value 0
        max_key = max(x, key=x.get) 
        
        # if is cortex but also white matter present, then choose white matter
        if max_key in [3, 42]:
            # get second max key
            if len(x) > 1:
                x.pop(max_key)
                try:
                    second_max_key = max(x, key=x.get)
                    if second_max_key in [2, 41]:
                        max_key = second_max_key
                except:
                    print(x)
        
        return max_key

    def get_percentages(x: pd.Series):
        
        x = {int(k):v for k, v in x.items()}
        
        if len(x) > 1:
            x.pop(0)
        # get percentages of total
        total = sum(x.values())
        x = {k: v/total for k, v in x.items()}
        return x

    def get_synthlabel_names(x: pd.Series, synth_labels):
        
        x = {int(k):v for k, v in x.items()}
        x.pop(0)
        x = {synth_labels[int(k)]: v for k, v in x.items()} 
        return x

    if not pred:
        df_location['count_dict'] = df_location['count_dict'].apply(lambda x: ast.literal_eval(x))
    df_location['counts_names'] = df_location['count_dict'].apply(get_synthlabel_names, args=(synth_labels,))
    df_location['percentages_name'] = df_location['count_dict'].apply(get_percentages)
    df_location['max_key'] = df_location['count_dict'].apply(get_max_key)
    df_location['label'] = df_location['max_key'].astype(int).apply(lambda x: synth_labels[x])
    df_location['BOMBS_label'] = df_location['label'].apply(lambda x: synthseg_mappings[synthseg_mappings['labels'] == x]['BOMBS'].values[0])
    
    return df_location

def load_and_prepare_data(args):
    all_studies_df = pd.read_csv(args.all_studies_csv)
    GT_metadata = pd.read_csv(args.gt_cmb_metadata_csv)
    GT_metadata_radiomics = pd.read_csv(args.gt_radiomics_metadata_csv)
    pred_metadata_df = get_predictions_df(args.cmb_pred_metadata_dir)

    # Compute extra metrics for pred
    pred_metadata_df['radius'] = pred_metadata_df['n_voxels'].apply(
        lambda x: ((x * (0.5**3)) / (4 / 3 * np.pi)) ** (1 / 3)
    )
    pred_metadata_df = add_location(pred_metadata_df) # add locations

    # Convert string representations of tuples to actual tuples
    GT_metadata["CM"] = GT_metadata["CM"].apply(lambda x: tuple(ast.literal_eval(x)))
    GT_metadata_radiomics["CM"] = GT_metadata_radiomics["CM"].apply(
        lambda x: tuple(ast.literal_eval(x))
    )

    GT_metadata_all = pd.merge(
        GT_metadata, GT_metadata_radiomics, on=["seriesUID", "CM"], how="inner"
    )
    GT_metadata_all = add_location(GT_metadata_all, False) # add locations
    
    
    

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

    # Check validity of data
    # jorge = GT_metadata[~(GT_metadata['seriesUID'].isin(GT_metadata_radiomics['seriesUID']) & GT_metadata['CM'].isin(GT_metadata_radiomics['CM']))]
    # print(jorge)
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

    for loc in [
    'Cortex / grey-white junction ', 'Subcortical white matter', 
        'Basal ganglia grey matter', 'Thalamus', 'Brainstem', 'Cerebellum', 
    ]:
        print("-----------------------------")
        print(f"Location: {loc}")
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
