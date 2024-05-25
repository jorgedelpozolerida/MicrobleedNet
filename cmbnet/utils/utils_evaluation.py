#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Script containing evaluation related functions


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
from collections import defaultdict, Counter
import ast

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

from scipy.ndimage import generate_binary_structure
from scipy.ndimage import label as nd_label
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from scipy.ndimage import center_of_mass, labeled_comprehension

from cmbnet.utils.utils_general import (
    calculate_radiomics_features,
    calculate_synthseg_features,
)
import json
import queue

###############################################################################
# Helper functions
###############################################################################


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """
    Performs element-wise division between two arrays. Returns NaN where the denominator is zero.
    """
    # Ensure the output array is initialized with a float dtype to hold floating-point division results
    output_array = np.full_like(numerator, fill_value=np.nan, dtype=np.float64)
    result = np.divide(
        numerator, denominator, where=(denominator != 0), out=output_array
    )
    return result


###############################################################################
# Subject level evaluation
###############################################################################


def compute_classification_eval(true_matrix, predicted_matrix):
    # Check if class is present in the true and predicted matrices
    true_contains_class = np.any(true_matrix == 1)
    predicted_contains_class = np.any(predicted_matrix == 1)

    if true_contains_class and predicted_contains_class:
        return "TP"
    elif not true_contains_class and predicted_contains_class:
        return "FP"
    elif true_contains_class and not predicted_contains_class:
        return "FN"
    else:  # Neither true nor predicted contain the class
        return "TN"


def compute_segmentation_eval(true_array, predicted_array, n_classes=2):
    # Flatten the arrays to treat them as 1D arrays for confusion matrix computation
    true_flat = true_array.ravel()
    pred_flat = predicted_array.ravel()
    # Generate confusion matrix
    cm = sklearn_confusion_matrix(true_flat, pred_flat, labels=range(n_classes))

    # Convert the numpy array to a Python list for JSON serialization
    cm_list = cm.tolist()

    return cm_list


def compute_detection_eval_subject_level(
    true_array: np.ndarray, predicted_array: np.ndarray, min_overlap_voxels=1
) -> dict:

    label_structure = generate_binary_structure(3, 3)
    results = defaultdict(dict)

    class_int = 1  # HARCODED: Assuming binary classification
    true_array_class = true_array == class_int
    predicted_array_class = predicted_array == class_int

    # Label connected components
    labeled_true, n_true_CC = nd_label(true_array_class, structure=label_structure)
    labeled_pred, n_pred_CC = nd_label(predicted_array_class, structure=label_structure)

    n_TP = 0
    true_cc_hit_ids = []
    n_repeated_overlaps = 0
    repeated_overlaps_ids = []

    for predicted_cc in range(1, n_pred_CC + 1):
        predicted_cc_mask = labeled_pred == predicted_cc
        for true_cc in range(1, n_true_CC + 1):
            true_cc_mask = labeled_true == true_cc
            overlap = np.logical_and(predicted_cc_mask, true_cc_mask)
            overlap_size = np.sum(overlap)

            if overlap_size >= min_overlap_voxels:
                if true_cc in true_cc_hit_ids:
                    n_repeated_overlaps += 1
                    repeated_overlaps_ids.append(true_cc)
                else:
                    n_TP += 1
                    true_cc_hit_ids.append(true_cc)

    # Update results with counts of connected components
    results["n_true_CC"] = n_true_CC
    results["n_pred_CC"] = n_pred_CC

    # Update results with true positives and repeated overlaps
    results["n_overlap_CC"] = n_TP
    results["n_repeated_hits"] = n_repeated_overlaps
    results["repeated_hits_per_CC"] = dict(Counter(repeated_overlaps_ids))

    return results


def compute_subject_level_evaluation(gt_mask, pred_mask, eval_method):
    """
    Selects function to use for each type of eval
    Possible: 'segmentation', 'classification', 'detection'
    """
    assert gt_mask.shape == pred_mask.shape
    func_mapping = {
        "segmentation": compute_segmentation_eval,
        "classification": compute_classification_eval,
        "detection": compute_detection_eval_subject_level,
    }

    metrics_out = {
        f"{eval_method}_results": func_mapping[eval_method](gt_mask, pred_mask)
    }
    return metrics_out


###############################################################################
# CMB level evaluation
###############################################################################


def compute_localization_criteria_CC(
    individual_CC_centerofmass,
    individual_CC,
    labelled_mask,
    fullmask,
    all_GTmask_centerofmass,
    mappings_label2CM,
    voxel_spacing=0.5,
):
    """
    Compares the individual CC to every CC in the provided labelled mask.

    Parameters:
        individual_CC_centerofmass (tuple): Center of mass for the individual connected component.
        individual_CC (np.ndarray): Binary mask where the individual connected component is 1.
        labelled_mask (np.ndarray): Mask with labelled connected components.
        fullmask (np.ndarray): Full binary mask of the region.
        all_mask_centerofmass (tuple): Center of mass for the full mask.
        mappings_label2CM (dict): Mapping from labels in labelled_mask to 'CM' identifiers.
        voxel_spacing (float): Physical distance represented by one voxel in millimeters.

    (Based on):
        https://metrics-reloaded.dkfz.de/metric?id=center_distance
        https://metrics-reloaded.dkfz.de/metric?id=point_inside_mask

    Returns:
        dict: Dictionary containing overlap and distance measures.
    """
    # Compute overlap and count labels within the region of the individual CC
    labels_in_individual_CC = labelled_mask[individual_CC == 1]
    unique_labels, counts = np.unique(labels_in_individual_CC, return_counts=True)
    overlaps = dict(zip(unique_labels, counts))

    # Remove the background label (0) from overlaps if present
    overlaps.pop(0, None)

    # Map overlaps to CM identifiers and gather data
    overlap_CM_counts = {
        mappings_label2CM.get(label, f"Unknown_CM_{label}"): count
        for label, count in overlaps.items()
    }

    # Compute distances to all centers of mass from the individual CC's center of mass
    distances_to_all_CMs = {
        cm: np.linalg.norm(np.array(individual_CC_centerofmass) - np.array(cm))
        * voxel_spacing
        for i, cm in enumerate(all_GTmask_centerofmass)
    }

    # Assemble results
    results = {
        "OverlapCMCounts": overlap_CM_counts,
        "DistancesToAllCMs": distances_to_all_CMs,
    }
    return results


def compute_greedy_matching(
    localization_results, criteria="OverlapCMCounts", overlap_th=1, distance_th=5
):
    """
    Performs a greedy matching of predicted CCs to GT CCs based on specified criteria with one-to-one correspondence using a queue system.

    Args:
        localization_results (list of dict): List containing overlap and distance info for each predicted CC.
        criteria (str): Criteria to use for matching, either "OverlapCMCounts" or "CenterDistance".
        overlap_th (int): Minimum overlap count threshold for a valid match.
        distance_th (float): Maximum distance threshold for a valid match.

    Returns:
        dict: Updated localization results with matching GT information included.
    """
    assert criteria in ["OverlapCMCounts", "DistancesToAllCMs"]

    # Initialize mapping of GT CCs to their best matching pred CC and the queue of predictions to process
    gt_matches = {}
    pred_matches = {}
    pred_queue = queue.Queue()

    # Enqueue all predictions
    for result_dict in localization_results:
        pred_queue.put(result_dict)

    # Process the queue until empty
    while not pred_queue.empty():
        result_dict = pred_queue.get()
        pred_CM = result_dict["pred_CM"]
        criteria_results = result_dict[criteria]

        best_gt = None
        best_score = float("inf") if criteria == "DistancesToAllCMs" else 0

        for gt_cc_CM, score in criteria_results.items():
            if (criteria == "OverlapCMCounts" and score < overlap_th) or (
                criteria == "DistancesToAllCMs" and score > distance_th
            ):
                continue  # Skip invalid matches based on threshold

            # Check if the current score is better and if this GT CC is available or matched with a worse score
            if (
                (criteria == "OverlapCMCounts" and score > best_score)
                or (criteria == "DistancesToAllCMs" and score < best_score)
            ) and (gt_cc_CM not in gt_matches or score > gt_matches[gt_cc_CM]["score"]):

                # Re-enqueue the previously matched pred CC if the current one is a better match
                if gt_cc_CM in gt_matches:
                    previous_pred_CM = gt_matches[gt_cc_CM]["pred_CM"]
                    previous_pred_CM_dict = next(
                        (
                            x
                            for x in localization_results
                            if x["pred_CM"] == previous_pred_CM
                        ),
                        None,
                    )
                    pred_queue.put(previous_pred_CM_dict)

                best_gt = gt_cc_CM
                best_score = score

        # Update the mappings if a suitable match has been found
        if best_gt:
            pred_matches[pred_CM] = best_gt
            gt_matches[best_gt] = {"pred_CM": pred_CM, "score": best_score}

    # Add match information to the localization results
    for result_dict in localization_results:
        pred_CM = result_dict["pred_CM"]
        matched_gt = pred_matches.get(pred_CM, None)
        result_dict[f"matched_GT_{criteria}"] = matched_gt

    return localization_results


def get_predicted_CC_matches_and_metadata(
    mri_nib, gt_nib, pred_nib, synth_nib, cmb_metadata_df, msg
):
    """
    Evaluates CMBs using different methods and includes evaluations for all predicted connected components.
    """
    mri_data, gt_mask, pred_mask, synth_mask = (
        mri_nib.get_fdata(),
        gt_nib.get_fdata(),
        np.squeeze(pred_nib.get_fdata()),
        np.squeeze(synth_nib.get_fdata()),
    )

    assert (
        gt_mask.shape == pred_mask.shape
    ), "Ground truth and prediction masks must have the same dimensions."

    # Label the connected components in both GT and predicted masks
    gt_labeled, num_gt_cc = nd_label(gt_mask)
    pred_labeled, num_pred_cc = nd_label(pred_mask)

    # Get labelnum mapping with CM for every GT CMB
    all_GT_CMs = [
        tuple(map(int, ast.literal_eval(x))) for x in cmb_metadata_df["CM"].to_list()
    ]
    mappings_label2CM = {gt_labeled[com]: com for com in all_GT_CMs}

    pred_CMBs_results = []
    for cc in range(1, num_pred_cc + 1):
        pred_CC_individual_mask = pred_labeled == cc
        predCC_com = center_of_mass(pred_CC_individual_mask)
        predCC_eval_results = compute_localization_criteria_CC(
            predCC_com,
            pred_CC_individual_mask,
            gt_labeled,
            gt_mask,
            all_GT_CMs,
            mappings_label2CM,
        )  # we compare individual pred CMB to all GT CCs

        radiomics_results, msg = calculate_radiomics_features(
            mri_data, pred_CC_individual_mask, msg
        )
        synthseg_results = calculate_synthseg_features(
            mri_data, pred_CC_individual_mask, synth_mask
        )
        predCC_com = tuple(map(int, predCC_com))
        pred_CMBs_results.append(
            {
                "pred_CM": predCC_com,
                "n_voxels": np.sum(pred_CC_individual_mask),
                **predCC_eval_results,
                **radiomics_results,
                **synthseg_results,
            }
        )

    # Perform Greedy (by Score) Matching
    pred_CMBs_results = compute_greedy_matching(
        pred_CMBs_results, criteria="OverlapCMCounts"
    )
    pred_CMBs_results = compute_greedy_matching(
        pred_CMBs_results, criteria="DistancesToAllCMs"
    )

    # Count the number of matches for each criterion
    matched_gts_overlap = len(
        {
            res["matched_GT_OverlapCMCounts"]
            for res in pred_CMBs_results
            if "matched_GT_OverlapCMCounts" in res
        }
    )
    matched_gts_distance = len(
        {
            res["matched_GT_DistancesToAllCMs"]
            for res in pred_CMBs_results
            if "matched_GT_DistancesToAllCMs" in res
        }
    )

    # Count unmatched GT CCs
    unmatched_gt_ccs_overlap = num_gt_cc - matched_gts_overlap
    unmatched_gt_ccs_distance = num_gt_cc - matched_gts_distance

    msg += f"\tTotal Matches for Overlap: {matched_gts_overlap}, Unmatched GT CCs: {unmatched_gt_ccs_overlap}\n"
    msg += f"\tTotal Matches for Distance: {matched_gts_distance}, Unmatched GT CCs: {unmatched_gt_ccs_distance}\n"
    return pred_CMBs_results, msg


def get_detection_metrics_from_call_counts(
    TP, FP, FN, n_true_cmb, n_pred_cmb, n_scans, fill_val=None, study="none"
):
    """
    Computes detection metrics from call counts of overlaps between GT and predicted CMBs.
    """
    summary = f"study={study} , n_true_cmb={n_true_cmb}, n_pred_cmb={n_pred_cmb}, n_scans={n_scans}, TP={TP}, FP={FP}, FN={FN} with fill_val={fill_val}"

    # Calculating metrics

    # Recall
    if (TP + FN) != 0:
        TPR = TP / (TP + FN)
    else:
        _logger.warning(f"TPR is None. {summary}")
        TPR = fill_val

    # Precision
    if (TP + FP) != 0:
        PPV = TP / (TP + FP)
    else:
        _logger.warning(f"TPR is None. {summary}")
        PPV = fill_val

    # F1
    if TPR is not None and PPV is not None:
        if (PPV + TPR) != 0:
            F1 = 2 * (PPV * TPR) / (PPV + TPR)
        else:
            F1 = fill_val
    else:
        _logger.warning(f"TPR is None. {summary}")
        F1 = fill_val

    # FPavg - per scan
    FPavg_scan = FP / n_scans

    # FPavg - per true CMB
    if n_true_cmb == 0:
        FPavg_true_cmb = None
    else:
        FPavg_true_cmb = FP / n_true_cmb
    return {
        "Precision": PPV,
        "Recall": TPR,
        "F1": F1,
        "FPavg": FPavg_scan,
        "FPcmb": FPavg_true_cmb,
    }


def perform_macro_averaging(metrics_list, id="seriesUID"):
    """
    Average metrics over all studies
    """

    metrics_df = pd.DataFrame(metrics_list)

    # Compute average for all columsn execept id column
    metrics_df = metrics_df.drop(id, axis=1)
    avg_metrics = metrics_df.mean(axis=0, skipna=True)
    std_metrics = metrics_df.std(axis=0, skipna=True)
    result_df = pd.DataFrame({"Mean": avg_metrics, "Std.": std_metrics})

    return result_df


def evaluate_detection_from_cmb_data(all_studies_df, GT_metadata_all, pred_metadata_df, match_col="matched_GT_DistancesToAllCMs"):
    """
    Computes detection metrics from the metadata of GT and predicted CMBs,
    including overall detection accuracy and segmentation performance using Dice scores.
    """
    study_results_detection = []
    study_results_dice = []
    
    # Initialize counters for global metrics
    true_positives_global = 0
    false_positives_global = 0
    false_negatives_global = 0
    overlap_global = 0
    n_voxels_pred_global = 0
    n_voxels_GT_global = 0
    

    all_cmbs_tracking = []

    # Process each study
    for study in all_studies_df["seriesUID"].unique():
        true_positives, false_positives, false_negatives, true_negatives = 0, 0, 0, 0
        
        gt_CM_hitted = []

        pred_metadata_df_study = pred_metadata_df[pred_metadata_df["seriesUID"] == study]
        gt_metadata_study = GT_metadata_all[GT_metadata_all["seriesUID"] == study]

        if gt_metadata_study.shape[0] == 0:
            continue
            # for i, row in pred_metadata_df[pred_metadata_df["seriesUID"] == study].iterrows():
            #     false_positives += 1
            #     false_positives_global += 1
            # if false_positives == 0:
            #     true_negatives_global += 1
            # detection_metrics_study = get_detection_metrics_from_call_counts(
            #     true_positives, false_positives, false_negatives,
            #     len(gt_metadata_study), len(pred_metadata_df_study), 1, fill_val=None, study=study
            # )
            # study_results_detection.append({"seriesUID": study, **detection_metrics_study})

        # Evaluate predictions for each study
        for i, row in pred_metadata_df_study.iterrows():
            matched_GT = row[match_col]
            gt_metadata_study_cm = gt_metadata_study[gt_metadata_study['CM'] == matched_GT]
            if matched_GT is None or pd.isnull(matched_GT) or gt_metadata_study_cm.shape[0] == 0:
                false_positives += 1
                false_positives_global += 1
                all_cmbs_tracking.append(
                    {"seriesUID": study, "CM": row["pred_CM"], "call": "FP", "type": "pred","matched_CM": None}
                )
                continue

            if matched_GT in gt_CM_hitted:
                continue

            true_positives += 1
            true_positives_global += 1
            all_cmbs_tracking.append(
                    {"seriesUID": study, "CM": row["pred_CM"], "call": "TP", "type": "pred", "matched_CM": matched_GT}
                )
            all_cmbs_tracking.append(
                    {"seriesUID": study, "CM": matched_GT, "call": "TP", "type": "GT", "matched_CM": row["pred_CM"]}
                )
            gt_CM_hitted.append(matched_GT)

            # Compute and accumulate Dice Score components
            OverlapCMCounts = row['OverlapCMCounts']
            matched_CM = row[match_col]
            try:
                overlap = OverlapCMCounts[matched_CM]
                overlap_global += overlap
                n_voxels_pred = row['n_voxels']
                n_voxels_pred_global += n_voxels_pred
                n_voxels_GT = gt_metadata_study_cm['size'].values[0]
                n_voxels_GT_global += n_voxels_GT
                
                dice_score = 2 * overlap / (n_voxels_pred + n_voxels_GT)
                study_results_dice.append({"seriesUID": study, "dice_score": dice_score})

            except Exception as e:
                print(f"Error in study {study}: Key error {e} in Dice score calculation.")

        # Check for false negatives
        for gt_CM in gt_metadata_study["CM"].unique():
            if gt_CM not in gt_CM_hitted:
                false_negatives += 1
                false_negatives_global += 1
                all_cmbs_tracking.append(
                    {"seriesUID": study, "CM": gt_CM, "call": "FN", "type": "GT", "matched_CM": None}
                )

        # Store study-specific detection metrics
        detection_metrics_study = get_detection_metrics_from_call_counts(
            true_positives, false_positives, false_negatives,
            len(gt_metadata_study), len(pred_metadata_df_study), 1, fill_val=None, study=study
        )
        study_results_detection.append({"seriesUID": study, **detection_metrics_study})

    # Compute global and macro metrics
    global_metrics = get_detection_metrics_from_call_counts(
        true_positives_global, false_positives_global, false_negatives_global,
        len(GT_metadata_all), len(pred_metadata_df), GT_metadata_all["seriesUID"].nunique()
    )
    micro_metrics = pd.DataFrame({"Mean": global_metrics.values()}, index=global_metrics.keys())
    micro_metrics.index = ["Micro - " + name for name in micro_metrics.index]

    macro_metrics = perform_macro_averaging(study_results_detection)
    macro_metrics.index = ["Macro - " + name for name in macro_metrics.index]

    detection_results = pd.concat([micro_metrics, macro_metrics])

    dice_score_macro = perform_macro_averaging(study_results_dice)
    dice_score_macro.index = ["Macro - Dice Score"]
    dice_score_micro = 2 * overlap_global / (n_voxels_pred_global + n_voxels_GT_global)
    dice_score_micro = pd.DataFrame({"Mean": dice_score_micro}, index=["Micro - Dice Score"])
    all_dice_score_results = pd.concat([dice_score_macro, dice_score_micro])

    return detection_results, all_dice_score_results, study_results_detection, study_results_detection, all_cmbs_tracking



def evaluate_classification_from_cmb_data(
    all_studies_df,
    GT_metadata_all,
    pred_metadata_df,
    threshold=1,
    match_col="matched_GT_DistancesToAllCMs",
):
    """
    Computes classification metrics from the metadata of GT and predicted CMBs.

    'Classification' here defined as binary classification into these groups:
        - Less than threshold
        - More or equal than threshold

    If e.g. threshold=1, then it is a binary classification into having at least 1 CMB or not (healthy)
    If e.g. threshold=3, then it is a binary classification into having at least 3 CMBs or not (healthy and unhealthy with 1 or 2 CMBs)

    """
    # print(threshold, "------------")
    results = []
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    for i, row in all_studies_df.iterrows():
        study = row["seriesUID"]
        n_CMB_gt = row["n_CMB_new"]

        pred_metadata_df_study = pred_metadata_df[
            pred_metadata_df["seriesUID"] == study
        ]
        gt_metadata_study = GT_metadata_all[GT_metadata_all["seriesUID"] == study]

        pred_metadata_df_filt = pred_metadata_df[pred_metadata_df["seriesUID"] == study]
        n_pred = pred_metadata_df_filt.shape[0]
        pred_metadata_df_filt = pred_metadata_df_filt[
            ~pred_metadata_df_filt["matched_GT_DistancesToAllCMs"].isnull()
        ]
        n_pred_TP = pred_metadata_df_filt.shape[0]
        n_gt = GT_metadata_all[GT_metadata_all["seriesUID"] == study].shape[0]

        if n_CMB_gt == 0:
            assert (
                study not in GT_metadata_all["seriesUID"]
            ), f"Study {study} has CMBs in GT metadata but shoud not"

        gt_group = n_gt >= threshold
        # pred_group = n_pred >= threshold
        pred_group = n_pred_TP >= threshold

        if gt_group == pred_group:
            if not gt_group:
                call = "TN"
                true_negatives += 1
            else:
                call = "TP"
                true_positives += 1
        elif gt_group:
            call = "FN"
            false_negatives += 1
        else:
            call = "FP"
            false_positives += 1

        # print(
        #     f"Study {study} - nGT: {n_gt} - nPred: {n_pred} - nPred_true: {n_pred_TP} - GT Group: {gt_group} - Pred Group: {pred_group} - Call: {call}"
        # )

    confusion_matrix = np.array(
        [[true_positives, false_negatives], [false_positives, true_negatives]]
    )
    metrics = compute_metrics_from_cm(confusion_matrix)
    # print(metrics)
    return metrics


def evaluate_segmentation_from_cmb_data(
    all_studies_df,
    GT_metadata_all,
    pred_metadata_df,
    match_col="matched_GT_DistancesToAllCMs",
):
    """
    Segmentation metrics for matching microbleeds
    """
    


###############################################################################
# COMBINE METRICS
###############################################################################


def combine_evaluate_detection(df):

    # Extracting detection-related data
    df_detection = df["detection"].apply(pd.Series)

    print(df_detection)

    n_true_CC_array = df_detection["n_true_CC"].tolist()
    n_pred_CC_array = df_detection["n_pred_CC"].tolist()
    n_overlap_CC_array = df_detection["n_overlap_CC"].tolist()
    n_repeated_hits_array = df_detection["n_repeated_hits"].tolist()
    repeated_hits_per_CC_array = df_detection["repeated_hits_per_CC"].tolist()

    # Convert inputs to numpy arrays for efficient computation
    gt_pos_array = np.array(n_true_CC_array)
    pred_pos_array = np.array(n_pred_CC_array)
    TP_array = np.array(n_overlap_CC_array)

    # Assert that the lengths of arrays are equal
    assert (
        len(gt_pos_array) == len(pred_pos_array) == len(TP_array)
    ), "The lengths of TP, FP, FN, and pos arrays do not match."
    n_patients = len(gt_pos_array)

    # Calculate False Positives (FP) and False Negatives (FN) for each patient
    FP_array = np.maximum(pred_pos_array - TP_array, 0)
    FN_array = np.maximum(gt_pos_array - TP_array, 0)

    # Macro Averaged Metrics --------------------------------------------
    TPR_macro = np.nanmean(safe_divide(TP_array, gt_pos_array))
    PPV_macro = np.nanmean(safe_divide(TP_array, pred_pos_array))
    F1_macro = np.nanmean(2 * safe_divide(PPV_macro * TPR_macro, PPV_macro + TPR_macro))

    TPavg = np.nanmean(TP_array)
    FPavg = np.nanmean(FP_array)
    FPmedian = np.nanmedian(FP_array)
    FNavg = np.nanmean(FN_array)
    FPcmb_array = safe_divide(
        FP_array, gt_pos_array
    )  # Using safe_divide directly, assuming it handles zeros as NaN
    FPcmb_avg = np.nanmean(
        FPcmb_array
    )  # FP per total true CC (averaged across patients)

    # Micro Averaged Metrics --------------------------------------------
    TP_total = np.sum(TP_array)
    FP_total = np.sum(FP_array)
    FN_total = np.sum(FN_array)
    TPR_micro = (
        TP_total / (TP_total + FN_total) if (TP_total + FN_total) > 0 else np.nan
    )
    PPV_micro = (
        TP_total / (TP_total + FP_total) if (TP_total + FP_total) > 0 else np.nan
    )
    F1_micro = (
        2 * (PPV_micro * TPR_micro) / (PPV_micro + TPR_micro)
        if (PPV_micro + TPR_micro) > 0
        else np.nan
    )

    FPavg_micro = FP_total / n_patients
    FNavg_micro = FN_total / n_patients
    TPavg_micro = TP_total / n_patients
    FPcmb_avg_micro = (
        FP_total / np.sum(gt_pos_array) if np.sum(gt_pos_array) > 0 else np.nan
    )

    # TOTALS -----------------------------------------------
    totals_dict = {
        "true_CC": sum(n_true_CC_array),
        "pred_CC": sum(n_pred_CC_array),
        "overlap_CC": sum(n_overlap_CC_array),
        "repeated_CC": sum(n_repeated_hits_array),
        "max_repeated_cmb": max(
            [
                val
                for hits_dict in repeated_hits_per_CC_array
                for val in hits_dict.values()
            ],
            default=0,
        ),
        "mean_repeated_cmb": np.mean(
            [
                val
                for hits_dict in repeated_hits_per_CC_array
                for val in hits_dict.values()
            ]
        ),
    }
    # Consolidating results
    results = {
        "macroaveraged_metrics": {
            "TPR": TPR_macro,
            "PPV": PPV_macro,
            "F1": F1_macro,
            "FPavg": FPavg,
            "TPavg": TPavg,
            "FNavg": FNavg,
            "FPcmb_avg": FPcmb_avg,
        },
        "microaveraged_metrics": {
            "TPR": TPR_micro,
            "PPV": PPV_micro,
            "F1": F1_micro,
            "FPavg": FPavg_micro,
            "TPavg": TPavg_micro,
            "FNavg": FNavg_micro,
            "FPcmb_avg": FPcmb_avg_micro,
            "FPmedian": FPmedian,
        },
        "totals": totals_dict,
    }

    # Creating a DataFrame for macroaveraged metrics
    macro_metrics = pd.DataFrame(results["macroaveraged_metrics"], index=[0]).round(4)

    # Creating a DataFrame for microaveraged metrics
    micro_metrics = pd.DataFrame(results["microaveraged_metrics"], index=[0]).round(4)

    # Creating a DataFrame for totals
    totals_metrics = pd.DataFrame(results["totals"], index=[0]).round(4)

    # Returning the three DataFrames
    return macro_metrics, micro_metrics, totals_metrics


def combine_evaluate_classification(df):
    # Counting occurrences of each metric
    TP = (df["classification"] == "TP").sum()
    FP = (df["classification"] == "FP").sum()
    TN = (df["classification"] == "TN").sum()
    FN = (df["classification"] == "FN").sum()

    # Calculating metrics
    TPR = TP / (TP + FN) if (TP + FN) != 0 else 0
    PPV = TP / (TP + FP) if (TP + FP) != 0 else 0
    F1 = 2 * (PPV * TPR) / (PPV + TPR) if (PPV + TPR) != 0 else 0
    TNR = TN / (TN + FP) if (TN + FP) != 0 else 0
    ACC = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) != 0 else 0

    # Create a DataFrame for classification results
    classification_metrics = pd.DataFrame(
        {"TPR": [TPR], "PPV": [PPV], "F1": [F1], "TNR": [TNR], "ACC": [ACC]}
    )

    return classification_metrics.round(4)


def compute_metrics_from_cm(
    confusion_matrix: np.ndarray, zero_division: float = np.nan
):
    # Compute F1, precision, recall and specificity for each class
    true_pos = np.diag(confusion_matrix)
    false_pos = np.sum(confusion_matrix, axis=0) - true_pos
    false_neg = np.sum(confusion_matrix, axis=1) - true_pos
    true_neg = np.sum(confusion_matrix) - (true_pos + false_pos + false_neg)

    # Compute precision while handling NaNs
    precision_mask = (true_pos + false_pos) != 0
    class_precision = np.empty(len(true_pos)) * zero_division
    class_precision[precision_mask] = true_pos[precision_mask] / (
        true_pos[precision_mask] + false_pos[precision_mask]
    )

    # Compute recall while handling NaNs
    recall_mask = (true_pos + false_neg) != 0
    class_recall = np.empty(len(true_pos)) * zero_division
    class_recall[recall_mask] = true_pos[recall_mask] / (
        true_pos[recall_mask] + false_neg[recall_mask]
    )

    # Compute F1s while handling NaNs
    f1_mask = (true_pos + false_pos + false_neg) != 0
    class_f1 = np.empty(len(true_pos)) * zero_division
    class_f1[f1_mask] = (
        2
        * true_pos[f1_mask]
        / (2 * true_pos[f1_mask] + false_pos[f1_mask] + false_neg[f1_mask])
    )

    # Compute specificity while handling NaNs
    specificity_mask = (true_neg + false_pos) != 0
    class_specificity = np.empty(len(true_pos)) * zero_division
    class_specificity[specificity_mask] = true_neg[specificity_mask] / (
        true_neg[specificity_mask] + false_pos[specificity_mask]
    )

    # Add class-wise metrics to metrics dict
    metrics = {}
    for i, (precision, recall, specificity, f1_score) in enumerate(
        zip(class_precision, class_recall, class_specificity, class_f1)
    ):
        metrics[f"precision_{i}"] = precision
        metrics[f"recall_{i}"] = recall
        metrics[f"specificity_{i}"] = specificity
        metrics[f"f1_{i}"] = f1_score

    return {
        "Precision": metrics[f"precision_1"],
        "Recall": metrics[f"recall_1"],
        "F1-Score": metrics[f"f1_1"],
        "Specificity": metrics[f"specificity_1"],
    }


def combine_evaluate_segmentation(df_cms, zero_division=np.nan):
    macro_metrics = {"Precision": [], "Recall": [], "F1-Score": [], "Specificity": []}
    # Initialize a zero matrix for microaveraging
    n_classes = 2  # harcoded
    micro_cm = np.zeros((n_classes, n_classes), dtype=np.int64)

    for _, row in df_cms.iterrows():
        cm = np.array(row["segmentation"], dtype=np.int64)
        micro_cm += cm  # Aggregate confusion matrices for microaveraging
        metrics = compute_metrics_from_cm(cm, zero_division=zero_division)
        # Collect metrics for macroaveraging
        for key in macro_metrics:
            macro_metrics[key].append(metrics[key])

    # Convert list of arrays into single array for each metric and compute mean for macroaveraged metrics
    macroaveraged_metrics = {
        key: np.nanmean(np.vstack(macro_metrics[key]), axis=0) for key in macro_metrics
    }
    macro_df = pd.DataFrame(macroaveraged_metrics)

    # Compute microaveraged metrics from the aggregated confusion matrix and create a DataFrame
    micro_metrics = compute_metrics_from_cm(micro_cm, zero_division=zero_division)
    micro_df = pd.DataFrame(
        {key: [value] for key, value in micro_metrics.items()}, index=[0]
    )

    return macro_df, micro_df
