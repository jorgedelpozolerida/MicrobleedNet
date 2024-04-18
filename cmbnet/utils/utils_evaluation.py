#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Script containing evaluation related functions


Three evaluaions are made:
- classification (has some CMB or not)
- segmentation (pixel-wise evaluation)
- detection (connected components evaluation)


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


logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

from scipy.ndimage import label, generate_binary_structure
from scipy.spatial.distance import dice as dice_score
from scipy.ndimage import label as nd_label
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix


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


def compute_classification_eval_individual(true_matrix, predicted_matrix):
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


def compute_segmentation_eval_individual(true_array, predicted_array, n_classes=2):
    # Flatten the arrays to treat them as 1D arrays for confusion matrix computation
    true_flat = true_array.ravel()
    pred_flat = predicted_array.ravel()
    # Generate confusion matrix
    cm = sklearn_confusion_matrix(true_flat, pred_flat, labels=range(n_classes))

    # Convert the numpy array to a Python list for JSON serialization
    cm_list = cm.tolist()

    return cm_list


def compute_detection_eval_individual(
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


def compute_individual_evaluation(gt_mask, pred_mask, eval_method):
    """
    Selects function to use for each type of eval
    Possible: 'segmentation', 'classification', 'detection'
    """
    assert gt_mask.shape == pred_mask.shape
    func_mapping = {
        "segmentation": compute_segmentation_eval_individual,
        "classification": compute_classification_eval_individual,
        "detection": compute_detection_eval_individual,
    }

    metrics_out = {eval_method: func_mapping[eval_method](gt_mask, pred_mask)}
    return metrics_out


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
    micro_cm = np.zeros((n_classes, n_classes), dtype=np.int)

    for _, row in df_cms.iterrows():
        cm = np.array(row["segmentation"])
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
