#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Script containing evaluation related functions


{Long Description of Script}


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


logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

from scipy.ndimage import label, generate_binary_structure
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score

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

def compute_segmentation_eval_individual( gt_mask, pred_mask, n_classes=2):
    
    # Flatten the arrays for metric calculation
    pred_mask_flat = pred_mask.flatten()
    gt_mask_flat = gt_mask.flatten()

    # Compute confusion matrix
    # cm = confusion_matrix(gt_mask_flat, pred_mask_flat, labels=list(range(n_classes)))

    # Calculate metrics
    # precision = precision_score(gt_mask_flat, pred_mask_flat, average='macro', labels=list(range(n_classes)))
    # recall = recall_score(gt_mask_flat, pred_mask_flat, average='macro', labels=list(range(n_classes)))
    f1 = f1_score(gt_mask_flat, pred_mask_flat, average='macro', labels=list(range(n_classes)))
    # kappa = cohen_kappa_score(gt_mask_flat, pred_mask_flat, labels=list(range(n_classes)))

    return {
        'Dice': f1,
        # 'precision': precision,
        # 'recall': recall,
        # 'kappa': kappa,
        # 'confusion_matrix': cm
    }

def compute_detection_eval_individual(gt_mask, pred_mask, min_overlap_voxels=1):
    structure = generate_binary_structure(3, 3)  # Adjust dimensions as needed

    # Label connected components in both masks
    labeled_pred, num_pred = label(pred_mask, structure)
    labeled_gt, num_gt = label(gt_mask, structure)

    predicted_overlapping_counts = 0

    # Iterate over each predicted connected component
    for predicted_cc in range(1, num_pred + 1):
        predicted_cc_mask = labeled_pred == predicted_cc

        # Check overlap with each true connected component
        for true_cc in range(1, num_gt + 1):
            true_cc_mask = labeled_gt == true_cc

            # Calculate overlap
            overlap = np.logical_and(predicted_cc_mask, true_cc_mask)
            overlap_size = np.sum(overlap)

            if overlap_size >= min_overlap_voxels:
                predicted_overlapping_counts += 1
                break  # Move to next predicted component after finding overlap

    # Compute basic metrics
    metrics = {
        'true_positive_count': predicted_overlapping_counts,
        'false_positive_count': max(0, num_pred - predicted_overlapping_counts),
        'false_negative_count': max(0, num_gt - predicted_overlapping_counts),
        'num_predicted_components': num_pred,
        'num_true_components': num_gt
    }
    return metrics


def compute_individual_evaluation(gt_mask, pred_mask, eval_method):
    """ 
    Selects function to use for each type of eval
    Possible: 'segmentation', 'classification', 'detection'
    """
    assert gt_mask.shape == pred_mask.shape
    func_mapping = {
        "segmentation": compute_segmentation_eval_individual,
        "classification": compute_classification_eval_individual,
        "detection": compute_detection_eval_individual
    }
    
    metrics_out = {eval_method: func_mapping[eval_method](gt_mask, pred_mask)}
    return metrics_out


###############################################################################
# COMBINE METRICS
###############################################################################


def combine_evaluate_detection(df):
    # Extracting detection-related data
    df_detection = df['detection'].apply(pd.Series)

    # Calculating TP, FP, FN
    TP = df_detection['true_positive_count'].sum()
    FP = df_detection['false_positive_count'].sum()
    FN = df_detection['false_negative_count'].sum()

    # Basic Metrics
    TPR = TP / (TP + FN) if (TP + FN) != 0 else 0
    PPV = TP / (TP + FP) if (TP + FP) != 0 else 0
    F1 = 2 * (PPV * TPR) / (PPV + TPR) if (PPV + TPR) != 0 else 0

    # Additional metrics calculations
    TPavg = TP / len(df)
    FPavg = FP / len(df)
    FPmedian = np.median(df_detection['false_positive_count'])  # Assuming you want the median of false positives per study
    FPcmb = FP / df_detection['num_true_components'].sum() if df_detection['num_true_components'].sum() != 0 else 0
    FNavg = FN / len(df)

    # Creating a DataFrame for results
    detection_metrics = pd.DataFrame({
        'TPR': [TPR],
        'PPV': [PPV],
        'F1': [F1],
        'TPavg': [TPavg],
        'FPavg': [FPavg],
        'FPmedian': [FPmedian],
        'FP/cmb': [FPcmb],
        'FNavg': [FNavg]
    })

    return detection_metrics.round(2)

def combine_evaluate_classification(df):
    # Counting occurrences of each metric
    TP = (df['classification'] == 'TP').sum()
    FP = (df['classification'] == 'FP').sum()
    TN = (df['classification'] == 'TN').sum()
    FN = (df['classification'] == 'FN').sum()

    # Calculating metrics
    TPR = TP / (TP + FN) if (TP + FN) != 0 else 0
    PPV = TP / (TP + FP) if (TP + FP) != 0 else 0
    F1 = 2 * (PPV * TPR) / (PPV + TPR) if (PPV + TPR) != 0 else 0
    TNR = TN / (TN + FP) if (TN + FP) != 0 else 0
    ACC = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) != 0 else 0

    # Create a DataFrame for classification results
    classification_metrics = pd.DataFrame({
        'TPR': [TPR],
        'PPV': [PPV],
        'F1': [F1],
        'TNR': [TNR],
        'ACC': [ACC]
    })

    return classification_metrics.round(2)

def combine_evaluate_segmentation(df):
    # Assuming 'segmentation' is a dictionary column in df
    df_segmentation = df['segmentation'].apply(pd.Series)

    # Calculate metrics
    segmentation_metrics = {}
    for metric in df_segmentation.columns:
        segmentation_metrics[metric] = df_segmentation[metric].mean()

    # Create a DataFrame for segmentation results
    segmentation_results = pd.DataFrame([segmentation_metrics])

    return segmentation_results.round(2)
