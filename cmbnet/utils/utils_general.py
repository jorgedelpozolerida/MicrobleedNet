#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Script containing general utility functions


{Long Description of Script}


@author: jorgedelpozolerida
@date: 31/01/2024
"""

import os
import sys
import glob
import argparse
import traceback


import logging
import numpy as np
import pandas as pd
from datetime import datetime as dt
import json
import nibabel as nib
import ast
from clearml import Task
import SimpleITK as sitk
from radiomics import featureextractor
import subprocess
import numpy as np
from scipy.ndimage import center_of_mass, label as nd_label

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

radiomics_logger = logging.getLogger("radiomics")
radiomics_logger.setLevel(logging.ERROR)  # Suppresses INFO and DEBUG messages


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

###############################################################################
# General
###############################################################################


def ensure_directory_exists(dir_path, verbose=False):
    """Create directory if non-existent"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        if verbose:
            print(f"Created the following dir: \n{dir_path}")
    return dir_path


def write_to_log_file(msg, log_file_path, printmsg=False):
    """
    Writes message to the log file.
    Args:
        msg (str): Message to be written to log file.
        log_file_path (str): Path to log file.
    """
    current_time = dt.now()
    with open(log_file_path, "a+") as f:
        f.write(f"\n{current_time}\n{msg}")
    if printmsg:
        print(msg)


def confirm_action(message=""):
    """Prompt the user for confirmation before proceeding."""
    while True:
        answer = input(f"Do you want to proceed? [Y/n]: ")
        if not answer or answer[0].lower() == "y":
            return answer
        elif answer[0].lower() == "n":
            print("You did not approve. Exiting...")
            sys.exit(1)
        else:
            print("Invalid input. Please enter Y or n.")


def read_json_to_dict(file_path):
    """
    Reads a JSON file and converts it into a Python dictionary.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The JSON file content as a Python dictionary.
    """
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except Exception as e:
        print(f"Error reading the JSON file: {e}")
        return None


def create_nifti(data, affine, header, is_annotation=False):
    """
    Creates Nifti1Image given data array, header and affine matrix.
    Args:
        data (np.ndarray): Input data array.
        affine (np.ndarray): Affine matrix.
        header (nib.Nifti1Header): Header.
        is_annotation (bool): Whether image is an annotation or not.
    Returns:
        image (nib.Nifti1Image): Created Nifti1Image.
    """
    if is_annotation:
        image = nib.Nifti1Image(data.astype(np.uint8), affine=affine, header=header)
        image.set_data_dtype(np.uint8)

    else:
        image = nib.Nifti1Image(data.astype(np.float32), affine=affine, header=header)
        image.set_data_dtype(np.float32)

    return image


###############################################################################
# Data Loading and Manipulation
###############################################################################


def get_metadata_from_processed_final(data_dir, sub):

    metadata_dict = read_json_to_dict(
        os.path.join(data_dir, sub, "Annotations_metadata", f"{sub}_metadata.json")
    )
    metadata_dict_keys = list(metadata_dict.keys())
    return {
        "id": sub,
        "anno_path": os.path.join(data_dir, sub, "Annotations", f"{sub}.nii.gz"),
        "mri_path": os.path.join(data_dir, sub, "MRIs", f"{sub}.nii.gz"),
        "seq_type": metadata_dict_keys[0],
        "raw_metadata_path": os.path.join(
            data_dir, sub, "Annotations_metadata", f"{sub}_raw.json"
        ),
        "processed_metadata_path": os.path.join(
            data_dir, sub, "Annotations_metadata", f"{sub}_processed.json"
        ),
    }


def get_metadata_from_cmb_format(data_dir, sub_id):
    """
    Get all metadata from study using its subject id
    """

    fullpath_processing_metadata = os.path.join(
        data_dir, sub_id, "processing_metadata", f"{sub_id}.json"
    )
    processing_metadata_dict = read_json_to_dict(fullpath_processing_metadata)

    return {
        "id": sub_id,
        "anno_path": os.path.join(data_dir, sub_id, "Annotations", f"{sub_id}.nii.gz"),
        "mri_path": os.path.join(data_dir, sub_id, "MRIs", f"{sub_id}.nii.gz"),
        "processing_metadata_path": fullpath_processing_metadata,
        **processing_metadata_dict,
    }


def load_clearml_predictions(pred_dir):

    subjects = os.listdir(pred_dir)
    metadata = []
    for sub in subjects:
        pred_files = glob.glob(
            os.path.join(pred_dir, sub, f"**/{sub}_PRED.nii.gz"), recursive=True
        )
        if len(pred_files) == 0:
            raise ValueError(f"No prediction files found for {sub}, check your data")
        elif len(pred_files) > 1:
            raise ValueError(
                f"Multiple prediction files found for {sub}, check your data"
            )
        assert os.path.exists(pred_files[0])
        metadata.append({"id": sub, "pred_path": pred_files[0]})

    return metadata


def add_groundtruth_metadata(groundtruth_dir, gt_dir_struct, metadata):
    """
    Adds ground truth metadata to dict for subjects present.
    This function should be adapted to varying folder structures.
    """
    subjects_selected = [s_item["id"] for s_item in metadata]

    if gt_dir_struct == "processed_final":
        load_func = get_metadata_from_processed_final

    elif gt_dir_struct == "cmb_format":
        load_func = get_metadata_from_cmb_format
    else:
        raise NotImplementedError

    gt_metadata = {}

    for sub in subjects_selected:
        sub_meta = load_func(groundtruth_dir, sub)
        gt_metadata[sub] = sub_meta
    for meta_item in metadata:
        matching_item = gt_metadata[meta_item["id"]]
        meta_item.update({"gt_path": matching_item["anno_path"], **matching_item})

    return metadata


def add_CMB_metadata(CMB_metadata_df, metadata):

    for study_dict in metadata:
        sub_id = study_dict["id"]
        CMB_dict = study_dict["CMBs_new"]
        for cmb_id, cmb_dict in CMB_dict.items():
            com = np.array(cmb_dict["CM"], dtype=np.int32)  # Center of mass
            cmb_row = CMB_metadata_df[
                (CMB_metadata_df["seriesUID"] == sub_id)
                & (CMB_metadata_df["cmb_id"].astype(int) == int(cmb_id))
            ]
            if cmb_row.empty:
                raise ValueError(f"CMB {cmb_id} not found for subject {sub_id}")
            cmb_row = cmb_row.to_dict(orient="records")[0]
            assert all(com == cmb_row["CM"]), f"CM not mathcing for {sub_id} - {cmb_id}"
            cmb_row["CM"] = tuple(map(int, com))
            cmb_dict.update(cmb_row)
    return metadata


##############################################################################
# Radiomics/Shape analysis
##############################################################################

RADIOMICS_KEYS = [
    "shape_Elongation",
    "shape_Flatness",
    "shape_LeastAxisLength",
    "shape_MajorAxisLength",
    "shape_Maximum2DDiameterColumn",
    "shape_Maximum2DDiameterRow",
    "shape_Maximum2DDiameterSlice",
    "shape_Maximum3DDiameter",
    "shape_MeshVolume",
    "shape_MinorAxisLength",
    "shape_Sphericity",
    "shape_SurfaceArea",
    "shape_SurfaceVolumeRatio",
    "shape_VoxelVolume",
    "firstorder_10Percentile",
    "firstorder_90Percentile",
    "firstorder_Energy",
    "firstorder_Entropy",
    "firstorder_InterquartileRange",
    "firstorder_Kurtosis",
    "firstorder_Maximum",
    "firstorder_MeanAbsoluteDeviation",
    "firstorder_Mean",
    "firstorder_Median",
    "firstorder_Minimum",
    "firstorder_Range",
    "firstorder_RobustMeanAbsoluteDeviation",
    "firstorder_RootMeanSquared",
    "firstorder_Skewness",
    "firstorder_TotalEnergy",
    "firstorder_Uniformity",
    "firstorder_Variance",
]


def calculate_radiomics_features(mri_data, mask_data, msg=''):
    """
    Calculate Shape and First Order radiomics features for an object in a binary mask using PyRadiomics,
    considering isotropic voxel spacing of 0.5mm.

    Args:
        mri_data (numpy.ndarray): The MRI data array.
        mask_data (numpy.ndarray): The binary mask array where the object is labeled with 1.

    Returns:
        dict: A dictionary containing Shape and First Order radiomics features with simplified names.
    """
    try:
        # Convert numpy arrays to SimpleITK images and set spacing
        image_sitk = sitk.GetImageFromArray(mri_data)
        image_sitk.SetSpacing((0.5, 0.5, 0.5))  # Set isotropic spacing

        mask_sitk = sitk.GetImageFromArray(mask_data.astype(np.uint8))
        mask_sitk.SetSpacing((0.5, 0.5, 0.5))  # Set isotropic spacing

        # Check that the mask is binary with the correct labels
        unique_labels = np.unique(mask_data)
        assert (
            len(unique_labels) == 2 and 0 in unique_labels and 1 in unique_labels
        ), "Mask must be binary"

        # Set up the PyRadiomics feature extractor with specific parameters
        settings = {
            "binWidth": 25,
            "resampledPixelSpacing": [
                0.5,
                0.5,
                0.5,
            ],  # Override pixel spacing if necessary
            "interpolator": sitk.sitkBSpline,
            "enableCExtensions": True,
        }

        extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
        extractor.enableFeatureClassByName("shape")  # Enable only Shape features
        extractor.enableFeatureClassByName(
            "firstorder"
        )  # Enable only First Order features

        # Extract features
        result = extractor.execute(image_sitk, mask_sitk)

        # Convert the result to a clean dictionary and rename features for better clarity
        features_dict = {}
        for key, value in result.items():
            # Exclude diagnostics data
            if "diagnostics" not in key and "shape" in key or "firstorder" in key:
                # Normalize the key to create a readable format
                simplified_key = key.replace("original_", "")
                # Convert numpy arrays to floats if they contain only one element
                features_dict[simplified_key] = (
                    float(value.item())
                    if isinstance(value, np.ndarray) and value.size == 1
                    else value
                )
    except Exception as e:
        msg += f"Error calculating radiomics features: {e}"
        print(f"Error calculating radiomics features: {e}. Setting all to None...")
        return {key: None for key in RADIOMICS_KEYS}, msg
    
    return features_dict, msg


##############################################################################
# Synhtseg
##############################################################################


def apply_synthseg(args, input_path, output_path, synthseg_repo_path):
    # Construct the command
    command = [
        "python",
        f"{synthseg_repo_path}/scripts/commands/SynthSeg_predict.py",
        "--i",
        input_path,
        "--o",
        output_path,
    ]

    if args.robust_synthseg:
        command.extend(["--robust"])

    # Log the command
    logging.info("Running command: " + " ".join(command))

    # Run the command
    print(" ".join(command))
    subprocess.run(command, check=True)


def calculate_synthseg_features(mri_data, mask_data, synthseg_mask_data):
    """
    Calculates and returns the number of times each label in synthseg_mask_data
    is present in the mask_data where mask_data is equal to 1, and determines
    to which label the center of mass of mask_data belongs.

    Args:
        mri_data (numpy.ndarray): MRI data array (not used in this function, but provided for context).
        mask_data (numpy.ndarray): Binary mask data array where 1 indicates regions of interest.
        synthseg_mask_data (numpy.ndarray): Mask data with multiple labels to compare against mask_data.

    Returns:
        dict: A dictionary containing:
            - count_dict: a dictionary of label counts where the keys are labels from synthseg_mask_data
              and the values are the counts of these labels where mask_data is 1.
            - com_label: the label from synthseg_mask_data that corresponds to the center of mass of mask_data.
    """
    # First, create a mask where synthseg labels are only considered where mask_data is 1
    filtered_synthseg_mask = np.where(mask_data == 1, synthseg_mask_data, 0)

    # Calculate how many times each label in synthseg_mask_data is present where mask_data is 1
    unique_labels, counts = np.unique(filtered_synthseg_mask, return_counts=True)
    count_dict = dict(zip(unique_labels, counts))

    # Calculate the center of mass of the mask_data
    com = center_of_mass(mask_data)
    com_label = synthseg_mask_data[
        int(com[0]), int(com[1]), int(com[2])
    ]  # Assume 3D data; adjust for 2D if necessary

    return {"count_dict": count_dict, "com_label": com_label}
