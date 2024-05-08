#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Script with utility functions to help loading and cleaning CMB datasets


{Long Description of Script}


@author: jorgedelpozolerida
@date: 25/01/2024
"""
import os
import argparse
import traceback

import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import nibabel as nib
import time
from scipy.io import loadmat
import glob
import re
from typing import List, Tuple
import warnings
import sys
from typing import Tuple, Dict, List, Any

current_dir_path = os.path.dirname(os.path.abspath(__file__))
parent_dir_path = os.path.abspath(os.path.join(current_dir_path, os.pardir))
sys.path.append(parent_dir_path)

import cmbnet.preprocessing.datasets as dat_load

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


##############################################################################
###################                 GENERAL                ###################
##############################################################################


def check_for_duplicates(lst):
    seen = set()
    for element in lst:
        if element in seen:
            print(f"Duplicate found: {element}")
            raise ValueError(f"Duplicate element found: {element}")
        seen.add(element)


def get_dataset_subjects(dataset_name, input_dir):
    """
    Returns list of ALL studies from raw dataset directory

    Note: performs osme QCs in the process
    """
    if dataset_name == "valdo":
        assert "VALDO" in input_dir
        subjects = [
            d
            for d in os.listdir(input_dir)
            if os.path.isdir(os.path.join(input_dir, d))
        ]

    elif dataset_name == "cerebriu":
        assert "CEREBRIU" in input_dir
        subjects = [
            d
            for d in os.listdir(input_dir)
            if os.path.isdir(os.path.join(input_dir, d))
        ]

    elif dataset_name == "cerebriu-neg":
        assert "CEREBRIU-neg" in input_dir
        datadir = os.path.join(input_dir, "Data")
        subjects = [
            d for d in os.listdir(datadir) if os.path.isdir(os.path.join(datadir, d))
        ]

    elif dataset_name == "momeni":
        assert "MOMENI" in input_dir
        mri_dir = os.path.join(
            input_dir, "data", "PublicDataShare_2020", "rCMB_DefiniteSubject"
        )
        mri_dir_healthy = os.path.join(
            input_dir, "data", "PublicDataShare_2020", "NoCMBSubject"
        )
        subjects_cmb = [d.split(".")[0] for d in os.listdir(mri_dir)]
        subjects_h = [d.split(".")[0] for d in os.listdir(mri_dir_healthy)]
        subjects = subjects_cmb + subjects_h

    elif dataset_name == "momeni-synth":
        assert "MOMENI" in input_dir
        mri_dir = os.path.join(
            input_dir, "data", "PublicDataShare_2020", "sCMB_DefiniteSubject"
        )
        mri_dir_healthy = os.path.join(
            input_dir, "data", "PublicDataShare_2020", "sCMB_NoCMBSubject"
        )
        subjects_cmb = [d.split(".")[0] for d in os.listdir(mri_dir)]
        subjects_h = [d.split(".")[0] for d in os.listdir(mri_dir_healthy)]
        subjects = subjects_cmb + subjects_h

    elif dataset_name == "dou":
        assert "DOU" in input_dir
        subjects_mri = [
            d.split(".")[0] for d in os.listdir(os.path.join(input_dir, "nii"))
        ]
        subjects_gt = [
            d.split(".")[0] for d in os.listdir(os.path.join(input_dir, "ground_truth"))
        ]
        if set(subjects_mri) == set(subjects_gt):
            subjects = subjects_mri
        else:
            raise ValueError("Not all subjects contain annotations, check data")
    elif dataset_name == "rodeja":
        assert "RODEJA" in input_dir
        basedir = os.path.join(input_dir, "cmb_annotations")
        masks = []
        for sub in os.listdir(os.path.join(basedir, "Annotations")):
            masks += os.listdir(os.path.join(basedir, "Annotations", sub))
        subjects_gt = [re.search(r"\d+", i.split(".")[0]).group() for i in masks]
        mris = []
        for sub in ["cph_annotated", "cph_annotated_mip"]:
            mris += os.listdir(os.path.join(basedir, sub, "images"))
        subjects_mri = [i.split(".")[0] for i in mris]
        if set(subjects_mri) == set(subjects_gt):
            subjects = subjects_mri
        else:
            raise ValueError("Not all subjects contain annotations, check data")
    else:
        raise NotImplementedError

    check_for_duplicates(subjects)

    return subjects


def process_coordinates(
    com_list: List[Tuple[int, int, int]], msg: str = "", log_level: str = "\t\t"
) -> Tuple[List[Tuple[int, int, int]], str]:
    """
    Processes a list of 3D coordinates to ensure uniqueness and checks for coordinates with minimal differences.
    Updates a message whenever the list's length is changed.

    Args:
        com_list (List[Tuple[int, int, int]]): A list of tuples, where each tuple represents a 3D coordinate (x, y, z).
        msg (str): Initial message to be updated throughout the processing.

    Returns:
        Tuple[List[Tuple[int, int, int]], str]: A tuple containing the list of unique 3D coordinates after removing
                                                duplicates and handling minimal differences, and the updated message.
    """
    unique_coords = set(com_list)
    if len(unique_coords) < len(com_list):
        msg += f"{log_level}Removed {len(com_list) - len(unique_coords)} duplicate coordinates.\n"
    processed_coords = list(unique_coords)

    i = 0
    while i < len(processed_coords):
        coord1 = processed_coords[i]
        j = i + 1
        while j < len(processed_coords):
            coord2 = processed_coords[j]
            # Check if the coordinates differ by only 1 unit in any dimension
            if sum(abs(c1 - c2) for c1, c2 in zip(coord1, coord2)) == 1:
                warning_msg = f"{log_level}Coordinates {coord1} and {coord2} differ by only 1 unit. Keeping {coord1} and removing {coord2}."
                warnings.warn(warning_msg)
                processed_coords.pop(j)
                msg += warning_msg + "\n"
            else:
                j += 1
        i += 1

    return processed_coords, msg


def convert_numpy(obj):
    if isinstance(obj, np.integer):
        if isinstance(obj, np.int64):
            return int(obj)
        else:
            return obj.item()
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.complexfloating):
        return complex(obj)
    elif isinstance(obj, (np.void, np.record)):
        return obj.tobytes().hex()
    elif isinstance(obj, np.datetime64):
        return np.datetime_as_string(obj)
    elif isinstance(obj, np.timedelta64):
        return np.timedelta64(obj, "ms").astype("timedelta64[ms]").astype("int64")
    return obj


def extract_im_specs(img):
    return {
        "shape": img.shape,
        "voxel_dim": img.header.get_zooms(),
        "orientation": nib.aff2axcodes(img.affine),
        "data_type": img.header.get_data_dtype().name,
    }


def load_mris_and_annotations(args, subject, msg="", log_level="\t\t"):
    """
    Loads MRI scans and their corresponding annotations for a given subject
    from a specific dataset performing data cleaning and orientation fix.

    Args:
        args (object): Contains configuration parameters, including input directory and dataset name.
        subject (str): Identifier of the subject whose MRI scans and annotations are to be loaded.
        msg (str, optional): A string for logging purposes. Default is an empty string.

    Returns:
        mris (dict): Dictionary where keys are sequence names (e.g., "T1", "T2") and values are
                        the corresponding MRI scans loaded as nibabel.Nifti1Image objects.
        annotations (dict): Dictionary where keys are sequence names and values are the corresponding
                            annotations loaded as nibabel.Nifti1Image objects.
        labels_metadata (dict): Dictionary containing metadata related to labels (annotations).
        prim_seq (str): Primry sequence used for this subject
        msg (str): Updated log message.

    """
    msg += f"{log_level}Loading MRI scans and annotations...\n"

    start = time.time()

    if args.dataset_name == "valdo":
        sequences_raw, labels_raw, nifti_paths, labels_metadata, prim_seq, msg = (
            dat_load.load_VALDO_data(args, subject, msg)
        )
    elif args.dataset_name == "cerebriu":
        sequences_raw, labels_raw, nifti_paths, labels_metadata, prim_seq, msg = (
            dat_load.load_CEREBRIU_data(args, subject, msg)
        )
    elif args.dataset_name == "cerebriu-neg":
        sequences_raw, labels_raw, nifti_paths, labels_metadata, prim_seq, msg = (
            dat_load.load_CEREBRIUneg_data(args, subject, msg)
        )
    elif args.dataset_name == "dou":
        sequences_raw, labels_raw, nifti_paths, labels_metadata, prim_seq, msg = (
            dat_load.load_DOU_data(args, subject, msg)
        )
    elif args.dataset_name == "momeni":
        sequences_raw, labels_raw, nifti_paths, labels_metadata, prim_seq, msg = (
            dat_load.load_MOMENI_data(args, subject, msg)
        )
    elif args.dataset_name == "momeni-synth":
        sequences_raw, labels_raw, nifti_paths, labels_metadata, prim_seq, msg = (
            dat_load.load_MOMENIsynth_data(args, subject, msg)
        )
    elif args.dataset_name == "rodeja":
        sequences_raw, labels_raw, nifti_paths, labels_metadata, prim_seq, msg = (
            dat_load.load_RODEJA_data(args, subject, msg)
        )
    else:
        # Implement here for other datasets
        raise NotImplementedError

    im_specs_orig = extract_im_specs(sequences_raw[prim_seq])

    end = time.time()
    msg += (
        f"{log_level}\tLoading of MRIs and annotations took {end - start} seconds!\n\n"
    )

    return (
        sequences_raw,
        labels_raw,
        nifti_paths,
        labels_metadata,
        im_specs_orig,
        prim_seq,
        msg,
    )


###############################################################################
# Re-processing
###############################################################################


def get_sphere_df(csv_path, study, dataset):

    df = pd.read_csv(csv_path, sep=";", dtype=str)
    df = df[df["dataset"] == dataset]
    df["x"] = df["x"].astype(int)
    df["y"] = df["y"].astype(int)
    df["z"] = df["z"].astype(int)
    df["radius"] = df["radius"].astype(float)

    if "momeni" in dataset:
        subject = study.split("_")[0] + "_" + study.split("_")[1]
    else:
        subject = study

    return df[df["studyUID"] == subject]