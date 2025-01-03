#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Module with functions for internal CEREBRIU dataset


@author: jorgedelpozolerida
@date: 13/02/2024
"""
import os
import argparse
import traceback

import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import nibabel as nib
from scipy.io import loadmat
import glob
import sys
import json
from typing import Tuple, Dict, List, Any

import cmbnet.preprocessing.process_masks as process_masks
import cmbnet.utils.utils_general as utils_general
import cmbnet.utils.utils_plotting as utils_plt


import numpy.linalg as npl

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


##############################################################################
###################                   CEREBRIU             ###################
##############################################################################


def load_CEREBRIU_raw(input_dir, study):
    """
    Load raw MRI and segmentation data for a given CEREBRIU study.

    Args:
        input_dir (str): Directory containing the study subfolders.
        study (str): Specific study identifier.

    Returns:
        Tuple[Dict, Dict, str, str]: Tuple containing dictionaries of raw MRI sequences and labels,
                                        sequence type, and subfolder name.

    Raises:
        ValueError: If no files or multiple files are found where only one is expected.
    """
    mri_dir = os.path.join(input_dir, study, "images")
    cmb_dir = os.path.join(input_dir, study, "segmentations")

    # Find the CMB file in segmentations folder
    cmb_files = glob.glob(os.path.join(cmb_dir, "*.nii.gz"))
    if not cmb_files:
        raise ValueError("No CMB files found")
    elif len(cmb_files) > 1:
        raise ValueError(f"Multiple CMB files found in {cmb_dir}")

    # Get the CMB file and determine corresponding MRI subfolder
    cmb_file = cmb_files[0]
    cmb_filename = os.path.basename(cmb_file).split(".")[
        0
    ]  # Filename without extension

    # Find corresponding MRI file
    mri_subfolder_path = os.path.join(mri_dir, cmb_filename)
    if not os.path.isdir(mri_subfolder_path):
        raise ValueError(f"No corresponding MRI subfolder found for {cmb_filename}")

    mri_files = glob.glob(os.path.join(mri_subfolder_path, "*.nii.gz"))
    if not mri_files:
        raise ValueError(f"No MRI files found in {mri_subfolder_path}")
    elif len(mri_files) > 1:
        raise ValueError(f"Multiple MRI files found in {mri_subfolder_path}")

    # Load Raw MRI Sequences and Labels
    seq_type = cmb_filename.split("_")[0]
    sequences_raw = {seq_type: nib.load(mri_files[0])}
    labels_raw = {seq_type: nib.load(cmb_file)}

    nifti_paths = {seq_type: mri_files[0], "CMB": cmb_file}

    return sequences_raw, labels_raw, nifti_paths, seq_type, cmb_filename


def process_CEREBRIU_cmb(
    label_im: nib.Nifti1Image,
    labelid: int,
    mri_im: nib.Nifti1Image,
    size_threshold: int,
    max_dist_voxels: int,
    msg: str,
    multiple: bool = False,
    show_progress: bool = False,
    log_level="\t\t\t",
) -> Tuple[nib.Nifti1Image, nib.Nifti1Image, List[Tuple[int]], List[Dict], str]:
    """
    Processes Cerebriu CMB data using region growing and other operations.

    Args:
        label_im (nib.Nifti1Image): NIfTI image containing label data.
        labelid (int): The label identifier to filter the mask.
        mri_im (nib.Nifti1Image): NIfTI image of the MRI scan.
        size_threshold (int): The size threshold used for region growing.
        max_dist_voxels (int): The voxel distance threshold for region growing
        msg (str): Log message string.
        multiple (bool): Flag indicating if multiple processing is required.
        show_progress (bool): Flag indicating if tqdm progress bar desired.

    Returns:
        Tuple[.., .., , List[Tuple[int]], List[Dict], str]:
        - Processed mask as np array
        - Raw mask as np array
        - List of seeds used for region growing
        - Metadata for each CMB regarding region growing
        - Updated log message

    Raises:
        RuntimeError: If overlap is detected between individual processed masks.
    """
    mask_data = label_im.get_fdata()
    mri_data = mri_im.get_fdata()

    mask_filt = mask_data == int(labelid)

    voxel_size = label_im.header.get_zooms()  # Extract voxel size from the image
    mask_filt_single_list = process_masks.isolate_single_CMBs(mask_filt, voxel_size)

    rg_metadata = {}
    seeds_calculated = []
    final_processed_mask = np.zeros_like(mask_data, dtype=bool)
    msg += f"{log_level}Number of CMBs found in label id {labelid}: {len(mask_filt_single_list)}.\n"

    # Processing loop with optional progress bar
    iterator = range(len(mask_filt_single_list))
    if show_progress:
        iterator = tqdm(
            iterator, total=len(mask_filt_single_list), desc="Processing CMBs"
        )

    com_list = []
    intensity_mod_ = "point"
    diff_mode_ = "normal"
    connectivity_ = 6

    for i in iterator:
        cmb_single_mask = mask_filt_single_list[i]
        seeds = [tuple(seed) for seed in np.array(np.where(cmb_single_mask)).T]
        seeds_calculated.extend(seeds)

        seeds_array = np.array(seeds)
        average_seed = np.mean(seeds_array, axis=0)
        average_seed_int = np.round(average_seed).astype(int)
        com = tuple(int(i) for i in average_seed_int)

        processed_mask, metadata, msg = (
            process_masks.region_growing_with_auto_tolerance(
                volume=mri_data,
                seeds=seeds,
                size_threshold=size_threshold,
                max_dist_voxels=max_dist_voxels,
                tolerance_values=np.arange(
                    0, 150, 0.5
                ),  # TODO: explore better for CERBRIU data how
                connectivity=connectivity_,
                show_progress=show_progress,
                diff_mode=diff_mode_,
                log_level=f"{log_level}\t",
                msg=msg,
                intensity_mode=intensity_mod_,
            )
        )

        # Check for overlap
        if np.any(final_processed_mask & processed_mask):
            print(
                f"{log_level}\t\tCAUTION: Overlap detected at {com}\n"
                + f"{log_level}\t\t         Previosly visited CMBs: {com_list[:i]}\n"
            )
            raise RuntimeError("Overlap detected between individual processed masks")

        final_processed_mask = final_processed_mask | processed_mask

        # radius
        radius = (3 * int(metadata["n_pixels"]) / (4 * np.pi)) ** (1 / 3)

        rg_metadata[i] = {
            "CM": com,
            "size": metadata["n_pixels"],
            "radius": round(radius, 2),
            "region_growing": {
                "distance_th": max_dist_voxels,
                "size_th": size_threshold,
                "sphericity_ind": metadata["sphericity_ind"],
                "selected_tolerance": metadata["tolerance_selected"],
                "n_tolerances": metadata["tolerances_inspected"],
                "elbow_i": metadata["elbow_i"],
                "elbow2end_tol": metadata["elbow2end_tol"],
                "connectivity": connectivity_,  # Assuming connectivity is fixed in this example
                "intensity_mode": intensity_mod_,  # Assuming this is fixed for all CMBs
                "diff_mode": diff_mode_,  # Placeholder if different modes are considered
            },
        }

        msg += f"{log_level}Processed CMB {i}. n_seeds={len(seeds)}, new_size={np.sum(processed_mask)}\n"

    if not multiple and len(mask_filt_single_list) > 1:
        msg += f"{log_level}WARNING: Expected single CMBs and detected several.\n"

    return final_processed_mask, mask_filt, seeds_calculated, rg_metadata, msg


def process_cerebriu_anno(
    args: Any,
    subject: str,
    label_im: nib.Nifti1Image,
    mri_im: nib.Nifti1Image,
    seq_folder: str,
    msg: str,
    log_level="\t\t",
) -> Tuple[nib.Nifti1Image, Dict, str]:
    """
    Process annotations for a CEREBRIU dataset subject.

    Args:
        args (Any): Configuration parameters including input directory.
        subject (str): Subject identifier.
        label_im (nib.Nifti1Image): Label image.
        mri_im (nib.Nifti1Image): MRI image.
        seq_folder (str): Sequence folder name.
        msg (str): Log message.

    Returns:
        Tuple[nib.Nifti1Image, Dict, str]: Processed annotation image, metadata, and log message.
    """
    tasks_dict = utils_general.read_json_to_dict(
        os.path.join(args.input_dir, subject, "tasks.json")
    )
    task_item = next((it for it in tasks_dict if it["name"] == subject), None)
    series_data = (
        next((seq for seq in task_item["series"] if seq["name"] == seq_folder), None)
        if task_item
        else None
    )

    if not series_data:
        raise ValueError("Series data not found for the specified sequence folder.")

    extracted_data = {
        "segmentMap": series_data.get("segmentMap", {}),
        "landmarks3d": series_data.get("landmarks3d", []),
        "sequence_meta": series_data.get("classifications", []),
        "study_meta": task_item.get("classification", []),
        "other_meta": task_item.get("metaData", []),
    }

    assert extracted_data["segmentMap"], "segmentMap is empty"

    # Compute size threshold and maximum distance in voxels
    size_th, max_dist_voxels = process_masks.calculate_size_and_distance_thresholds(
        mri_im, max_dist_mm=10
    )
    msg = f"{log_level}Thresholds for RegionGrowing --> Max. distance ={max_dist_voxels}, Max Size={size_th}\n"

    label_mask_all = np.zeros_like(label_im.get_fdata(), dtype=bool)

    all_metadata = {}
    metadata_counter = 0

    for labelid, mask_dict in extracted_data["segmentMap"].items():
        multiple = mask_dict["attributes"].get("Multiple", False)
        msg += f"{log_level}Processing label {labelid} with {'multiple' if multiple else 'single'} CMB annotations.\n"

        label_mask, raw_mask, seeds, cmb_metadata, msg = process_CEREBRIU_cmb(
            label_im, labelid, mri_im, size_th, max_dist_voxels, msg, multiple
        )

        # Store metadata into single dict to keep consistency across datasets
        for k, cmb_data in cmb_metadata.items():
            all_metadata[str(metadata_counter)] = {
                **cmb_data,
                "seeds": seeds,
                "labelid": labelid,
            }
            metadata_counter += 1

        # Check for overlap
        if np.any(label_mask_all & label_mask):
            raise RuntimeError("Overlap detected between different CMB annotated masks")

        label_mask_all |= label_mask

    annotation_processed_nib = nib.Nifti1Image(
        label_mask_all.astype(np.int16), label_im.affine, label_im.header
    )

    return annotation_processed_nib, all_metadata, msg


def process_CEREBRIU_mri(args, subject, mri_im, seq_folder, msg):

    return mri_im, msg


def perform_CEREBRIU_QC(
    args, subject, mris, annotations, sequence_type, seq_folder, msg
):
    """
    Perform Quality Control (QC) specific to the CEREBRIU dataset on MRI sequences and labels.

    Args:
        args (Namespace): Arguments passed to the main function.
        subject (str): The subject identifier.
        mris (dict): Dictionary of MRI sequences.
        annotations (dict): Dictionary of labels.
        msg (str): Log message.

    Returns:
        mris_qc (dict): Dictionary of QC'ed MRI sequences.
        annotations_qc (dict): Dictionary of QC'ed labels.
        annotations_metadata (dict): Metadata associated with the QC'ed labels.
        msg (str): Updated log message.
    """

    mris_qc, annotations_qc, annotations_metadata = {}, {}, {}

    # Quality Control of MRI Sequences
    for mri_sequence, mri_im in mris.items():
        mris_qc[mri_sequence], msg = process_CEREBRIU_mri(
            args, subject, mri_im, seq_folder, msg
        )

    # Quality Control of Labels
    for anno_sequence, anno_im in annotations.items():

        if args.reprocess_file is None:
            annotations_qc[anno_sequence], metadata, msg = process_cerebriu_anno(
                args, subject, anno_im, mris_qc[anno_sequence], seq_folder, msg
            )
            annotations_metadata[anno_sequence] = metadata
            # Prepare metadata in correct format
            metadata_out = {
                sequence_type: {
                    "healthy": (
                        "no" if annotations_metadata.get(sequence_type) else "yes"
                    ),
                    "CMBs_old": annotations_metadata.get(sequence_type, {}),
                }
            }
        else:
            json_file = os.path.join(
                args.processed_dir,
                "Data",
                subject,
                "Annotations_metadata",
                f"{subject}_metadata.json",
            )
            with open(json_file, "r") as file:
                metadata_dict = json.load(file)
            com_list = [
                tuple(int(i) for i in cc["CM"])
                for cc in metadata_dict["CMBs_old"].values()
            ]

            annotations_qc[anno_sequence], metadata, msg = (
                process_masks.reprocess_study(
                    study=subject,
                    processed_dir=args.processed_dir,
                    mapping_file=args.reprocess_file,
                    dataset=args.dataset_name,
                    mri_im=mris_qc[anno_sequence],
                    com_list=com_list,
                    msg=msg,
                )
            )
            annotations_metadata[anno_sequence] = metadata
            metadata_out = {sequence_type: metadata}

    return mris_qc, annotations_qc, metadata_out, msg


def load_CEREBRIU_data(args, subject, msg):
    """
    Load MRI sequences and labels specific to the CEREBRIU dataset. Performs QC in the process.

    Args:
        args (Namespace): Command-line arguments or other configuration.
        subject (str): The subject identifier.
        msg (str): Log message.

    Returns:
        sequences_qc (dict): Dictionary of QC'ed MRI sequences.
        labels_qc (dict): Dictionary of QC'ed labels.
        labels_metadata (dict): Metadata associated with the labels.
        msg (str): Updated log message.
    """

    # 1. Load raw data
    sequences_raw, labels_raw, nifti_paths, sequence_type, seq_folder = (
        load_CEREBRIU_raw(args.input_dir, subject)
    )

    # 2. Perform Quality Control (QC) and Data Cleaning
    sequences_qc, labels_qc, labels_metadata, msg = perform_CEREBRIU_QC(
        args, subject, sequences_raw, labels_raw, sequence_type, seq_folder, msg
    )

    new_n_CMB = len(labels_metadata[sequence_type]["CMBs_old"])
    labels_metadata[sequence_type].update({"n_CMB_raw": new_n_CMB, "CMB_raw": []})

    # 3. Save plots for debugging
    utils_plt.generate_cmb_plots(
        subject,
        sequences_raw[sequence_type],
        labels_raw[sequence_type],
        labels_qc[sequence_type],
        labels_metadata[sequence_type]["CMBs_old"],
        plots_path=utils_general.ensure_directory_exists(
            os.path.join(args.plots_path, "pre")
        ),
        zoom_size=100,
    )

    return sequences_qc, labels_qc, nifti_paths, labels_metadata, sequence_type, msg


##############################################################################
###################               CEREBRIU - neg           ###################
##############################################################################


def load_CEREBRIUneg_raw(input_dir, study):
    """
    Load raw MRI and segmentation data for a given CEREBRIU-neg study.

    Args:
        input_dir (str): Directory containing the study subfolders.
        study (str): Specific study identifier.

    Returns:
        Tuple[Dict, Dict, str, str]: Tuple containing dictionaries of raw MRI sequences and labels,
                                        sequence type, and subfolder name.

    Raises:
        ValueError: If no files or multiple files are found where only one is expected.
    """
    mri_dir = os.path.join(input_dir, "Data", study)

    # Find the CMB file in segmentations folder
    mri_files = glob.glob(os.path.join(mri_dir, "*.nii.gz"))
    if not mri_files:
        raise ValueError("No MRI files found")
    elif len(mri_files) > 1:
        mri_files = [
            f for f in mri_files if "SWI" in f
        ]  # preference to negative SWI as there are less

    # Get the CMB file and determine corresponding MRI subfolder
    mri_file = mri_files[0]
    mri_filename = os.path.basename(mri_file).split("/")[-1]

    # Load Raw MRI Sequences and Labels
    seq_type = mri_filename.split("_")[1]
    # print(f"....................\n{mri_file}\n{mri_filename}\n{seq_type}\n")
    mri_im = nib.load(mri_file)
    sequences_raw = {seq_type: mri_im}
    labels_raw = {
        # empty CMB mask
        seq_type: nib.Nifti1Image(
            np.zeros_like(mri_im.get_fdata()), mri_im.affine, mri_im.header
        )
    }
    nifti_paths = {seq_type: mri_file, "CMB": None}
    return sequences_raw, labels_raw, nifti_paths, seq_type, mri_filename


def perform_CEREBRIUneg_QC(args, subject, mris, annotations, sequence_type, msg):
    """
    Perform Quality Control (QC) specific to the CEREBRIU-neg dataset on MRI sequences and labels.

    Args:
        args (Namespace): Arguments passed to the main function.
        subject (str): The subject identifier.
        mris (dict): Dictionary of MRI sequences.
        annotations (dict): Dictionary of labels.
        msg (str): Log message.

    Returns:
        mris_qc (dict): Dictionary of QC'ed MRI sequences.
        annotations_qc (dict): Dictionary of QC'ed labels.
        annotations_metadata (dict): Metadata associated with the QC'ed labels.
        msg (str): Updated log message.
    """

    mris_qc, annotations_qc, annotations_metadata = {}, {}, {}

    # Quality Control of Labels
    for anno_sequence, anno_im in annotations.items():
        annotations_qc[anno_sequence], metadata, msg = process_masks.process_cmb_mask(
            anno_im, msg
        )
        annotations_metadata[anno_sequence] = metadata

    # Quality Control of MRI Sequences
    for mri_sequence, mri_im in mris.items():
        mris_qc[mri_sequence], msg = process_CEREBRIU_mri(
            args, subject, mri_im, None, msg
        )

    # Prepare metadata in correct format
    metadata_out = {
        sequence_type: {
            "healthy": "no" if annotations_metadata.get(sequence_type) else "yes",
            "CMBs_old": annotations_metadata.get(sequence_type, {}),
        }
    }

    return mris_qc, annotations_qc, metadata_out, msg


def load_CEREBRIUneg_data(args, subject, msg):
    """
    Load MRI sequences and labels specific to the CEREBRIU-neg dataset.
    Performs QC in the process.

    Args:
        args (Namespace): Command-line arguments or other configuration.
        subject (str): The subject identifier.
        msg (str): Log message.

    Returns:
        sequences_qc (dict): Dictionary of QC'ed MRI sequences.
        labels_qc (dict): Dictionary of QC'ed labels.
        labels_metadata (dict): Metadata associated with the labels.
        msg (str): Updated log message.
    """

    # 1. Load raw data
    sequences_raw, labels_raw, nifti_paths, sequence_type, mri_filename = (
        load_CEREBRIUneg_raw(args.input_dir, subject)
    )

    # 2. Perform Quality Control (QC) and Data Cleaning
    sequences_qc, labels_qc, labels_metadata, msg = perform_CEREBRIUneg_QC(
        args, subject, sequences_raw, labels_raw, sequence_type, msg
    )

    new_n_CMB = len(labels_metadata[sequence_type]["CMBs_old"])
    labels_metadata[sequence_type].update({"n_CMB_raw": new_n_CMB, "CMB_raw": []})

    return sequences_qc, labels_qc, nifti_paths, labels_metadata, sequence_type, msg


##############################################################################
# Redbrick metadata manipulation
##############################################################################


def enrich_reprocessmetadata_with_processed(
    study, rawdir, processed_dir, reprocessed_dir
):
    """
    Adds metadata from redbrick to newest metadata
    """

    metadata_raw = utils_general.read_json_to_dict(
        os.path.join(rawdir, study, "tasks.json")
    )
    metadata_processed = utils_general.read_json_to_dict(
        os.path.join(
            processed_dir, study, "Annotations_metadata", f"{study}_metadata.json"
        )
    )
    metadata_reprocessed = utils_general.read_json_to_dict(
        os.path.join(
            reprocessed_dir, study, "Annotations_metadata", f"{study}_metadata.json"
        )
    )
    seq_folder = metadata_reprocessed["seq_type"]

    task_item = next((it for it in metadata_raw if it["name"] == study), None)
    series_data = (
        next(
            (seq for seq in task_item["series"] if f"{seq_folder}_" in seq["name"]),
            None,
        )
        if task_item
        else None
    )

    if not series_data:
        raise ValueError("Series data not found for the specified sequence folder.")

    extracted_data = {
        "segmentMap": series_data.get("segmentMap", {}),
        "landmarks3d": series_data.get("landmarks3d", []),
        "sequence_meta": series_data.get("classifications", []),
        "study_meta": task_item.get("classification", []),
        "other_meta": task_item.get("metaData", []),
    }

    # Map: get labelid from the first processing by COM as id
    mapping_reprocessed2processed = {}
    for cmb_id, cmb_meta in metadata_processed["CMBs_old"].items():
        com = cmb_meta["CM"]
        mapping_reprocessed2processed[tuple(com)] = [cmb_id, cmb_meta["labelid"]]

    # Affine transformations
    sequences_raw, labels_raw, nifti_paths, seq_type, cmb_filename = load_CEREBRIU_raw(
        rawdir, study
    )
    affine_before = labels_raw[seq_type].affine

    label_reprocessed = nib.load(
        (os.path.join(reprocessed_dir, study, "MRIs", f"{study}.nii.gz"))
    )
    affine_after = label_reprocessed.affine

    original2resample = npl.inv(affine_after).dot(affine_before)

    hitted_CMs = set()
    mapping_reprocessed_new2old = {}

    for cmb_id, cmb_meta in metadata_reprocessed["CMBs_old"].items():
        old_com = cmb_meta["CM"]
        new_com = process_masks.apply_affine(original2resample, np.array(old_com))
        new_com = tuple(map(int, new_com))  # Transform and round to integer coordinates

        # Find the closest new COM by comparing transformed old COM to new COMs
        matching_CM = min(
            metadata_reprocessed["CMBs_new"].items(),
            key=lambda item: np.linalg.norm(new_com - np.array(item[1]["CM"])),
        )

        # Extract the COM from the matched tuple
        matched_new_com = matching_CM[1][
            "CM"
        ]  # matching_CM is (id, meta), meta contains 'CM'
        matched_new_com = tuple(map(int, matched_new_com))

        # Ensure uniqueness in mapping
        if matched_new_com in hitted_CMs:
            print(
                f"ERROR: New {matched_new_com} maps to Old {old_com} (through ({new_com})"
            )
            raise ValueError(f"Repeated matching COM {matched_new_com}")

        hitted_CMs.add(matched_new_com)
        mapping_reprocessed_new2old[matched_new_com] = old_com

    for cmb_id, cmb_meta in metadata_reprocessed["CMBs_new"].items():
        com = cmb_meta["CM"]
        com = tuple(map(int, com))
        if com in mapping_reprocessed_new2old:
            com_old = mapping_reprocessed_new2old[tuple(com)]
            com_old = tuple(map(int, com_old))
            processed_id, rb_labelid = mapping_reprocessed2processed[com_old]
            try:
                metadata_reprocessed["CMBs_new"][cmb_id].update(
                    {
                        "processed_id": processed_id,
                        "RB_label": rb_labelid,
                        "RB_metadata": extracted_data["segmentMap"][rb_labelid],
                    }
                )
            except:
                print(com_old, "  ", mapping_reprocessed2processed[com_old])
                print(
                    f"ERROR in {study} for processed_label {rb_labelid}: {extracted_data}"
                )
        else:
            raise ValueError

    metadata_reprocessed.update(
        {
            "RB_landmarks": extracted_data["landmarks3d"],
            "RB_sequence_meta": extracted_data["sequence_meta"],
            "RB_study_meta": extracted_data["study_meta"],
            "RB_other_meta": extracted_data["other_meta"],
        }
    )

    return metadata_reprocessed
