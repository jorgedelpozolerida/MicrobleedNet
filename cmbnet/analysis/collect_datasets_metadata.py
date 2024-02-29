#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Script to collect metadata from preprocessed datasets


{Long Description of Script}

TODO:
- add patient level data and other distinct metadata for each dataset
- add acquisition and scanner data

@author: jorgedelpozolerida
@date: 20/02/2024
"""

import os
import sys
import argparse
import traceback


import logging
import numpy as np
import pandas as pd
import json

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

def adjust_and_save_study_df(study_df, filename):
    """
    Adjusts the column order for study level DataFrame and saves it to CSV.
    
    Parameters:
    - study_df: DataFrame to be adjusted and saved.
    - filename: Filename for saving the DataFrame.
    """
    # Define the desired column order for the study level DataFrame
    desired_columns = [
        'subject', 'Dataset', 'seq_type', 'healthy', 
        'n_CMB_old', 'n_CMB_new', 'old_shape', 'new_shape', 
        'old_voxel_dim', 'new_voxel_dim', 'old_orientation', 'new_orientation'
    ]
    
    # Reorder the DataFrame according to the desired column order
    # This will place columns in the specified order, ignoring any that are not present
    study_df = study_df[desired_columns]
    
    # Save to CSV
    study_df.to_csv(filename, index=False)
def adjust_and_save_cmb_df(cmb_df, filename, base_columns, additional_columns=[]):
    """
    Adjusts the column order for CMB DataFrames and saves them to CSV.
    
    Parameters:
    - cmb_df: DataFrame to be adjusted and saved.
    - filename: Filename for saving the DataFrame.
    - base_columns: List of base column names to be ordered first.
    - additional_columns: List of additional column names to be included after the base columns.
    """
    # Ensure all base columns are present in the DataFrame
    final_columns = [col for col in base_columns if col in cmb_df.columns]
    
    # Add any additional columns that aren't in the base_columns list
    final_columns += [col for col in cmb_df.columns if col not in base_columns and col in additional_columns]
    
    # Reorder the DataFrame
    cmb_df = cmb_df[final_columns]
    
    # Save to CSV
    cmb_df.to_csv(filename, index=False)


def parse_metadatajson_to_dfs(meta_dict):
    # Study level DataFrame preparation
    study_level_data = {
        "subject": meta_dict["subject"],
        "seq_type": meta_dict["seq_type"],
        "healthy": meta_dict["healthy"],
        "n_CMB_old": meta_dict["n_CMB_old"],
        "n_CMB_new": meta_dict["n_CMB_new"],
        "old_shape": meta_dict["old_specs"]["shape"],
        "new_shape": meta_dict["new_specs"]["shape"],
        "old_voxel_dim": meta_dict["old_specs"]["voxel_dim"],
        "new_voxel_dim": meta_dict["new_specs"]["voxel_dim"],
        "old_orientation": meta_dict["old_specs"]["orientation"],
        "new_orientation": meta_dict["new_specs"]["orientation"]
    }
    study_level_df = pd.DataFrame([study_level_data])
    
    # CMB level DataFrames preparation
    cmb_level_old_data = []
    for cmb_id, cmb_info in meta_dict["CMBs_old"].items():
        cmb_info["subject"] = meta_dict["subject"]
        cmb_info["id"] = cmb_id
        if "region_growing" in cmb_info:
            # Flatten the region growing dictionary into the main dictionary
            cmb_info.update({"RG_" + k: v for k, v in cmb_info.pop("region_growing").items()})
            cmb_level_old_data.append(cmb_info)
    cmb_level_old_df = pd.DataFrame(cmb_level_old_data)
    
    cmb_level_new_data = []
    for cmb_id, cmb_info in meta_dict["CMBs_new"].items():
        cmb_info["subject"] = meta_dict["subject"]
        cmb_info["id"] = cmb_id
        cmb_level_new_data.append(cmb_info)
    cmb_level_new_df = pd.DataFrame(cmb_level_new_data)

    return study_level_df, cmb_level_old_df, cmb_level_new_df

def get_datasets_metadata(dataset_dir, dataset_name):
    """ 
    Read and aggregate metadata json into DataFrames for studies, old CMBs, and new CMBs.
    """
    metadata_studies = []
    metadata_cmb_old = []
    metadata_cmb_new  = []
    
    subjects = os.listdir(os.path.join(dataset_dir, "Data"))
    json_files = [os.path.join(dataset_dir, "Data", sub, "Annotations_metadata", f"{sub}_metadata.json") for sub in subjects]
    json_files = [j for j in json_files if os.path.exists(j)]
    
    # Loop through each file
    for file_path in json_files:
        with open(file_path, 'r') as file:
            metadata_dict = json.load(file)

            study_level, cmb_level_old, cmb_level_new = parse_metadatajson_to_dfs(metadata_dict)

            metadata_studies.append(study_level)
            metadata_cmb_old.append(cmb_level_old)  # Use extend for list of DataFrames
            metadata_cmb_new.append(cmb_level_new)  # Use extend for list of DataFrames

    # Concatenate all data into DataFrames
    study_df = pd.concat(metadata_studies)
    study_df['Dataset'] = dataset_name
    
    cmb_old_df = pd.concat(metadata_cmb_old, ignore_index=True)
    cmb_old_df['Dataset'] = dataset_name
    
    cmb_new_df = pd.concat(metadata_cmb_new, ignore_index=True)
    cmb_new_df['Dataset'] = dataset_name

    return study_df, cmb_old_df, cmb_new_df

def main(args):
    all_datasets = os.listdir(args.datasets_dir)
    datasets = args.datasets if args.datasets else all_datasets
    assert set(datasets) <= set(all_datasets), "Some specified datasets are not in the datasets directory."

    _logger.info(f"Will collect metadata from these datasets:")
    print(datasets)

    all_metadata_studies = []
    all_metadata_cmb_old = []
    all_metadata_cmb_new = []

    for dataset_name in datasets:
        study_df, cmb_old_df, cmb_new_df = get_datasets_metadata(os.path.join(args.datasets_dir, dataset_name), dataset_name)
        all_metadata_studies.append(study_df)
        all_metadata_cmb_old.append(cmb_old_df)
        all_metadata_cmb_new.append(cmb_new_df)

    # Concatenate study level DataFrame
    study_dataout = pd.concat(all_metadata_studies)
    
    # Adjust column order and save study level DataFrame
    adjust_and_save_study_df(study_dataout, os.path.join(args.savedir, "datasets_overview.csv"))

    # Define base and additional columns for cmb_old and cmb_new
    base_columns = ['subject', 'id', 'Dataset', 'CM', 'size', 'radius']
    additional_columns_old = [col for col in cmb_old_df.columns if col.startswith('region_growing_')]

    # Concatenate, adjust column order, and save CMB old level DataFrame
    cmb_old_dataout = pd.concat(all_metadata_cmb_old)
    adjust_and_save_cmb_df(cmb_old_dataout, os.path.join(args.savedir, "cmb_old_overview.csv"), base_columns, additional_columns_old)

    # Since cmb_new does not have additional region growing params, additional_columns is empty
    cmb_new_dataout = pd.concat(all_metadata_cmb_new)
    adjust_and_save_cmb_df(cmb_new_dataout, os.path.join(args.savedir, "cmb_new_overview.csv"), base_columns)

    _logger.info(f"Succesfully saved CSVs in the following dataset:")
    print(args.savedir)



def parse_args():
    '''
    Parses all script arguments.
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--datasets_dir', type=str, default=None,
                        help='Path to the dir with processed datasets')
    parser.add_argument('--datasets',  nargs='+', type=str, default=[],
                        help='Specific datasets to include. If None, all included in --datasets_dir')
    parser.add_argument('--savedir', type=str, default=None,
                        help='Path where to save CSV with overview')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)