#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Script to rename all studies from different sources to new ids and put together


command example:
python cmbnet/commands/rename_studies.py 
--in_dir '/datadrive_m2/jorge/datasets/processed_final' 
--test_dir '/datadrive_m2/jorge/datasets/cmb_test' 
--train_dir '/datadrive_m2/jorge/datasets/cmb_train' 
--splits 'train'


@author: jorgedelpozolerida
@date: 13/03/2024
"""

import os
import sys
import argparse
import traceback


import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import glob
import shutil

import cmbnet.utils.utils_general as utils_gen

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

TRAIN_DATASETS = ["RODEJA", "CRBneg", "VALDO", "MOMENI", "sMOMENI"]
TEST_DATASETS = ['DOU', 'CRB']

def copy_dataset_logs(dataset_name, old_dataset_dir, new_dataset_dir):
    """
    Create dataset dir, copy log txt and csv files, and add a suffix with the dataset name.
    """
    log_dir  =os.path.join(new_dataset_dir, "log_processing")
    utils_gen.ensure_directory_exists(log_dir, verbose=False)
    
    txt_files = glob.glob(os.path.join(old_dataset_dir, '*.txt'))
    csv_files = glob.glob(os.path.join(old_dataset_dir, '*.csv'))
    files_to_copy = txt_files + csv_files

    for file_path in files_to_copy:
        base_name, extension = os.path.splitext(os.path.basename(file_path))
        new_file_name = f"{dataset_name}_{base_name}{extension}"
        new_file_path = os.path.join(log_dir, new_file_name)
        shutil.copy(file_path, new_file_path)
        

def copy_and_rename_study(old_series, new_series, old_study_dir, new_study_dir, plots_paths):
    """
    Moves and renames files for some subject to new naming.
    """
    utils_gen.ensure_directory_exists(new_study_dir, verbose=False)
    
    # Directories to replicate and rename their contents
    directories = ["Annotations", "Annotations_metadata", "MRIs"]

    for dir_name in directories:
        old_dir_path = os.path.join(old_study_dir, dir_name)
        if dir_name == "Annotations_metadata":
            new_dir_path = os.path.join(new_study_dir, "processing_metadata")
        else:
            new_dir_path = os.path.join(new_study_dir, dir_name)
        utils_gen.ensure_directory_exists(new_dir_path, verbose=False)

        # For each file in the old directory, copy and rename it to the new directory
        for filename in os.listdir(old_dir_path):
            old_file_path = os.path.join(old_dir_path, filename)
            new_filename = filename.replace(old_series, new_series)
            new_filename = new_filename.replace("_metadata", "")
            new_file_path = os.path.join(new_dir_path, new_filename)

            # Copy and rename the file
            shutil.copy(old_file_path, new_file_path)

    # Copy all plots to its own plots folder
    plots_folder = os.path.join(new_study_dir, "plots")
    utils_gen.ensure_directory_exists(plots_folder, verbose=False)

    for plot_path in plots_paths:
        cmb_num = plot_path.split("-")[-1]
        _, extension = os.path.splitext(plot_path)
        new_plot_path = os.path.join(plots_folder, f"{new_series}-{cmb_num}{extension}")
        shutil.copy(plot_path, new_plot_path)

def rename_all_studies(indir, studies_df, datasets_used, outdir):
    studies_df_filt = studies_df[studies_df['Dataset'].isin(datasets_used)]
    
    # Group by 'Dataset' for processing each dataset separately
    for dataset, df_group in studies_df_filt.groupby('Dataset'):
        _logger.info(f"Processing dataset: {dataset}")
        old_dataset_dir = os.path.join(indir, dataset)
        copy_dataset_logs(dataset, old_dataset_dir, outdir)

        for i, row_i in tqdm(df_group.iterrows(), total=len(df_group)):
            study_folder_old = os.path.join(old_dataset_dir, "Data", row_i['subject'])
            study_folder_new = os.path.join(outdir, "Data", row_i['seriesUID'])
            plots_pattern = os.path.join(old_dataset_dir, "plots", "post", f"{row_i['subject']}-*")
            matching_plots = glob.glob(plots_pattern)
            
            try:
                assert all([os.path.exists(p) for p in [study_folder_old] + matching_plots]), \
                    f"some paths do not exists for study {row_i['seriesUID']}"
                copy_and_rename_study(row_i['subject'], row_i['seriesUID'], study_folder_old, study_folder_new, matching_plots)
            except AssertionError as e:
                print(e)

def main(args):

    assert all([os.path.exists(p) for p in [args.in_dir, args.studiescsv_path]]), \
        f"Some of the paths provided do not exists, check input"
    [utils_gen.ensure_directory_exists(p) for p in [args.train_dir, args.test_dir]]

    datasets = os.listdir(args.in_dir)
    studies_df = pd.read_csv(args.studiescsv_path)
    print(len(studies_df))

    _logger.info(f"Datasets found in folder: {datasets}")
    
    if "train" in args.splits:
        datasets_filt = [d for d in datasets if d in TRAIN_DATASETS]
        _logger.info(f"Datasets used for TRAIN set:")
        print(datasets_filt)
        rename_all_studies(args.in_dir, studies_df, datasets_filt, args.train_dir)
    

    if "test" in args.splits:
        datasets_filt = [d for d in datasets if d in TEST_DATASETS]
        _logger.info(f"Datasets used for TEST set:")
        print(datasets_filt)
        rename_all_studies(args.in_dir, studies_df, datasets_filt, args.test_dir)
        


def parse_args():
    '''
    Parses all script arguments.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--studiescsv_path', type=str, default="./data-misc/csv/ALL_studies.csv",
                        help='Path to the CSV with metadata')
    parser.add_argument('--in_dir', type=str, default=None,
                        help='Path to the input directory with processed studies')
    parser.add_argument('--train_dir', type=str, default=None,
                        help='Path to the output directory with renamaed processed studies')
    parser.add_argument('--test_dir', type=str, default=None,
                        help='Path to the output directory with renamaed processed studies')
    parser.add_argument('--splits',  nargs='+', type=str, choices=['train', 'test'],
                        help='Specific splits to generate')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)