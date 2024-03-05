#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Script to download into folder s3 studies

@author: jorgedelpozolerida
@date: 03/03/2024
"""

import os
import sys
import argparse
import traceback
from datetime import datetime

import logging
import numpy as np
import pandas as pd
import shutil


logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

sys.path.append("/home/cerebriu/data/DM/data-management/aws_s3/")
BUCKET_NAME = "cerebriu-data-management-bucket"
import boto3
import utils_s3
from tqdm import tqdm 
from typing import Tuple, List, Union, Any, Dict


def log_message(log_file_path, messages):
    """
    Writes a consolidated log message for a study to a log file.
    Each message is timestamped and the entire log for a study is framed with separators for better readability.

    Parameters:
    - log_file_path (str): Path to the log file where messages will be written.
    - messages (list): A list of strings representing log messages to be written for a single study. These messages include both the operations performed (e.g., file downloads) and any errors encountered.

    The function opens the log file in append mode, ensuring that new messages are added to the end of the file without overwriting existing content. It formats each message with a current timestamp before writing it to the file, enhancing the log's usefulness by providing temporal context for each operation.
    """
    with open(log_file_path, "a") as log_file:
        # Write study header
        log_file.write("=" * 80 + "\n")
        # Write each message
        for message in messages:
            # Format the current time
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            formatted_message = f"[{current_time}] {message}"
            log_file.write(formatted_message + "\n")
        # Write footer
        log_file.write("=" * 80 + "\n\n")

def download_study_data(s3, dataset_name: str, study_uid: str, savedir: str, log_file_path: str) -> int:
    """
    Downloads all files from a specific S3 folder based on the given study UID and dataset name.
    It applies criteria to download only relevant .nii or .nii.gz files based on filename content.
    The function logs its actions and any errors encountered to a specified log file, providing a comprehensive record of the download process for each study.

    Parameters:
    - s3: AWS S3 client instance used to access the S3 service.
    - dataset_name (str): The name of the dataset to look for in the S3 bucket. This is part of the S3 folder path from which files will be downloaded.
    - study_uid (str): The unique identifier for the study of interest. Used to construct the S3 folder path and to create a local directory for downloaded files.
    - savedir (str): The local directory where downloaded files will be stored. A subdirectory with the study UID will be created here.
    - log_file_path (str): Path to the log file where the download process will be logged. The function will record the start of the download, the success of each file download, and any errors, as well as a summary message.

    Returns:
    - int: 1 if the download process completes successfully for all relevant files, 0 if an error occurs during the process.

    The function attempts to download all relevant files from the specified S3 folder, skipping those that do not meet the criteria for .nii or .nii.gz files of interest. It handles exceptions by logging error messages and returns a status code indicating the success or failure of the download process.
    """
    folder_aws_key = f"{dataset_name}/Restructured_Data_Annotations_V1/{study_uid}/"
    local_save_dir = os.path.join(savedir, study_uid)
    os.makedirs(local_save_dir, exist_ok=True)

    messages = [f"Starting download for study UID: {study_uid}"]
    downloaded_files = []

    try:
        object_list = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=folder_aws_key)
        if 'Contents' not in object_list:
            raise Exception(f"No contents found in {folder_aws_key}")

        save_paths = []
        for obj in object_list['Contents']:
            file_name = obj['Key'].split('/')[-1]
            if file_name and (file_name.endswith(('.nii', '.nii.gz')) and "SWI" not in file_name and "T2S" not in file_name):
                continue
            if "not_needed" in obj['Key']:
                continue
            if "EXCLUDED_" in file_name:
                messages.append("ATTENTION: Exclusion file found, study will be ignored and folder removed")
                shutil.rmtree(local_save_dir)
                log_message(log_file_path, messages)
                return 0
            
            save_paths.append((file_name, os.path.join(local_save_dir, file_name), obj['Key']))
        for fname, local_path, s3_path in save_paths:
            s3.download_file(BUCKET_NAME, s3_path, local_path)
            downloaded_files.append(fname)
            messages.append(f"Successfully downloaded: {fname}")

    except Exception as e:
        messages.append(f"Could not find or download files from AWS: {folder_aws_key}. Error: {str(e)}. Will now delete subdirectory.")
        shutil.rmtree(local_save_dir)
        log_message(log_file_path, messages)
        return 0

    if downloaded_files:
        messages.append(f"Downloaded {len(downloaded_files)} files for study {study_uid}.")
    else:
        messages.append(f"No files were downloaded for study {study_uid}. Will delete subdir now")
        shutil.rmtree(local_save_dir)

    log_message(log_file_path, messages)
    return 1

def main(args):

    s3 = boto3.client('s3')
    df = pd.read_csv(args.studiescsv_path)

    assert "StudyInstanceUID" in df.columns, "Please provide CSV with StudyInstanceUID column"

    if "Dataset" not in df.columns:
        _logger.info(
            "No 'Dataset' column provided, will obtain dataset querying s3 for every study"
        )
        utils_s3.confirm_execution()
        s3 = boto3.client('s3')
        df['Dataset'] = [utils_s3.find_s3_dataset(s3, BUCKET_NAME, x) for x in tqdm(df['StudyInstanceUID'], desc="Finding dataset for input studies")]
        df.to_csv(os.path.join(args.cachefolder, "study_dataset_list.csv"))
    
    # Handle missing studies
    df_nodataset = df[df['Dataset'].isnull()]
    if df_nodataset.shape[0] > 0:
        _logger.warning(f"Could not find Dataset column for these studies: {df_nodataset['StudyInstanceUID'].to_list()}")

    else:
        _logger.info("Succesfully found Dataset column for all studies")

    df_clean = df[~df['StudyInstanceUID'].isin(df_nodataset['StudyInstanceUID'])]
    df_clean = df_clean.sort_values("Dataset")

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    log_file_path = os.path.join(args.logdir, f"download_LOG_{current_time}.txt")
    _logger.info("Log file:")
    print(log_file_path)
    _logger.info(f"Will download a total of {len(df_clean)}")
    utils_s3.confirm_execution()
    for i, row in tqdm(df_clean.iterrows(), desc="Downloading studies", total= len(df_clean)):
        download_study_data(s3, row['Dataset'], row['StudyInstanceUID'], args.savedir, log_file_path)


def parse_args():
    '''
    Parses all script arguments.
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--studiescsv_path', type=str, default=None,
                        help='CSV containing StudyInstanceUID and possibly Dataset')
    parser.add_argument('--savedir', type=str, default=None,
                        help='Path where to save studies')
    parser.add_argument('--cachefolder', type=str, default=os.getcwd(),
                        help='Path where to save cache files. Current wd if not specified.')
    parser.add_argument('--logdir', type=str, default=None, required=True,
                        help='Path where to save log file')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
