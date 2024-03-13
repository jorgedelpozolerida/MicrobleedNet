#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Generate splits for training data


Two splits files are generated with train-valid for:
- pretraining phase
- fine-tuning phase

Ensures equal proportions for these factors:
- field strength (1.5, 1.5/3, 3)
- resolution level (low, high)
- sequence type (SWI/T2S)
- healthy/unhealthy (in all scans through time)
- cmb per case (<=5, >5)

Also synthetic and negative extar cases are only added to training split.

Example:
python cmbnet/commands/generate_train_splits.py  
--data_dir '/datadrive_m2/jorge/datasets/cmb_train/Data' 
--prop_valid 0.25 
--seed 42 
--savedir '/datadrive_m2/jorge/MicrobleedNet/data-misc/training'

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
from tabulate import tabulate
from sklearn.model_selection import train_test_split
import json


TRAIN_DATASETS = ["RODEJA", "CRBneg", "VALDO", "MOMENI", "sMOMENI"]

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

def save_splits_to_json(train_series_uids, valid_series_uids, file_path):
    splits = {
        "train": train_series_uids,
        "valid": valid_series_uids
    }
    with open(file_path, 'w') as f:
        json.dump(splits, f, indent=4)

def log_split_details(train_df, valid_df):
    train_unique_patients = len(train_df['patientUID'].unique())
    valid_unique_patients = len(valid_df['patientUID'].unique())
    total_patients = train_unique_patients + valid_unique_patients
    
    train_proportion_p = 100 * train_unique_patients / total_patients
    valid_proportion_p = 100 * valid_unique_patients / total_patients

    train_proportion = 100 * len(train_df) / (len(train_df) + len(valid_df))
    valid_proportion = 100 * len(valid_df) / (len(train_df) + len(valid_df))

    print(f"Training split: {len(train_df)} series ({train_proportion:.2f}%) with {train_unique_patients} ({train_proportion_p:.2f}%) unique patients")
    print(f"Validation split: {len(valid_df)} series ({valid_proportion:.2f}%) with {valid_unique_patients} ({valid_proportion_p:.2f}%) unique patients")


def create_stratified_group_tables(df, columns):
    # Initialize a list to hold all the table rows
    all_table_data = []
    
    # For each column, collect stratification information
    for column_name in columns:
        counts = df[column_name].value_counts()
        proportions = (counts / len(df) * 100).round(2)  # round to 2 decimal places
        
        # Add stratification info for this column to the table data
        for value, count in counts.items():
            proportion = proportions[value]
            all_table_data.append([column_name, value, count, f"{proportion}%"])
    
    # Print the aggregated table
    print(tabulate(all_table_data, headers=['Column', 'Value', 'Count', 'Proportion (%)'], tablefmt='pretty'))
    
    return counts, proportions

def main(args):
    studies = pd.read_csv(args.studiescsv_path)
    studies.fillna("Unspecified", inplace=True)
    studies_filt = studies[studies['Dataset'].isin(TRAIN_DATASETS)]
    studies_present = [d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))]
    print(studies_present)
    # print(set(studies_filt['seriesUID'].to_list()) - set(studies_present))
    # print(set(studies_present) - set(studies_filt['seriesUID'].to_list()))
    assert set(studies_present) == set(studies_filt['seriesUID'].to_list()), "Mismatch between present studies and filtered studies"
    
    studies_filt_cmb_real = studies_filt[~studies_filt['Dataset'].isin(['sMOMENI', 'CRBneg'])]
    print(f"Filtered from {len(studies)} -->  {len(studies_filt)} --> {len(studies_filt_cmb_real)} (no synth, no neg)")
    
    columns_of_interest = ['Dataset', 'healthy_all', 'seq_type', 'res_level', 'field_strength', 'CMB_level', 'TE']
    studies_filt_cmb_real['stratify_label'] = studies_filt_cmb_real[columns_of_interest].astype(str).agg('-'.join, axis=1)

    columns_of_interest.append("stratify_label")

    print("All data strata:")
    create_stratified_group_tables(studies_filt_cmb_real, columns_of_interest)

    # Group by patientUID to make sure we don't split a single patient across train and validation
    grouped = studies_filt_cmb_real.groupby('patientUID')['stratify_label'].agg(lambda x: x.mode()[0])
    train_patientUIDs, valid_patientUIDs = train_test_split(grouped.index, test_size=args.prop_valid, stratify=grouped.values, random_state=args.seed)


    # Filter the DataFrame based on patientUID to get the corresponding train and valid DataFrames
    train_df = studies_filt_cmb_real[studies_filt_cmb_real['patientUID'].isin(train_patientUIDs)]
    valid_df = studies_filt_cmb_real[studies_filt_cmb_real['patientUID'].isin(valid_patientUIDs)]

    # Ensure no patientID is in both groups
    assert set(train_df['patientUID']).isdisjoint(set(valid_df['patientUID'])), "Some patientIDs appear in both train and valid splits"
    
    log_split_details(train_df, valid_df)

    # Then continue with stratification logging for training and validation splits as before
    print("\nTraining split stratification:")
    create_stratified_group_tables(train_df, columns_of_interest)

    print("\nValidation split stratification:")
    create_stratified_group_tables(valid_df, columns_of_interest)

    # Collect seriesUIDs for each split
    train_series_uids = train_df['seriesUID'].tolist()
    valid_series_uids = valid_df['seriesUID'].tolist()
    save_splits_to_json(train_series_uids, valid_series_uids, os.path.join(args.savedir, 'splits_without_sMOMENI_CRBneg.json'))
    print(f"Initial split without sMOMENI and CRBneg. Train: {len(train_series_uids)}, Valid: {len(valid_series_uids)}")

    # Another with sMOMENI and CRBneg
    sMOMENI_CRBneg = studies[(studies['Dataset'] == 'sMOMENI') | (studies['Dataset'] == 'CRBneg')]

    # Filter the validation DataFrame to exclude patients from sMOMENI and CRBneg datasets
    valid_patient_uids = valid_df['patientUID'].unique()

    # Now filter the sMOMENI and CRBneg DataFrame to exclude these patientUIDs
    sMOMENI_CRBneg_filtered = sMOMENI_CRBneg[~sMOMENI_CRBneg['patientUID'].isin(valid_patient_uids)]

    # Prepare seriesUID lists for training (including filtered sMOMENI and CRBneg) and validation
    train_series_uids_with_sMOMENI_CRBneg_filtered = train_series_uids + sMOMENI_CRBneg_filtered['seriesUID'].tolist()
    valid_series_uids = valid_df['seriesUID'].tolist()

    # Logging the update
    print(f"Filtered sMOMENI and CRBneg to exclude validation patientUIDs. Now, Train: {len(train_series_uids_with_sMOMENI_CRBneg_filtered)}, Valid: {len(valid_series_uids)}")
    print(f"A total of {len(sMOMENI_CRBneg) - len(sMOMENI_CRBneg_filtered)} sMOMENI/CRBneg studies removed to avoid patientID repetition in training.")

    # Save the updated splits with filtered sMOMENI and CRBneg from the training set
    save_splits_to_json(train_series_uids_with_sMOMENI_CRBneg_filtered, valid_series_uids, os.path.join(args.savedir, 'splits_with_sMOMENI_CRBneg_filtered.json'))


def parse_args():
    '''
    Parses all script arguments.
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--studiescsv_path', type=str, default="./data-misc/csv/ALL_studies.csv",
                        help='Path to the CSV with metadata')
    parser.add_argument('--data_dir', type=str, default=None, required=True,
                        help='Path to the dir with all processed studies')
    parser.add_argument('--savedir', type=str, default=None, required=True,
                        help='Path where to save splits file')
    parser.add_argument('--prop_valid', type=float, required=True,
                        help='Proportion of all training data to use for validation')
    parser.add_argument('--seed', type=int, required=True,
                        help='Seed')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)