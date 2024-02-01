#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Script to extract metadata from processed dataset

Generates csv with all metadata extracted.

TODO:
- do properly

@author: jorgedelpozolerida
@date: 31/01/2024
"""


import os
import sys
import argparse
import traceback


import logging                                                                      # NOQA E402
import numpy as np                                                                  # NOQA E402
import pandas as pd                                                                 # NOQA E402
import shutil
from tqdm import tqdm
import csv
import nibabel as nib
import re
import multiprocessing

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)
import re

import utils.utils_datasets as utils_datasets


def ensure_directory_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def get_image_metadata(nifti_path):
    
    img = nib.load(nifti_path)
    shape = img.shape
    xd, yd, zd = shape[0], shape[1], shape[2]
    zooms = img.header.get_zooms()
    dx, dy, dz = zooms[0], zooms[1], zooms[2]
    data = img.get_fdata()
    axcodes = nib.aff2axcodes(img.affine)
    
    return {
        'shape': (xd, yd, zd),
        'zooms': (dx, dy, dz),
        'data_type': img.get_data_dtype(),
        'orientation': axcodes,
        'mean_pixel': np.nanmean(data),
        'min_pixel': np.nanmin(data),
        'max_pixel': np.nanmax(data),
        'data': data
    }

def process_study_rawdataset(args, subject):
    
    subject_dir = os.path.join(args.in_dir, subject)

    niftis = [n for n in os.listdir(subject_dir) if not n.startswith('._')]

    data = {
        'subject': [],  'X_dim': [], 'Y_dim': [], 
        'Z_dim': [], 'dx': [], 'dy': [], 'dz': [], 
        'has_nan': [], 'nan_percent': [],
        'pix_mean_val': [], 'pix_min_val': [], 'pix_man_val': [], 'Space': [],
        'Description': [], 
        'MRSequence': [], 'CMB_label': [], 'CMB_npix': [], 'data_type': [], 
        'orientation': [], 'filename': [], 'full_path': []
    }
    mri = {}

    for nifti in niftis:
        match = re.search(r'sub-\w+_space-(\w+)_desc-(\w+)_(\w+).', nifti)
        full_path = os.path.join(subject_dir, nifti)
        metadata = get_image_metadata(full_path)

        data['subject'].append(subject)
        data['filename'].append(nifti)
        data['X_dim'].append(metadata['shape'][0])
        data['Y_dim'].append(metadata['shape'][1])
        data['Z_dim'].append(metadata['shape'][2])
        data['dx'].append(metadata['zooms'][0])
        data['dy'].append(metadata['zooms'][1])
        data['dz'].append(metadata['zooms'][2])
        data['pix_mean_val'].append(metadata['mean_pixel'])
        data['pix_min_val'].append(metadata['min_pixel'])
        data['pix_man_val'].append(metadata['max_pixel'])
        data['data_type'].append(metadata['data_type'])
        data['orientation'].append(metadata['orientation'])
        data['full_path'].append(full_path)
        data['has_nan'].append(np.any(np.isnan(metadata['data'])) )
        data['nan_percent'].append(np.sum(np.isnan(metadata['data']))/len(metadata['data'].flatten())*100)


        if match:
            data['Space'].append(match[1])
            data['Description'].append(match[2])
            data['MRSequence'].append(match[3])
        else:
            data['Space'].append(None)
            data['Description'].append(None)
            data['MRSequence'].append("Label")

        if "CMB" in nifti:
            unique_labels = np.unique(metadata['data'])
            amount_each = [np.sum(metadata['data'] == l) for l in unique_labels]
            cmb_label = unique_labels[np.argmin(amount_each)]
            data['CMB_label'].append(np.argmin(amount_each) if len(unique_labels) == 2 else None)
            data['CMB_npix'].append(np.sum(metadata['data'] == cmb_label) if len(unique_labels) == 2 else None)
        else:
            data['CMB_label'].append(None)
            data['CMB_npix'].append(None)
            if match:
                mri[match[3]] = full_path

    assert len(mri) == 3, f"Following study has some issue: {subject}"

    return pd.DataFrame(data)


def process_study_processeddataset(args, subject):
    
    subject_dir = os.path.join(args.in_dir, subject)
    label_dir = os.path.join(subject_dir, args.label_dir)
    
    
    label_niftis = [os.path.join(label_dir, n) for n in os.listdir(label_dir)]
    seq = os.path.basename(label_niftis[0]).split('.')[0]  # Filename without extension
    mri_dir = os.path.join(subject_dir, args.nifti_dir, seq)

    mri_niftis = [os.path.join(mri_dir, n) for n in os.listdir(mri_dir)]

    all_niftis = mri_niftis + label_niftis

    data = {
        'subject': [],  'X_dim': [], 'Y_dim': [], 
        'Z_dim': [], 'dx': [], 'dy': [], 'dz': [], 
        'has_nan': [], 'nan_percent': [],
        'pix_mean_val': [], 'pix_min_val': [], 'pix_man_val': [],  
        'CMB_npix': [], 'data_type': [], 
        'orientation': [], 'filename': [], 'full_path': []
    }

    for nifti_path in all_niftis:
        
        metadata = get_image_metadata(nifti_path)

        data['subject'].append(subject)
        data['filename'].append(os.path.basename(nifti_path).split('/')[-1])
        data['X_dim'].append(metadata['shape'][0])
        data['Y_dim'].append(metadata['shape'][1])
        data['Z_dim'].append(metadata['shape'][2])
        data['dx'].append(metadata['zooms'][0])
        data['dy'].append(metadata['zooms'][1])
        data['dz'].append(metadata['zooms'][2])
        data['pix_mean_val'].append(metadata['mean_pixel'])
        data['pix_min_val'].append(metadata['min_pixel'])
        data['pix_man_val'].append(metadata['max_pixel'])
        data['data_type'].append(metadata['data_type'])
        data['orientation'].append(metadata['orientation'])
        data['full_path'].append(nifti_path)
        data['has_nan'].append(np.any(np.isnan(metadata['data'])) )
        data['nan_percent'].append(np.sum(np.isnan(metadata['data']))/len(metadata['data'].flatten())*100)


        if args.label_dir in nifti_path:
            unique_labels = np.unique(metadata['data'])
            amount_each = [np.sum(metadata['data'] == l) for l in unique_labels]
            cmb_label = unique_labels[np.argmin(amount_each)]
            data['CMB_npix'].append(np.sum(metadata['data'] == cmb_label) if len(unique_labels) == 2 else None)
        else:
            data['CMB_npix'].append(None)

    return pd.DataFrame(data)

def process_study(args, subject):

    if args.processed_struct:
        return process_study_processeddataset(args, subject)
    else:
        return process_study_rawdataset(args, subject)
    


    
def worker(args_subject):
    '''
    Worker function for parallel processing of subjects.
    '''
    args, subject = args_subject
    try:
        return process_study(args, subject)
    except Exception as e:
        traceback.print_exc()
        _logger.error(f"Error processing subject {subject}: {e}")
        return None




def main(args):

    # Handle paths
    if args.processed_struct:
        # args.nifti_dir = "MRIs"
        args.nifti_dir = "images"
        # args.label_dir = "Annotations"
        args.label_dir = "segmentations"
        
        
    # Ignore folder starting with weird symbol
    subjects = [d for d in os.listdir(args.in_dir) if os.path.isdir(os.path.join(args.in_dir, d))]

    # Create a multiprocessing pool
    pool = multiprocessing.Pool(args.num_workers)

    # Use the pool to process the subjects in parallel
    all_dataframes = pool.map(worker, [(args, subject) for subject in subjects])
    pool.close()
    pool.join()

    # Filter out any None values from failed processes
    all_dataframes = [df for df in all_dataframes if df is not None]

    # Concatenate all dataframes and save to CSV
    df_global = pd.concat(all_dataframes, ignore_index=True)
    df_global.to_csv(os.path.join(args.out_dir, f'{args.dataset_name}_metadata.csv'), index=False)

def parse_args():
    '''
    Parses all script arguments.
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, default=None, required=True,
                        help='Name of dataset being processed')
    parser.add_argument('--in_dir', type=str, default=None, required=True,
                        help='Path to the input directory of dataset. Must contains subject dirs as directories.')
    parser.add_argument('--out_dir', type=str, default=None, required=True,
                        help='Path to the output directory to save dataset')
    parser.add_argument('--processed_struct', action='store_true', default=False,
                        help='Add this flag if your data is in processed folder structure already')
    parser.add_argument('--num_workers', type=int, default=5,
                            help='Number of workers running in parallel')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
    