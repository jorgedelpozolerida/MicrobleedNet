# -*-coding:utf-8 -*-
""" Loops over all the studies and generates the SynthSeg masks for each study.

{Long Description of Script}

@author: jorgedelpozolerida
@date: 19/05/2024
"""
import os
import sys
import argparse
import traceback
import logging
import subprocess
from multiprocessing import Pool
import nibabel as nib
from nilearn.image import resample_to_img
import numpy as np
from tqdm import tqdm
import cmbnet.utils.utils_general as utils_general

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def process_study(args, study):
    study_folder = os.path.join(args.data_dir, study)
    mri_file = os.path.join(study_folder, "MRIs", f"{study}.nii.gz")
    synthseg_output_path = os.path.join(args.output_dir, f"{study}_synthseg.nii.gz")


    if not os.path.exists(synthseg_output_path) or args.overwrite_synthseg:
        if os.path.exists(synthseg_output_path):
            logging.info(f"Overwriting for study {study}")
        utils_general.apply_synthseg(args, mri_file, synthseg_output_path, args.synthseg_repo_path)
    else:
        logging.info(f"SynthSeg mask already exists for study {study}")
        
    mri_img = nib.load(mri_file)
    synthseg_img = nib.load(synthseg_output_path)
    resampled_synthseg_img = resample_to_img(
        synthseg_img, mri_img, interpolation="nearest"
    )

    # Get the data arrays and squeeze just in case
    _logger.info(f"Resampling SynthSeg image to match MRI for {study}")
    
    # Save the resampled SynthSeg mask
    nib.save(resampled_synthseg_img, os.path.join(args.output_dir_resampled, f"{study}_synthseg_resampled.nii.gz"))

def main(args):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_devices
    studies = [
        study for study in os.listdir(args.data_dir) if not study.startswith("sMOMENI") and not study.startswith("CRBneg")  and os.path.isdir(os.path.join(args.data_dir, study))
    ]
    args.num_workers = min(args.num_workers, os.cpu_count())
    if args.num_workers == 1:
        for study in tqdm(studies, desc="Processing studies"):
            process_study(args, study)
    # else:
    #     with Pool(args.num_workers) as pool:
    #         pool.starmap(process_study, [(args, study) for study in studies])


def parse_args():
    """
    Parses all script arguments.
    """
    parser = argparse.ArgumentParser(description="Apply SynthSeg to MRI images.")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing MRI data files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for SynthSeg masks.",
    )
    parser.add_argument(
        "--output_dir_resampled",
        type=str,
        required=True,
        help="Output directory for SynthSeg masks RESAMPLED.",
    )
    parser.add_argument(
        "--synthseg_repo_path",
        type=str,
        required=True,
        help="Path to the SynthSeg repository.",
    )
    parser.add_argument(
        "--overwrite_synthseg",
        action="store_true",
        help="Overwrite existing SynthSeg outputs.",
    )
    parser.add_argument(
        "--robust_synthseg",
        action="store_true",
        help="Use robust SynthSeg processing if needed.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes to use for multiprocessing.",
    )
    parser.add_argument(
        "--visible_devices",
        type=str,
        default="0",
        help="CUDA_VISIBLE_DEVICES environment variable.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
