#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Short Description: Script to extract radiomics features for CMB studies in parallel.

Long Description: This script processes MRI and CMB data to extract radiomics features for each study using multiprocessing for enhanced performance.
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import label as nd_label
from multiprocessing import Pool, cpu_count
import ast
from logging.handlers import RotatingFileHandler
from cmbnet.utils.utils_general import calculate_radiomics_features, calculate_synthseg_features

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

# ATTENTION: this changes fomr server to server
# maps subfolder in data dir for every dataset
datasets_mapping = {
    "DOU": {"folder": "cmb_dou"},
    "CRB": {"folder": "cmb_crb"},
    "rest": {"folder": "cmb_train"},
}


def setup_logging(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # File handler with rotation
    file_handler = RotatingFileHandler(log_file, maxBytes=10485760, backupCount=5)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def process_study(study_data):
    try:
        study, data_dir, datasets_mapping, df_cmb_metadata, cache_folder = study_data
        dataset = study.split("-")[0]

        # NOTE: to remove later
        if dataset not in datasets_mapping:
            # print(f"IGNORED study {study}")
            # return f"IGNORED study {study}"
            return 

        folder = datasets_mapping.get(dataset, datasets_mapping["rest"])
        study_folder = os.path.join(data_dir, folder["folder"], "Data", study)

        mri_file = os.path.join(study_folder, "MRIs", f"{study}.nii.gz")
        cmb_file = os.path.join(study_folder, "Annotations", f"{study}.nii.gz")
        synthseg_file = os.path.join(args.synthseg_dir, f"{study}_synthseg_resampled.nii.gz")

        mri_im = nib.load(mri_file)
        mri_data = mri_im.get_fdata()
        cmb_im = nib.load(cmb_file)
        cmb_data = cmb_im.get_fdata()
        synth_im = nib.load(synthseg_file)
        synth_data = synth_im.get_fdata()

        cmbs_df = df_cmb_metadata[df_cmb_metadata["seriesUID"] == study]

        labeled_array, num_features = nd_label(
            cmb_data == 1
        )  # Ensure we're labeling the correct regions
        results = []

        for i in range(1, num_features + 1):
            # print(f"label {i} .....")
            cmb_mask_individual = labeled_array == i
            hitted_CCs = []

            for CM in cmbs_df["CM"]:
                CM = ast.literal_eval(CM)
                CM = tuple(map(int, CM))

                if cmb_mask_individual[CM]:
                    if CM not in hitted_CCs:
                        hitted_CCs.append(CM)
                        # print(f"Found CMB for CC-{i} in CM-{CM}")
                        radiomics_results = calculate_radiomics_features(
                            mri_data, cmb_mask_individual
                        )
                        synthseg_results = calculate_synthseg_features(
                            mri_data, cmb_mask_individual, synth_data
                        )
                        results.append(
                            {"seriesUID": study, "CM": CM, 
                             **radiomics_results,
                             **synthseg_results
                             }
                        )

            if not hitted_CCs:
                print(f"CC-{i} not found in metadata for study {study}")

        # Save results to cache
        output_path = os.path.join(cache_folder, f"{study}_results.csv")
        pd.DataFrame(results).to_csv(output_path, index=False)
        _logger.info(f"Processed {study} and saved results to {output_path}")

        return f"Processed {study}"
    except Exception as e:
        print(f"Error processing {study}: {e}")
        return f"Error processing {study}: {e}"


def main(data_dir, cache_folder, log_file, num_workers):
    setup_logging(log_file)
    df_cmb_metadata = pd.read_csv(args.cmb_metadata)
    studies = df_cmb_metadata["seriesUID"].unique()
    num_workers = min(num_workers, len(studies))
    
    with Pool(processes=num_workers) as pool:
        results = pool.map(
            process_study,
            [(study, data_dir, datasets_mapping, df_cmb_metadata, cache_folder) for study in studies],
        )
        for result in results:
            logging.info(result)
            
    all_files = [os.path.join(cache_folder, f) for f in os.listdir(cache_folder) if f.endswith('.csv') and f != "CMB_radiomics_metadata.csv"]
    df_list = [pd.read_csv(file) for file in all_files]
    
    combined_df = pd.concat(df_list, ignore_index=True)
    outpath = os.path.join(cache_folder, 'CMB_radiomics_metadata.csv')
    combined_df.to_csv(outpath, index=False)
    print(f"Final combined metadata saved in: {outpath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process CMB studies to extract radiomics features."
    )
    parser.add_argument(
        "--cmb_metadata",
        required=True,
        help="Path to the metadata file containing the CMB studies.",
    )
    parser.add_argument(
        "--data_dir", required=True, help="Directory where the study data is stored."
    )
    parser.add_argument(
        "--synthseg_dir", required=True, help="Directory where all SynthSeg masks are saved."
    )
    parser.add_argument(
        "--cache_folder",
        required=True,
        help="Directory where to save the intermediate results.",
    )
    parser.add_argument(
        "--log_file", required=True, help="File path for saving log output."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes to use. Default is the number of CPU cores.",
    )

    args = parser.parse_args()

    main(args.data_dir, args.cache_folder, args.log_file, args.workers)
