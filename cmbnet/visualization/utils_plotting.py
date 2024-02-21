#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Script contianing utility functions for plotting


{Long Description of Script}


@author: jorgedelpozolerida
@date: 30/01/2024
"""

import os
import sys
import argparse
import traceback

import logging
import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from typing import Any, List, Tuple


################################################################################
# Region Growing plotting
################################################################################

def plot_cross_with_gap(ax: Any, x: float, y: float, color: str = 'r', gap_size: int = 5, line_length: int = 10, **kwargs: Any) -> None:
    """
    Draws a cross on the plot with a specified gap around the center, accounting for y-axis inversion.

    Args:
        ax: The axis on which to draw.
        x, y: The center coordinates of the cross, considering y-axis inversion.
        color: The color of the cross.
        gap_size: The size of the gap around the center.
        line_length: The length of each line of the cross.
    """
    # Vertical line (adjustment only needed for y-axis inversion)
    ax.plot([x, x], [y - gap_size - line_length, y - gap_size], color=color, **kwargs)
    ax.plot([x, x], [y + gap_size, y + gap_size + line_length], color=color, **kwargs)
    
    # Horizontal line
    ax.plot([x - gap_size - line_length, x - gap_size], [y, y], color=color, **kwargs)
    ax.plot([x + gap_size, x + gap_size + line_length], [y, y], color=color, **kwargs)


def check_orientation_consistency(orientation_matrices: List[Any]) -> bool:
    """
    Check if all NiBabel objects have the same orientation convention.

    Args:
        orientation_matrices: List of affine transformation matrices.

    Returns:
        bool: True if all matrices have the same orientation, False otherwise.
    """
    # Check if all NiBabel objects have the same orientation convention
    orientation_strs = [nib.aff2axcodes(mat) for mat in orientation_matrices]
    return len(set(orientation_strs)) == 1


def plot_processed_mask(mri_im: Any, cmb_im: Any, processed_cmb_im: Any, cmb_coords: Tuple[float, float, float], zoom_size: float, metadata_str: str = None, save_path: str = None) -> None:
    """
    Plot the processed mask on a matplotlib figure.

    Args:
        mri_im: NiBabel object containing MRI image.
        cmb_im: NiBabel object containing CMB image.
        processed_cmb_im: NiBabel object containing processed CMB image.
        cmb_coords: Coordinates of the CMB.
        zoom_size: Size of the zoomed region.
        metadata_str: String containing metadata information.
        save_path: Path to save the plot image.
    """
    half_size = zoom_size / 2

    # Check if all NiBabel objects have the same orientation
    orientation_matrices = [img.affine for img in [mri_im, cmb_im, processed_cmb_im]]
    if not check_orientation_consistency(orientation_matrices):
        print("Orientation matrices:")
        for i, mat in enumerate(orientation_matrices):
            print(f"Image {i+1}:")
            print(nib.aff2axcodes(mat))
        raise ValueError("All images must have the same orientation.")

    # Extract data from NiBabel objects
    mri_data = mri_im.get_fdata()
    cmb_data = cmb_im.get_fdata()
    out_mask = processed_cmb_im.get_fdata()

    # Handle 4D data by selecting the first volume
    if mri_data.ndim > 3:
        mri_data = mri_data[..., 0]
    if cmb_data.ndim > 3:
        cmb_data = cmb_data[..., 0]
    if out_mask.ndim > 3:
        out_mask = out_mask[..., 0]

    # Calculate the start and end indices for the crop
    start_idx = [max(0, int(cm - half_size)) for cm in cmb_coords[:2]]  # Crop only in the first two dimensions
    end_idx = [min(dim, int(cm + half_size)) for cm, dim in zip(cmb_coords[:2], mri_data.shape[:2])]  # Crop only in the first two dimensions

    # Create slices for cropping
    crop_slice = tuple(slice(start, end) for start, end in zip(start_idx, end_idx))

    # Apply cropping
    t2s_data_cropped = mri_data[crop_slice]
    cmb_cropped = cmb_data[crop_slice]
    pred_cmb_cropped = out_mask[crop_slice]

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))  # 2x2 layout

    # Plot original and generated masks with cross
    axs[1, 0].imshow((t2s_data_cropped[:, :, cmb_coords[-1]]), cmap='gray')  
    axs[1, 0].imshow((cmb_cropped[:, :, cmb_coords[-1]]), alpha=0.5, cmap="Reds")  
    plot_cross_with_gap(axs[1, 0], y=t2s_data_cropped.shape[0] / 2, x=t2s_data_cropped.shape[1] / 2, color='yellow', line_length=25, gap_size=15, linewidth=1)
    axs[1, 0].set_title('Original Mask')

    axs[1, 1].imshow((t2s_data_cropped[:, :, cmb_coords[-1]]), cmap='gray')  
    axs[1, 1].imshow((pred_cmb_cropped[:, :, cmb_coords[-1]]), alpha=0.5, cmap="Reds")  
    plot_cross_with_gap(axs[1, 1], y=t2s_data_cropped.shape[0] / 2, x=t2s_data_cropped.shape[1] / 2, color='yellow', line_length=25, gap_size=15, linewidth=1)
    axs[1, 1].set_title('Generated Mask')

    # Plot the full view without zoom
    axs[0, 0].imshow((mri_data[:, :, cmb_coords[-1]]), cmap='gray')
    plot_cross_with_gap(axs[0, 0], y=cmb_coords[0], x=cmb_coords[1], color='yellow', line_length=25, gap_size=15, linewidth=1)

    # Plot the full view with zoom
    axs[0, 1].imshow((t2s_data_cropped[:, :, cmb_coords[-1]]), cmap='gray')  
    plot_cross_with_gap(axs[0, 1], y=t2s_data_cropped.shape[0] / 2, x=t2s_data_cropped.shape[1] / 2, color='yellow', line_length=25, gap_size=15, linewidth=1)

    # Add metadata to the second plot
    if metadata_str:
        axs[1, 0].text(0.5, 1.15, metadata_str, transform=axs[1, 0].transAxes, fontsize=10, ha='center')

    for ax in axs.ravel():
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def generate_cmb_plots(subject, mri_im, raw_cmb, processed_cmb, cmb_metadata, plots_path, zoom_size=100):
    """
    Saves plots for zoomed cerebral microbleeds

    Parameters:
    - subject: Subject identifier.
    - mri_im: Nibabel image object for MRI data.
    - raw_cmb: Nibabel image object for center of mass/unprocessed CMB.
    - processed_cmb: Nibabel image object for processed CMB.
    - cmb_metadata: Metadata about the CMB for the subject. Key must be CMB id, and
        values metadata related.
    - plots_path: Path to save the plots.
    - zoom_size: in voxels
    """
    for i, met_i in cmb_metadata.items():
        CM = met_i['CM']
        filename_temp = os.path.join(plots_path, f"{subject}-CMB-{i}.png")

        # Start generating the metadata string
        metadata_str_parts = [
            f"Image shape: {mri_im.shape}, zoom_size={zoom_size}",
            f"CMB ---- CM={CM},  size={met_i['size']},  radius={met_i['radius']}",
        ]
        # Add optional parts only if they are present
        if region_growing := met_i.get('region_growing', {}):
            optional_keys = ['connectivity', 'intensity_mode', 'diff_mode', 'distance_th', 'size_th']
            optional_parts = [
                f'"{key}": {region_growing[key]}' for key in optional_keys if key in region_growing
            ]

            # Group the optional parts two by two and join each pair with a newline
            # If the number of optional parts is odd, the last part will stand alone without a newline at the end
            metadata_str_parts.extend(
                "\n".join(optional_parts[i : i + 2])
                for i in range(0, len(optional_parts), 2)
            )
        # Join all parts of the metadata string
        metadata_str = "\n".join(metadata_str_parts)

        # Plot and save the data
        plot_processed_mask(mri_im, raw_cmb, processed_cmb, CM, zoom_size, metadata_str=metadata_str, save_path=filename_temp)
