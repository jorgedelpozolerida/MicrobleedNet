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
import seaborn as sns

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from typing import Any, List, Tuple
from scipy.ndimage import label as nd_label
from scipy.ndimage import center_of_mass
from skimage.segmentation import find_boundaries
from skimage.morphology import binary_dilation
import random


################################################################################
# Region Growing plotting
################################################################################


def plot_cross_with_gap(
    ax: Any,
    x: float,
    y: float,
    color: str = "r",
    gap_size: int = 5,
    line_length: int = 10,
    **kwargs: Any,
) -> None:
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


def plot_processed_mask(
    mri_im: Any,
    cmb_im: Any,
    processed_cmb_im: Any,
    cmb_coords: Tuple[float, float, float],
    zoom_size: float,
    metadata_str: str = None,
    save_path: str = None,
) -> None:
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
    start_idx = [
        max(0, int(cm - half_size)) for cm in cmb_coords[:2]
    ]  # Crop only in the first two dimensions
    end_idx = [
        min(dim, int(cm + half_size))
        for cm, dim in zip(cmb_coords[:2], mri_data.shape[:2])
    ]  # Crop only in the first two dimensions

    # Create slices for cropping
    crop_slice = tuple(slice(start, end) for start, end in zip(start_idx, end_idx))

    # Apply cropping
    t2s_data_cropped = mri_data[crop_slice]
    cmb_cropped = cmb_data[crop_slice]
    pred_cmb_cropped = out_mask[crop_slice]

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))  # 2x2 layout

    # Plot original and generated masks with cross
    axs[1, 0].imshow((t2s_data_cropped[:, :, cmb_coords[-1]]), cmap="gray")
    axs[1, 0].imshow((cmb_cropped[:, :, cmb_coords[-1]]), alpha=0.5, cmap="Reds")
    plot_cross_with_gap(
        axs[1, 0],
        y=t2s_data_cropped.shape[0] / 2,
        x=t2s_data_cropped.shape[1] / 2,
        color="yellow",
        line_length=25,
        gap_size=15,
        linewidth=1,
    )
    axs[1, 0].set_title("Original Mask")

    axs[1, 1].imshow((t2s_data_cropped[:, :, cmb_coords[-1]]), cmap="gray")
    axs[1, 1].imshow((pred_cmb_cropped[:, :, cmb_coords[-1]]), alpha=0.5, cmap="Reds")
    plot_cross_with_gap(
        axs[1, 1],
        y=t2s_data_cropped.shape[0] / 2,
        x=t2s_data_cropped.shape[1] / 2,
        color="yellow",
        line_length=25,
        gap_size=15,
        linewidth=1,
    )
    axs[1, 1].set_title("Generated Mask")

    # Plot the full view without zoom
    axs[0, 0].imshow((mri_data[:, :, cmb_coords[-1]]), cmap="gray")
    plot_cross_with_gap(
        axs[0, 0],
        y=cmb_coords[0],
        x=cmb_coords[1],
        color="yellow",
        line_length=25,
        gap_size=15,
        linewidth=1,
    )

    # Plot the full view with zoom
    axs[0, 1].imshow((t2s_data_cropped[:, :, cmb_coords[-1]]), cmap="gray")
    plot_cross_with_gap(
        axs[0, 1],
        y=t2s_data_cropped.shape[0] / 2,
        x=t2s_data_cropped.shape[1] / 2,
        color="yellow",
        line_length=25,
        gap_size=15,
        linewidth=1,
    )

    # Add metadata to the second plot
    if metadata_str:
        axs[1, 0].text(
            0.5,
            1.15,
            metadata_str,
            transform=axs[1, 0].transAxes,
            fontsize=10,
            ha="center",
        )

    for ax in axs.ravel():
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def plot_processed_mask_2x3(
    mri_im: Any,
    cmb_im: Any,
    processed_cmb_im: Any,
    cmb_coords: Tuple[float, float, float],
    zoom_size: float,
    metadata_str: str = None,
    save_path: str = None,
    subplot_spacing=0.005,
) -> None:
    """
    Plot MRI, CMB, and processed CMB images in a 2x3 subplot layout, highlighting the specified CMB location across axial, sagittal, and coronal views.

    Args:
        mri_im: NiBabel object containing MRI image data.
        cmb_im: NiBabel object containing original CMB mask data.
        processed_cmb_im: NiBabel object containing processed CMB mask data.
        cmb_coords: Tuple specifying the coordinates (x, y, z) of the CMB.
        zoom_size: The diameter of the zoom window centered on the CMB coordinates.
        metadata_str: Optional string containing metadata to be displayed.
        save_path: Optional path to save the generated plot image.
    """

    half_size = zoom_size / 2

    # Verify orientation consistency across images
    if not check_orientation_consistency(
        [mri_im.affine, cmb_im.affine, processed_cmb_im.affine]
    ):
        raise ValueError("All images must have the same orientation.")

    # Extracting data and handling 4D images by selecting the first volume
    mri_data = np.squeeze(mri_im.get_fdata())
    cmb_data = np.squeeze(cmb_im.get_fdata())
    out_mask = np.squeeze(processed_cmb_im.get_fdata())

    vmin = mri_data.min()
    vmax = mri_data.max()

    # Define cropping slices for axial, sagittal, and coronal views
    axial_crop = [
        slice(
            max(0, int(cmb_coords[i] - half_size)),
            min(dim, int(cmb_coords[i] + half_size)),
        )
        for i, dim in enumerate(mri_data.shape)
    ]
    sagittal_crop = axial_crop.copy()
    coronal_crop = axial_crop.copy()

    # Adjust for full range in non-cropped dimensions
    coronal_crop[1] = slice(None)
    sagittal_crop[0] = slice(None)
    axial_crop[2] = slice(None)  # Ensure full range for z in axial

    # Convert slices to tuples for indexing
    sagittal_crop = tuple(sagittal_crop)
    coronal_crop = tuple(coronal_crop)
    axial_crop = tuple(axial_crop)

    # Cropping data for each view
    t2s_cropped_dict = {
        "axial": mri_data[axial_crop],
        "sagittal": mri_data[sagittal_crop],
        "coronal": mri_data[coronal_crop],
    }
    cmb_cropped_dict = {
        "axial": cmb_data[axial_crop],
        "sagittal": cmb_data[sagittal_crop],
        "coronal": cmb_data[coronal_crop],
    }
    processed_cropped_dict = {
        "axial": out_mask[axial_crop],
        "sagittal": out_mask[sagittal_crop],
        "coronal": out_mask[coronal_crop],
    }

    # Initialize figure and axes for 2x3 layout
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    # Plot the full view without zoom
    axs[0, 0].imshow(mri_data[:, :, cmb_coords[-1]], cmap="gray")
    plot_cross_with_gap(
        axs[0, 0],
        y=cmb_coords[0],
        x=cmb_coords[1],
        color="yellow",
        line_length=25,
        gap_size=15,
        linewidth=1,
    )

    # Plot the full view with zoom on axial
    axs[0, 1].imshow(t2s_cropped_dict["axial"][:, :, cmb_coords[-1]], cmap="gray")
    plot_cross_with_gap(
        axs[0, 1],
        y=t2s_cropped_dict["axial"].shape[0] / 2,
        x=t2s_cropped_dict["axial"].shape[1] / 2,
        color="yellow",
        line_length=25,
        gap_size=15,
        linewidth=1,
    )

    # Plot original and generated masks with cross on axial
    axs[0, 2].imshow(t2s_cropped_dict["axial"][:, :, cmb_coords[-1]], cmap="gray")
    axs[0, 2].imshow(
        cmb_cropped_dict["axial"][:, :, cmb_coords[-1]], alpha=0.5, cmap="Reds"
    )
    plot_cross_with_gap(
        axs[0, 2],
        y=t2s_cropped_dict["axial"].shape[0] / 2,
        x=t2s_cropped_dict["axial"].shape[1] / 2,
        color="yellow",
        line_length=25,
        gap_size=15,
        linewidth=1,
    )

    # Plotting cropped views for axial, sagittal, and coronal with processed masks
    for i, view in enumerate(["axial", "sagittal", "coronal"]):
        img_slice, mask_slice = None, None
        if view == "axial":
            ind = 2
            slice_index = cmb_coords[ind]  # Z-axis for axial
            img_slice = t2s_cropped_dict[view][:, :, slice_index]
            mask_slice = processed_cropped_dict[view][:, :, slice_index]
            y_cross = t2s_cropped_dict[view].shape[0] / 2
            x_cross = t2s_cropped_dict[view].shape[1] / 2
        elif view == "sagittal":
            ind = 0
            slice_index = cmb_coords[ind]  # X-axis for sagittal
            img_slice = t2s_cropped_dict[view][slice_index, :, :]
            mask_slice = processed_cropped_dict[view][slice_index, :, :]
            y_cross = t2s_cropped_dict[view].shape[1] / 2
            x_cross = t2s_cropped_dict[view].shape[2] / 2
        elif view == "coronal":
            ind = 1
            slice_index = cmb_coords[ind]  # Y-axis for coronal
            img_slice = t2s_cropped_dict[view][:, slice_index, :]
            mask_slice = processed_cropped_dict[view][:, slice_index, :]
            y_cross = t2s_cropped_dict[view].shape[0] / 2
            x_cross = t2s_cropped_dict[view].shape[2] / 2

        axs[1, i].imshow(img_slice, cmap="gray")
        axs[1, i].imshow(mask_slice, alpha=0.5, cmap="Reds")

        # plot_cross_with_gap(axs[1, i], y=y_cross, x=x_cross, color='yellow', line_length=25, gap_size=15, linewidth=1)

    # Optionally add metadata
    if metadata_str:
        fig.suptitle(metadata_str)
    # Hide axes and show layout
    for ax in axs.ravel():
        ax.axis("off")

    fig.subplots_adjust(
        wspace=0,
        hspace=0,
        left=subplot_spacing,
        right=1 - subplot_spacing,
        bottom=0,
        top=1,
    )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


# def plot_brain(mri_im: Any, cmb_im: Any, processed_cmb_im: Any, cmb_coords: Tuple[float, float, float], zoom_size: float, metadata_str: str = None, save_path: str = None, subplot_spacing = 0.005) -> None:

#     # Verify orientation consistency across images
#     if not check_orientation_consistency([mri_im.affine, cmb_im.affine, processed_cmb_im.affine]):
#         raise ValueError("All images must have the same orientation.")

#     # Extracting data and handling 4D images by selecting the first volume
#     mri_data = np.squeeze(mri_im.get_fdata())

#     vmin = mri_data.min()
#     vmax = mri_data.max()

#     # Initialize figure and axes for 2x3 layout
#     fig, ax = plt.subplots()


#     # Plot the full view without zoom
#     ax.imshow(mri_data[:, :, cmb_coords[-1]], cmap='gray', vmin=vmin, vmax=vmax)
#     plot_cross_with_gap(ax, y=cmb_coords[0], x=cmb_coords[1], color='yellow', line_length=25, gap_size=15, linewidth=1)

#     ax.axis('off')
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300)
#     plt.close(fig)


def plot_brain(
    mri_im,
    cmb_im,
    processed_cmb_im,
    cmb_coords,
    zoom_size,
    metadata_str=None,
    save_path=None,
):
    """
    Plot MRI slices in axial, sagittal, and coronal views centered on a specific coordinate.

    Args:
        mri_im: MRI image data (nibabel image object).
        cmb_im: CMB mask data (nibabel image object).
        processed_cmb_im: Processed CMB mask data (nibabel image object).
        cmb_coords: Coordinates to center the crosshair (x, y, z).
        zoom_size: The size of the area around the coordinates to display.
        metadata_str: Optional string to display as title or metadata.
        save_path: Path to save the resulting image.
    """
    # Verify orientation consistency across images
    if not all(
        np.allclose(mri_im.affine, img.affine) for img in [cmb_im, processed_cmb_im]
    ):
        raise ValueError("All images must have the same orientation.")

    # Extract the image data
    mri_data = np.squeeze(mri_im.get_fdata())
    cmb_data = np.squeeze(cmb_im.get_fdata())
    processed_data = np.squeeze(processed_cmb_im.get_fdata())

    # Determine intensity range for MRI images
    vmin, vmax = mri_data.min(), mri_data.max()

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Titles for each subplot
    titles = ["Axial View", "Coronal View", "Sagittal View"]
    # Indexing for the cmb_coords (x, y, z)
    slices = [cmb_coords[2], cmb_coords[1], cmb_coords[0]]

    for ax, slice_idx, title in zip(axes, slices, titles):
        if title == "Axial View":
            image_slice = mri_data[:, :, slice_idx]

        elif title == "Coronal View":
            image_slice = mri_data[:, slice_idx, :]

        else:  # Sagittal View
            image_slice = mri_data[slice_idx, :, :]

        ax.imshow(image_slice, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
        if title == "Axial View":
            plot_cross_with_gap(
                ax,
                y=cmb_coords[0],
                x=cmb_coords[1],
                color="yellow",
                line_length=25,
                gap_size=15,
                linewidth=1,
            )
        elif title == "Coronal View":
            plot_cross_with_gap(
                ax,
                y=cmb_coords[0],
                x=cmb_coords[2],
                color="yellow",
                line_length=25,
                gap_size=15,
                linewidth=1,
            )
        elif title == "Sagittal View":
            plot_cross_with_gap(
                ax,
                y=cmb_coords[1],
                x=cmb_coords[2],
                color="yellow",
                line_length=25,
                gap_size=15,
                linewidth=1,
            )
        # ax.set_title(title)
        ax.axis("off")
    # axs[0, 0].imshow(mri_data[:, :, cmb_coords[-1]], cmap='gray')
    # plot_cross_with_gap(axs[0, 0], y=cmb_coords[0], x=cmb_coords[1], color='yellow', line_length=25, gap_size=15, linewidth=1)

    # Adding optional metadata
    if metadata_str:
        plt.suptitle(metadata_str)

    plt.tight_layout()

    # Hide axes and show layout
    for ax in axes.ravel():
        ax.axis("off")

    # Save or display the image
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_processed_mask_3x3(
    mri_im: Any,
    cmb_im: Any,
    processed_cmb_im: Any,
    cmb_coords: Tuple[float, float, float],
    zoom_size: float,
    metadata_str: str = None,
    save_path: str = None,
    subplot_spacing=0.005,
) -> None:
    """
    Plot MRI, CMB, and processed CMB images in a 2x3 subplot layout, highlighting the specified CMB location across axial, sagittal, and coronal views.

    Args:
        mri_im: NiBabel object containing MRI image data.
        cmb_im: NiBabel object containing original CMB mask data.
        processed_cmb_im: NiBabel object containing processed CMB mask data.
        cmb_coords: Tuple specifying the coordinates (x, y, z) of the CMB.
        zoom_size: The diameter of the zoom window centered on the CMB coordinates.
        metadata_str: Optional string containing metadata to be displayed.
        save_path: Optional path to save the generated plot image.
    """

    half_size = zoom_size / 2

    # Verify orientation consistency across images
    if not check_orientation_consistency(
        [mri_im.affine, cmb_im.affine, processed_cmb_im.affine]
    ):
        raise ValueError("All images must have the same orientation.")

    # Extracting data and handling 4D images by selecting the first volume
    mri_data = np.squeeze(mri_im.get_fdata())
    cmb_data = np.squeeze(cmb_im.get_fdata())
    out_mask = np.squeeze(processed_cmb_im.get_fdata())

    vmin = mri_data.min()
    vmax = mri_data.max()

    # Define cropping slices for axial, sagittal, and coronal views
    axial_crop = [
        slice(
            max(0, int(cmb_coords[i] - half_size)),
            min(dim, int(cmb_coords[i] + half_size)),
        )
        for i, dim in enumerate(mri_data.shape)
    ]
    sagittal_crop = axial_crop.copy()
    coronal_crop = axial_crop.copy()

    # Adjust for full range in non-cropped dimensions
    coronal_crop[1] = slice(None)
    sagittal_crop[0] = slice(None)
    axial_crop[2] = slice(None)  # Ensure full range for z in axial

    # Convert slices to tuples for indexing
    sagittal_crop = tuple(sagittal_crop)
    coronal_crop = tuple(coronal_crop)
    axial_crop = tuple(axial_crop)

    # Cropping data for each view
    t2s_cropped_dict = {
        "axial": mri_data[axial_crop],
        "sagittal": mri_data[sagittal_crop],
        "coronal": mri_data[coronal_crop],
    }
    cmb_cropped_dict = {
        "axial": cmb_data[axial_crop],
        "sagittal": cmb_data[sagittal_crop],
        "coronal": cmb_data[coronal_crop],
    }
    processed_cropped_dict = {
        "axial": out_mask[axial_crop],
        "sagittal": out_mask[sagittal_crop],
        "coronal": out_mask[coronal_crop],
    }

    # Initialize figure and axes for 2x3 layout
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))

    # # Plot the full view without zoom
    # axs[0, 0].imshow(mri_data[:, :, cmb_coords[-1]], cmap='gray')
    # plot_cross_with_gap(axs[0, 0], y=cmb_coords[0], x=cmb_coords[1], color='yellow', line_length=25, gap_size=15, linewidth=1)

    # # Plot the full view with zoom on axial
    # axs[0, 1].imshow(t2s_cropped_dict['axial'][:, :, cmb_coords[-1]], cmap='gray')
    # plot_cross_with_gap(axs[0, 1], y=t2s_cropped_dict['axial'].shape[0] / 2, x=t2s_cropped_dict['axial'].shape[1] / 2, color='yellow', line_length=25, gap_size=15, linewidth=1)

    # # Plot original and generated masks with cross on axial
    # axs[0, 2].imshow(t2s_cropped_dict['axial'][:, :, cmb_coords[-1]], cmap='gray')
    # axs[0, 2].imshow(cmb_cropped_dict['axial'][:, :, cmb_coords[-1]], alpha=0.5, cmap="Reds")
    # plot_cross_with_gap(axs[0, 2], y=t2s_cropped_dict['axial'].shape[0] / 2, x=t2s_cropped_dict['axial'].shape[1] / 2, color='yellow', line_length=25, gap_size=15, linewidth=1)

    # Plotting cropped views for axial, sagittal, and coronal with processed masks
    for i, view in enumerate(["axial", "sagittal", "coronal"]):
        img_slice, mask_slice = None, None
        if view == "axial":
            ind = 2
            slice_index = cmb_coords[ind]  # Z-axis for axial
            img_slice = t2s_cropped_dict[view][:, :, slice_index]
            mask_slice = processed_cropped_dict[view][:, :, slice_index]
            gt_slice = cmb_cropped_dict[view][:, :, slice_index]

        elif view == "sagittal":
            ind = 0
            slice_index = cmb_coords[ind]  # X-axis for sagittal
            img_slice = t2s_cropped_dict[view][slice_index, :, :]
            mask_slice = processed_cropped_dict[view][slice_index, :, :]
            gt_slice = cmb_cropped_dict[view][slice_index, :, :]

        elif view == "coronal":
            ind = 1
            slice_index = cmb_coords[ind]  # Y-axis for coronal
            img_slice = t2s_cropped_dict[view][:, slice_index, :]
            mask_slice = processed_cropped_dict[view][:, slice_index, :]
            gt_slice = cmb_cropped_dict[view][:, slice_index, :]

        axs[0, i].imshow(
            img_slice,
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
        )
        plot_cross_with_gap(
            axs[0, i],
            y=t2s_cropped_dict["axial"].shape[0] / 2,
            x=t2s_cropped_dict["axial"].shape[1] / 2,
            color="yellow",
            line_length=25,
            gap_size=15,
            linewidth=1,
        )

        axs[1, i].imshow(
            img_slice,
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
        )
        axs[1, i].imshow(gt_slice, alpha=0.5, cmap="Reds")

        axs[2, i].imshow(
            img_slice,
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
        )
        axs[2, i].imshow(mask_slice, alpha=0.5, cmap="Reds")

    # Optionally add metadata
    if metadata_str:
        fig.suptitle(metadata_str)
    # Hide axes and show layout
    for ax in axs.ravel():
        ax.axis("off")

    fig.subplots_adjust(
        wspace=0,
        hspace=0,
        left=subplot_spacing,
        right=1 - subplot_spacing,
        bottom=0,
        top=1,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def generate_cmb_plots(
    subject, mri_im, raw_cmb, processed_cmb, cmb_metadata, plots_path, zoom_size=100
):
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
        CM = met_i["CM"]
        filename_temp = os.path.join(plots_path, f"{subject}-CMB-{i}.png")

        # Start generating the metadata string
        metadata_str_parts = [
            f"Im. shape: {mri_im.shape}, zoom_size={zoom_size}",
            f"CM={CM},  size={met_i['size']},  radius={met_i['radius']}",
        ]
        # Add optional parts only if they are present
        if region_growing := met_i.get("region_growing", {}):
            optional_keys = [
                "distance_th",
                "size_th",
                "connectivity",
                "intensity_mode",
                "diff_mode",
                "sphericity_ind",
            ]
            optional_parts = [
                f'"{region_growing[key]}"'
                for key in optional_keys
                if key in region_growing
            ]

            metadata_str_parts.extend(optional_parts)
        # Join all parts of the metadata string with a comma and a space for a single-line string
        metadata_str = ", ".join(metadata_str_parts)
        # Plot and save the data
        plot_processed_mask_2x3(
            mri_im,
            raw_cmb,
            processed_cmb,
            CM,
            zoom_size,
            metadata_str=metadata_str,
            save_path=filename_temp,
        )


################################################################################
# CMB finding
################################################################################


def get_FP_coords(ground_truth, prediction):

    # Early return if there are no positives in prediction
    if np.sum(prediction) == 0:
        return []

    labeled_array, num_features = nd_label(prediction)
    com_list = center_of_mass(
        prediction, labels=labeled_array, index=np.arange(1, num_features + 1)
    )
    com_list = [(int(coord[0]), int(coord[1]), int(coord[2])) for coord in com_list]

    FP_list = []
    for com in com_list:
        com = tuple(com)
        # Efficient check if a COM overlaps with ground_truth
        if ground_truth[com] == 0:
            FP_list.append(com)

    return FP_list


################################################################################
# Report
################################################################################


def create_boxplot(
    data,
    column,
    group_by,
    title,
    xlabel=None,
    ylabel=None,
    figsize=(12, 6),
    rotation=45,
):
    """
    Creates a boxplot for a specified column grouped by a specified group column.

    Parameters:
    - data (pd.DataFrame): DataFrame containing the data.
    - column (str): Column name for which to create the boxplot.
    - group_by (str): Column name to group by for the boxplot.
    - title (str): Title of the plot.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    - figsize (tuple): Dimensions of the figure (width, height).
    - rotation (int): Degrees to rotate x-axis labels.
    """
    plt.figure(figsize=figsize)
    sns.boxplot(x=group_by, y=column, data=data)
    plt.title(title)
    plt.xlabel(xlabel if xlabel else group_by)
    plt.ylabel(ylabel if ylabel else column)
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    plt.show()
