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
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt


def apply_2D_zoom(image, zoom_coords, zoom_size):
    if zoom_coords and zoom_size:
        x_center, y_center = zoom_coords
        half_size = zoom_size // 2

        # Ensure the zoom region doesn't exceed the image dimensions
        x_start = max(x_center - half_size, 0)
        y_start = max(y_center - half_size, 0)
        x_end = min(x_center + half_size, image.shape[1])
        y_end = min(y_center + half_size, image.shape[0])

        newim = image[y_start:y_end, x_start:x_end]
        return newim
    return image

def adjust_for_radiological_view(image: np.ndarray) -> np.ndarray:
    """
    Adjusts the image for the radiological view by rotating it 90 degrees clockwise.
    """
    return np.rot90(image)

def plot_cross_with_gap(ax, x, y, width, height, color='r', gap_size=5, line_length=10, **kwargs):
    """
    Draws a cross on the plot with a specified gap around the center, accounting for y-axis inversion.

    Args:
        ax: The axis on which to draw.
        x, y: The center coordinates of the cross, considering y-axis inversion.
        width, height: The width and height of the plotting area, used for y-axis inversion.
        color: The color of the cross.
        gap_size: The size of the gap around the center.
        line_length: The length of each line of the cross.
    """
    # Invert y-coordinate for plotting, assuming y increases downwards
    inverted_y = height - y

    # Vertical line (adjustment only needed for y-axis inversion)
    ax.plot([x, x], [inverted_y - gap_size - line_length, inverted_y - gap_size], color=color, **kwargs)
    ax.plot([x, x], [inverted_y + gap_size, inverted_y + gap_size + line_length], color=color, **kwargs)
    
    # Horizontal line
    ax.plot([x - gap_size - line_length, x - gap_size], [inverted_y, inverted_y], color=color, **kwargs)
    ax.plot([x + gap_size, x + gap_size + line_length], [inverted_y, inverted_y], color=color, **kwargs)


def plot_processed_mask(mri_data, cmb_data, out_mask, cmb_coords, 
                        seq_type, zoom_coords=None, zoom_size=None,
                        save_path=None
                        ):
    assert len(cmb_coords) == 3
    if zoom_coords is None:
        zoom_coords = cmb_coords[:2]

    fig, axs = plt.subplots(1, 2, figsize=(10, 10))

    zoomed_mri = adjust_for_radiological_view(apply_2D_zoom(mri_data[:, :, cmb_coords[-1]], zoom_coords, zoom_size))
    zoomed_mask = adjust_for_radiological_view(apply_2D_zoom(cmb_data[:, :, cmb_coords[-1]], zoom_coords, zoom_size))
    zoomed_generated_mask = adjust_for_radiological_view(apply_2D_zoom(out_mask[:, :, cmb_coords[-1]], zoom_coords, zoom_size))

    # Calculate the relative position after rotation
    if zoom_size is not None:
        rel_x = cmb_coords[0] - zoom_coords[0] + zoom_size // 2
        rel_y = cmb_coords[1] - zoom_coords[1] + zoom_size // 2
    else:
        rel_x, rel_y = cmb_coords[:2]

    width, height = zoomed_mask.shape[1], zoomed_mask.shape[0]  # Dimensions after zoom and before rotation

    # Original Annotation Overlay
    axs[0].imshow((zoomed_mri), cmap='gray')
    axs[0].imshow(zoomed_mask, alpha=0.5, cmap="Reds")
    axs[0].set_title('Original Mask')
    plot_cross_with_gap(axs[0], rel_x, rel_y, width, height, line_length=25, gap_size=10, linewidth=1)
    axs[0].axis('off')
    
    # Generated Mask Overlay
    axs[ 1].imshow((zoomed_mri), cmap='gray')
    axs[ 1].imshow((zoomed_generated_mask), alpha=0.5, cmap="Reds")
    plot_cross_with_gap(axs[1], rel_x, rel_y, width, height, line_length=25, gap_size=10, linewidth=1)
    axs[ 1].set_title('Generated Mask')
    axs[ 1].axis('off')

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()