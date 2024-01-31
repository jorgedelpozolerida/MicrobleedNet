#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" {Short Description of Script}


{Long Description of Script}


@author: jorgedelpozolerida
@date: 16/01/2024
"""

import os
import sys
import argparse
import traceback


import logging                                                                      # NOQA E402
import numpy as np                                                                  # NOQA E402
import pandas as pd                                                                 # NOQA E402

import numpy as np
import nibabel as nib
import nilearn as nil
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
import pickle
import os
from collections import deque
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure, \
    center_of_mass, binary_fill_holes
from scipy.ndimage import generate_binary_structure, binary_closing, binary_dilation, binary_erosion, center_of_mass
from scipy.ndimage import label as nd_label
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from scipy.spatial.distance import cdist
import heapq
from kneed import KneeLocator

current_dir_path = os.path.dirname(os.path.abspath(__file__))
parent_dir_path = os.path.abspath(os.path.join(current_dir_path, os.pardir))
sys.path.append(parent_dir_path)

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)



##############################################################################
###################            Region Growing              ###################
##############################################################################


def is_within_bounds(coords, shape):
    """Check if the coordinates are within the bounds of the array."""
    return all(0 <= coords[dim] < shape[dim] for dim, val in enumerate(coords))

def get_neighbors_6n(coords):
    """Get the 6 face-adjacent neighbors in 3D."""
    x, y, z = coords
    return [(x-1, y, z), (x+1, y, z), 
            (x, y-1, z), (x, y+1, z), 
            (x, y, z-1), (x, y, z+1)]

def get_neighbors_26n(coords):
    """Get all 26 possible neighbors in 3D (including diagonals)."""
    neighbors = []
    for x in [-1, 0, 1]:
        for y in [-1, 0, 1]:
            neighbors.extend(
                (coords[0] + x, coords[1] + y, coords[2] + z)
                for z in [-1, 0, 1]
                if (x, y, z) != (0, 0, 0)
            )
    return neighbors

def dice_score(mask1, mask2):
    """Computes the dice score between two binary masks.

    Args:
    mask1: A 3D numpy array of binary values.
    mask2: A 3D numpy array of binary values.

    Returns:
    The dice score, a float value between 0 and 1.
    """
    assert np.any(np.unique(mask1) == np.unique(mask2))
    intersection = np.sum(mask1 & mask2)
    union = np.sum(mask1) + np.sum(mask2)
    return (2 * intersection) / (union)

def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def get_current_intensity(point_intensity, seed_avg_intensity, total_intensity, total_points, intensity_mode):
    if intensity_mode == "point":
        # return parent intensity
        return point_intensity
    elif intensity_mode == "average":
        # return average intensity of seed points
        return seed_avg_intensity
    elif intensity_mode == "running_average":
        # return average intensity of current grown region
        return (total_intensity) / (total_points)
    else:
        raise ValueError("Invalid intensity mode")
    

def region_growing_3d_seed(volume, seed_points, tolerance, size_threshold,
                            max_dist_voxels, connectivity=26, intensity_mode="point"):
    """
    Grow a region in a 3D volume based on intensity differences with multiple seed points.
    Allows selection between point intensity, average intensity, and running average intensity for region comparison.
    """
    region = np.zeros_like(volume, dtype=bool)  # The resulting binary mask
    queue = deque()  # Initialize the queue with the seed points
    stop_signal = False  # Signal to indicate whether to stop investigating further tolerances
    total_intensity = 0
    total_points = 0

    for seed_point in seed_points:
        if is_within_bounds(seed_point, volume.shape):
            region[seed_point] = True  # Mark the seed point as included in the region
            queue.append(seed_point)
            total_intensity += volume[seed_point]
            total_points += 1
    seed_avg_int = total_intensity / total_points
    seed_points_cm = np.mean(np.array(seed_points), axis=0)
    get_neighbors = get_neighbors_26n if connectivity == 26 else get_neighbors_6n

    while queue:
        point = queue.popleft()  # Get a point from the queue
        point_intensity = volume[point]

        for neighbor in get_neighbors(point):
            if is_within_bounds(neighbor, volume.shape) and not region[neighbor]:
                # Get reference intensity to compare with potential point intensity
                current_intensity = get_current_intensity(point_intensity, seed_avg_int, total_intensity, total_points, intensity_mode)
                # neighbor_dist = np.linalg.norm(seed_points_cm - neighbor)

                # if (abs(volume[neighbor] - current_intensity) <= tolerance) and (neighbor_dist < max_dist_voxels):
                if (abs(volume[neighbor] - current_intensity) <= tolerance):

                    region[neighbor] = True  # Include the neighbor in the region
                    queue.append(neighbor)  # Add the neighbor to the queue for further exploration
                    total_intensity += volume[neighbor]
                    total_points += 1

                    # If adding neighbors caused the region to grow too quickly, set the stop signal and finish
                    if np.sum(region) > size_threshold:
                        stop_signal = True
                        break

        if stop_signal:
            break

    return region, stop_signal


def region_growing_with_auto_tolerance(volume, seeds, size_threshold, max_dist_voxels,
                                        tolerance_range=(0, 100, 1), connectivity=26, 
                                        intensity_mode="point"):
    """ 
    Calculates results for several tolerance values and yields optimal based on 
    elbow-method
    """
    grown_regions = []
    len_list = []

    # Loop over the tolerance values and perform region growing
    # for tolerance in tqdm(np.arange(*tolerance_range), desc="Looping over tolerances"):
    for tolerance in np.arange(*tolerance_range):
        grown_region, exceeded = region_growing_3d_seed(volume, seeds, tolerance,
                                                        size_threshold, max_dist_voxels, 
                                                        connectivity,  intensity_mode)
        
        grown_regions.append(grown_region)
        len_list.append(np.sum(grown_region))

        # Do not continue if max size exceeded
        if exceeded:
            break  # Exit if the size threshold is exceeded

    # Determine the selected tolerance based on the sudden rise (exceeded signal)
    tolerances = np.arange(*tolerance_range)[:len(len_list)]
    knee_locator = KneeLocator(tolerances, len_list, curve='convex', direction='increasing', interp_method='interp1d')
    selected_tolerance = knee_locator.knee
    knee_index = np.where(tolerances == selected_tolerance)[0][0]

    # selected_tolerance_index = knee_index if exceeded_size_threshold else len(grown_regions) - 1
    selected_tolerance_index = knee_index
    # selected_tolerance = np.arange(*tolerance_range)[selected_tolerance_index]
    selected_mask = grown_regions[selected_tolerance_index]
    
    # -------------------- Cleaning of mask ----------------------------------
    
    # Define the structure for dilation and erosion based on connectivity
    struct = generate_binary_structure(volume.ndim, connectivity)
    
    # Perform one final dilation and then erosion (closing)
    closed_mask = binary_dilation(selected_mask, structure=struct, iterations=2)
    closed_mask = binary_erosion(closed_mask, structure=struct, iterations=2)
    
    # Fill holes in the mask to ensure it's solid
    closed_mask = binary_fill_holes(closed_mask, structure=struct)

    # Label the connected components
    labeled_mask, num_features = nd_label(closed_mask, structure=struct)

    # If there are multiple features, select the largest one
    if num_features > 1:
        print(f"More than one CC found, a total of {num_features}")
        max_label = 1 + np.argmax([np.sum(labeled_mask == i) for i in range(1, num_features + 1)])
        cleaned_mask = (labeled_mask == max_label)
    else:
        cleaned_mask = closed_mask

    # Metadata
    metadata = {
        'n_pixels': np.sum(cleaned_mask),
        'tolerance_selected': selected_tolerance,
        'tolerance_pixel_counts': len_list,
        'tolerances_inspected': len(len_list), 
        'elbow_i': knee_index 
    }

    return cleaned_mask, metadata



##############################################################################
###################            CMB processing              ###################
##############################################################################

def calculate_size_threshold(image: nib.Nifti1Image) -> int:
    """
    Calculate the maximum size limit for a mask based on the voxel size of the image.

    Args:
        image (nibabel.Nifti1Image): The MRI image.

    Returns:
        int: The calculated size threshold in number of voxels.
    """
    # Radius of the sphere in mm
    radius_mm = 5

    # Get voxel dimensions from the image header
    voxel_dims = image.header.get_zooms()
    voxel_volume_mm3 = np.prod(voxel_dims)

    # Volume of the sphere in mm^3
    sphere_volume_mm3 = (4/3) * np.pi * (radius_mm ** 3)

    # Calculate the number of voxels that fit into the sphere
    size_threshold = int(sphere_volume_mm3 / voxel_volume_mm3)

    return size_threshold


def isolate_single_CMBs(mask: np.ndarray, voxel_size: list, max_dist_mm: float = 10.0) -> list:
    """
    Isolate regions in the given binary mask into separate masks based on a distance threshold.
    Each mask corresponds to a single CMB entity.

    Args:
        mask (numpy.ndarray): Binary mask of the image.
        voxel_size (list): List of voxel dimensions in mm [x, y, z].
        max_dist_mm (float): Maximum distance in mm between any two points in a cluster.

    Returns:
        List[np.ndarray]: A list of binary masks, each for an individual CMB.
    """
    # Convert to voxel coordinates
    indices = np.argwhere(mask)

    # Special case: if there's only one voxel, return it directly
    if len(indices) == 1:
        cmb_mask = np.zeros_like(mask)
        cmb_mask[tuple(indices[0])] = 1
        return [cmb_mask]

    # Convert distance threshold from mm to voxels
    max_dist_voxels = max_dist_mm / np.mean(voxel_size)  # Average voxel size for simplicity

    # Calculate pairwise distances between points
    Y = pdist(indices * voxel_size, metric='euclidean')

    # Perform hierarchical/agglomerative clustering
    Z = linkage(Y, method='single')  # 'single' linkage ensures max distance in a cluster

    # Form flat clusters
    labels = fcluster(Z, max_dist_voxels, criterion='distance')

    # Creating individual masks for each cluster
    cmb_masks = []
    for label in np.unique(labels):
        cmb_mask = np.zeros_like(mask)
        cmb_mask[tuple(indices[labels == label].T)] = 1
        cmb_masks.append(cmb_mask)

    return cmb_masks


def process_cmb_mask(label_im, msg, log_level="\t\t"):
    """
    Process a nibabel object containing a mask of cerebral microbleeds (CMBs).

    Args:
        label_im (nibabel.Nifti1Image): The nibabel object of the mask.
        msg (str): Log message to be updated.

    Returns:
        processed_mask_nib (nibabel.Nifti1Image): Processed mask as a nibabel object.
        com_list (list[tuple]): List of centers of mass for each connected component.
        pixel_counts (list[int]): List of pixel counts for each connected component.
        radii (list[float]): List of equivalent radii for each connected component.
        msg (str): Updated log message.
    """

    # Extract data from the label image
    data = label_im.get_fdata()

    # Identify and handle unique labels in the mask
    unique_labels, counts = np.unique(data, return_counts=True)
    if len(unique_labels) > 2:
        raise ValueError("More than two unique labels found in the mask.")
    elif len(unique_labels) == 1:
        msg += f"{log_level}Only one label found: {unique_labels[0]}\n"
        data[:] = 0
    else:
        majority_label, minority_label = unique_labels[np.argmax(counts)], unique_labels[np.argmin(counts)]
        data[data == majority_label], data[data == minority_label] = 0, 1

    # Find connected components in the mask
    labeled_array, num_features = nd_label(data)

    # Calculate centers of mass and pixel counts
    com_list = center_of_mass(data, labels=labeled_array, index=np.arange(1, num_features + 1))
    com_list = [(int(coord[0]), int(coord[1]), int(coord[2])) for coord in com_list]
    pixel_counts = np.bincount(labeled_array.ravel())[1:]

    # Calculate radii assuming CMBs are spherical
    radii = [(3 * count / (4 * np.pi))**(1/3) for count in pixel_counts]
    radii = [round(rad, 2) for rad in radii]

    # Convert the processed mask data back to a nibabel object
    processed_mask_nib = nib.Nifti1Image(data, label_im.affine, label_im.header)

    # Update the log message
    msg += f'{log_level}Number of CMBs: {len(com_list)}. Sizes: {pixel_counts},' + \
            f' Radii: {radii}, Unique labels: {unique_labels}, Counts: {counts}\n'

    metadata = {
        'centers_of_mass': com_list,
        'pixel_counts': pixel_counts,
        'radii': radii
    }

    return processed_mask_nib, metadata, msg