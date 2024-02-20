#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Functions to process CMB masks

@author: jorgedelpozolerida
@date: 16/01/2024
"""

import os
import sys
import argparse
import traceback


import logging
import numpy as np

import nibabel as nib
from tqdm import tqdm
from collections import deque
from scipy.ndimage import binary_dilation, binary_erosion, binary_closing, generate_binary_structure, \
    center_of_mass, binary_fill_holes
from skimage.morphology import ball
from scipy.ndimage import label as nd_label # to avoid conflict below
from skimage.measure import label
from skimage.filters import threshold_otsu
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from kneed import KneeLocator

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)




##############################################################################
###################                General                 ###################
##############################################################################

def get_largest_cc(segmentation):
    """
    Gets the largest connected component in image.
    Args:
        segmentation (np.ndarray): Image with blobs.
    Returns:
        largest_cc (np.ndarray): A binary image containing nothing but the largest
                                    connected component.
    """
    labels = label(segmentation)
    bincount = np.array(np.bincount(labels.flat))
    ind_large = np.argmax(bincount)  # Background is initially largest
    bincount[ind_large] = 0  # Remove background
    ind_large = np.argmax(bincount)  # This should now be largest connected component
    largest_cc = labels == ind_large

    return np.double(largest_cc)


def get_brain_mask(image):
    """
    Computes brain mask using Otsu's thresholding and morphological operations.
    Args:
        image (nib.Nifti1Image): Primary sequence image.
    Returns:
        mask (np.ndarray): Computed brain mask.
    """
    # TODO: investigate if this a good fit for brain mask in all cases. Play around
    image_data = image.get_fdata()
    
    # Otsu's thresholding
    threshold = threshold_otsu(image_data)
    mask = image_data > threshold

    # Apply morphological operations
    struct = generate_binary_structure(3, 2)  # this defines the connectivity
    mask = binary_closing(mask, structure=struct)
    mask = get_largest_cc(mask)
    mask = binary_dilation(mask, iterations=5, structure=struct)

    return mask



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
                            max_dist_voxels, connectivity=26, 
                            intensity_mode="point", difference_mode = "relative",
                            epsilon=1e-5
                            ):
    """
    Grow a region in a 3D volume based on intensity differences with multiple seed points.
    Allows selection between point intensity, average intensity, and running average intensity for region comparison.
    """
    region = np.zeros_like(volume, dtype=bool)  # The resulting binary mask
    queue = deque()  # Initialize the queue with the seed points
    visited = set()  # Track visited points
    stop_signal = False  # Signal to indicate whether to stop investigating further tolerances
    total_intensity = 0
    total_points = 0

    for seed_point in seed_points:
        if is_within_bounds(seed_point, volume.shape):
            region[seed_point] = True  # Mark the seed point as included in the region
            queue.append(seed_point)
            visited.add(seed_point)
            total_intensity += volume[seed_point]
            total_points += 1
    seed_avg_int = total_intensity / total_points
    seed_points_cm = np.mean(np.array(seed_points), axis=0)
    get_neighbors = get_neighbors_26n if connectivity == 26 else get_neighbors_6n

    while queue:
        point = queue.popleft()  # Get a point from the queue
        point_intensity = volume[point]

        for neighbor in get_neighbors(point):
            if is_within_bounds(neighbor, volume.shape) and neighbor not in visited:
                visited.add(neighbor)  # Mark the neighbor as visited
                # Get reference intensity to compare with potential point intensity
                current_intensity = get_current_intensity(point_intensity, seed_avg_int, total_intensity, total_points, intensity_mode)
                if max_dist_voxels is not None:
                    neighbor_dist = np.linalg.norm(seed_points_cm - neighbor)
                else:
                    max_dist_voxels = 50 # large number
                    neighbor_dist = 0 # large number

                if difference_mode == "relative":
                    if current_intensity == 0:
                        current_intensity = epsilon
                    intensity_diff = abs((volume[neighbor] - current_intensity) / current_intensity )
                else: 
                    intensity_diff = abs((volume[neighbor] - current_intensity))
                if (intensity_diff <= tolerance) and (neighbor_dist < max_dist_voxels):
                # if (intensity_diff <= tolerance):
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
                                        tolerance_values, connectivity=26, 
                                        intensity_mode="point", show_progress=False, 
                                        diff_mode="normal",
                                        log_level="\t\t\t", msg=""):
    """ 
    Calculates results for several tolerance values and yields optimal based on 
    elbow-method
    """
    grown_regions = []
    len_list = []

    # Loop over the tolerance values and perform region growing
    iterator = tolerance_values
    if show_progress:
        iterator = tqdm(tolerance_values, desc="Looping over tolerances")

    for tolerance in iterator:
        grown_region, exceeded = region_growing_3d_seed(volume, seeds, tolerance,
                                                        size_threshold, max_dist_voxels, 
                                                        connectivity,  intensity_mode,
                                                        diff_mode)
        grown_regions.append(grown_region)
        len_list.append(np.sum(grown_region))

        # Do not continue if max size exceeded
        if exceeded:
            break  # Exit if the size threshold is exceeded

    # Determine the selected tolerance based on the sudden rise (exceeded signal)
    tolerances = tolerance_values[:len(len_list)]
    knee_locator = KneeLocator(tolerances, len_list, curve='convex', direction='increasing', interp_method='interp1d')

    if knee_locator.knee is None:
        msg += f"{log_level}Knee could not be found, selecting second to last tolerance"
        selected_tolerance = tolerances[-2]  # or any default tolerance
    else:
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
    closed_mask = binary_dilation(selected_mask, structure=struct, iterations=4)
    closed_mask = binary_erosion(closed_mask, structure=struct, iterations=4)
    
    # Fill holes in the mask to ensure it's solid
    closed_mask = binary_fill_holes(closed_mask, structure=struct)

    # Label the connected components
    labeled_mask, num_features = nd_label(closed_mask, structure=struct)

    # If there are multiple features, select the largest one
    if num_features > 1:
        counts = np.bincount(labeled_mask.ravel())
        msg += f"{log_level}Found {num_features} CCs in one CMB. Counts: {counts[1:]}"
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
        'elbow_i': knee_index,
        'elbow2end_tol': len_list[knee_index-1:]
    }

    return cleaned_mask, metadata, msg


# def optimize_region_growing(volume, seeds, size_threshold, max_dist_voxels=None, 
#                             connectivity_options=[6, 26], 
#                             intensity_mode_options=['point', 'running_average'], 
#                             diff_mode_ranges=['relative', 'normal'],
#                             log_level=""):  # sourcery skip: default-mutable-arg
#     """
#     Optimize the region growing process by iterating over combinations of parameters provided by the user.

#     Parameters:
#     - volume: The 3D volume data as a NumPy array.
#     - seeds: Seed points for region growing.
#     - size_threshold: The minimum size threshold for consideration.
#     - connectivity_options: A list of connectivity values to iterate over.
#     - intensity_mode_options: A list of intensity modes to iterate over.
#     - diff_mode_ranges: A dictionary with diff modes as keys and their corresponding tolerance ranges as values.
#     - max_dist_voxels: The maximum distance in voxels to be used, can be None.
#     - log_level: String to indicate logging level for messages.

#     Returns:
#     - best_processed_mask: The best processed mask as a NumPy array.
#     - best_metadata: Metadata dictionary for the best mask.
#     - optimization_msg: A message summarizing the optimization results.
#     """
#     best_n_pixels = 1
#     best_processed_mask = None
#     best_metadata = None
#     best_msg = ""

#     # Iterate over combinations of provided parameters
#     for connectivity in connectivity_options:
#         for intensity_mode in intensity_mode_options:
#             for diff_mode, range_temp in diff_mode_ranges.items():
#                 msg_temp = ""
#                 processed_mask, metadata, msg_temp = region_growing_with_auto_tolerance(
#                     volume=volume,
#                     seeds=seeds,
#                     size_threshold=size_threshold,
#                     max_dist_voxels=max_dist_voxels,
#                     tolerance_values=range_temp,
#                     connectivity=connectivity,
#                     show_progress=False,
#                     intensity_mode=intensity_mode,
#                     diff_mode=diff_mode,
#                     log_level=f"{log_level}\t\t",
#                     msg=msg_temp
#                 )
#                 n_pixels = metadata['n_pixels']
#                 # Update best results if this combination yielded bigger size
#                 if n_pixels > best_n_pixels:
#                     best_n_pixels = n_pixels
#                     best_processed_mask = processed_mask
#                     best_metadata = metadata
#                     best_msg = msg_temp
#                     best_intensity_mode = intensity_mode
#                     best_diff_mode = diff_mode

#     # Construct a final message summarizing the optimization result
#     optimization_msg = best_msg
#     optimization_msg += f"{log_level}Optimization results: connectivity={connectivity}, " \
#                             f"intensity_mode={best_intensity_mode}, diff_mode={best_diff_mode}, " \
#                             f"size={best_n_pixels}, max_dist_voxels={max_dist_voxels}.\n"

#     return best_processed_mask, best_metadata, optimization_msg


##############################################################################
###################            CMB processing              ###################
##############################################################################

def calculate_size_and_distance_thresholds(mri_im: nib.Nifti1Image, max_dist_mm: int = 10) -> tuple:
    """
    Calculate the maximum size limit and maximum distance allowed for an MRI image based on the voxel size.

    Parameters:
    - mri_im (nibabel.Nifti1Image): The MRI image.
    - max_dist_mm (int, optional): The maximum distance in millimeters allowed between any two points. Defaults to 10 mm.

    Returns:
    - tuple: A tuple containing the calculated size threshold in number of voxels and the maximum distance allowed in voxels.
    """
    # Get voxel dimensions from the image header
    voxel_dims = mri_im.header.get_zooms()
    voxel_volume_mm3 = np.prod(voxel_dims)

    # Volume of the sphere in mm^3 based on the radius (half of max_dist_mm, which is supposed to be diameter)
    sphere_volume_mm3 = (4/3) * np.pi * ((max_dist_mm / 2) ** 3)

    # Calculate the number of voxels that fit into the sphere (size threshold)
    size_threshold = int(sphere_volume_mm3 / voxel_volume_mm3)

    # Calculate the maximum distance allowed in voxels
    voxel_size = np.amin(voxel_dims) # we allow for maximum distance possible (so minimum size)
    max_dist_voxels = int(max_dist_mm / voxel_size)


    return size_threshold, max_dist_voxels

def calculate_size_threshold(image: nib.Nifti1Image, radius_mm=5) -> int:
    """
    Calculate the maximum size limit for a CMB mask based on the voxel size of the image.

    Args:
        image (nibabel.Nifti1Image): The MRI image.

    Returns:
        int: The calculated size threshold in number of voxels.
    """
    # Get voxel dimensions from the image header
    voxel_dims = image.header.get_zooms()
    voxel_volume_mm3 = np.prod(voxel_dims)

    # Volume of the sphere in mm^3
    sphere_volume_mm3 = (4/3) * np.pi * (radius_mm ** 3)

    # Calculate the number of voxels that fit into the sphere
    size_threshold = int(sphere_volume_mm3 / voxel_volume_mm3)

    return size_threshold

def calculate_max_size_and_distance(mri_im, log_level, max_dist_mm = 10):
    """
    Calculate and return the maximum size threshold and maximum distance allowed for an MRI image.

    Parameters:
    - mri_im: The MRI nibabel image object
    - log_level: A string indicating the number of tabs for logging
    - max_dist_mm: mm of max distance allowed between any two points (max can be diameter)
    Returns:
    - size_th: The calculated size threshold.
    - max_dist_voxels: The maximum distance allowed, measured in voxels.
    """
    size_th = calculate_size_threshold(mri_im, radius_mm=max_dist_mm/2)
    voxel_size = np.mean(mri_im.header.get_zooms())
    max_dist_voxels = int(max_dist_mm / voxel_size)

    return size_th, max_dist_voxels

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
    
    Extracts connected components and performs morhpological operaitons to clean
    mask. Then returns list of centers of mass, radius and size for each CMB.

    Args:
        label_im (nibabel.Nifti1Image): The nibabel object of the mask.
        msg (str): Log message to be updated.

    Returns:
        processed_mask_nib (nibabel.Nifti1Image): Processed mask as a nibabel object.
        metadata (dict): Metadata including centers of mass, pixel counts, and radii.
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

    # Perform dilation and erosion to clean the mask
    struct_elem = ball(2)  # Using a spherical structuring element
    data = binary_dilation(data, structure=struct_elem).astype(np.uint8)
    data = binary_erosion(data, structure=struct_elem).astype(np.uint8)

    # Fill holes in the mask
    data = binary_fill_holes(data).astype(np.uint8)

    # Find connected components in the cleaned mask
    labeled_array, num_features = nd_label(data)

    # Calculate centers of mass, pixel counts, and radii
    com_list = center_of_mass(data, labels=labeled_array, index=np.arange(1, num_features + 1))
    com_list = [(int(coord[0]), int(coord[1]), int(coord[2])) for coord in com_list]
    pixel_counts = np.bincount(labeled_array.ravel())[1:]
    radii = [(3 * count / (4 * np.pi))**(1/3) for count in pixel_counts]
    radii = [round(r, ndigits=2) for r in radii]
    # Convert the processed mask data back to a nibabel object
    processed_mask_nib = nib.Nifti1Image(data, label_im.affine, label_im.header)

    # Update the log message
    msg += f"{log_level}Number of CMBs: {num_features}. Sizes: {pixel_counts}, Radii: {radii}, Unique labels: {unique_labels}, Counts: {counts}\n"

    # Generate metadata
    metadata = {
        i: {"CM": com, "size": pixel_counts[i], "radius": round(radii[i], 2)}
        for i, com in enumerate(com_list)
    }

    return processed_mask_nib, metadata, msg
