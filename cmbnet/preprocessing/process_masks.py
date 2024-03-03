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
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from kneed import KneeLocator
from typing import List, Tuple
import warnings
import json
warnings.filterwarnings("ignore", category=RuntimeWarning, module='kneed')


import cmbnet.preprocessing.loading as loading

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


# def region_growing_with_auto_tolerance(volume, seeds, size_threshold, max_dist_voxels,
#                                         tolerance_values, connectivity=26, 
#                                         intensity_mode="point", show_progress=False, 
#                                         diff_mode="normal",
#                                         log_level="\t\t\t", msg=""):
#     """ 
#     Calculates results for several tolerance values and yields optimal based on 
#     elbow-method
#     """
#     grown_regions = []
#     len_list = []

#     # Loop over the tolerance values and perform region growing
#     iterator = tolerance_values
#     if show_progress:
#         iterator = tqdm(tolerance_values, desc="Looping over tolerances")

#     for tolerance in iterator:
#         grown_region, exceeded = region_growing_3d_seed(volume, seeds, tolerance,
#                                                         size_threshold, max_dist_voxels, 
#                                                         connectivity,  intensity_mode,
#                                                         diff_mode)
#         grown_regions.append(grown_region)
#         len_list.append(np.sum(grown_region))

#         # Do not continue if max size exceeded
#         if exceeded:
#             break  # Exit if the size threshold is exceeded

#     # Determine the selected tolerance based on the sudden rise (exceeded signal)
#     tolerances = tolerance_values[:len(len_list)]
#     knee_locator = KneeLocator(tolerances, len_list, curve='convex', direction='increasing', interp_method='interp1d')

#     if knee_locator.knee is None:
#         msg += f"{log_level}Knee could not be found, selecting second to last tolerance\n"
#         selected_tolerance = tolerances[-2]  # or any default tolerance
#     else:
#         selected_tolerance = knee_locator.knee
#     knee_index = np.where(tolerances == selected_tolerance)[0][0]

#     selected_tolerance_index = knee_index
#     selected_mask = grown_regions[selected_tolerance_index]
    
#     # -------------------- Cleaning of mask ----------------------------------
    
#     # Define the structure for dilation and erosion based on connectivity
#     connectivity_struct = 1
#     struct = generate_binary_structure(volume.ndim, connectivity_struct)
    
#     # Initialize erosion counter
#     erosion_iterations = 0
#     eroded_mask = np.copy(selected_mask)  # Start with the selected mask

#     # Ensure there's more than 1 voxel to erode
#     if np.sum(selected_mask) > 1:  
#         for _ in range(4):  # Attempt up to 2 iterations of erosion
#             # Perform an erosion iteration
#             temp_eroded_mask = binary_erosion(eroded_mask, structure=struct)
            
#             # Label the eroded mask to identify connected components
#             labeled_mask, num_features = nd_label(temp_eroded_mask)
            
#             # Calculate the size of each connected component
#             component_sizes = np.bincount(labeled_mask.ravel())[1:]  # Exclude background
            
#             # Check if any component is reduced to a size of 1 voxel
#             if np.any(component_sizes == 1):
#                 break  # Stop erosion if any component is reduced to 1 voxel
#             elif np.sum(temp_eroded_mask) >= 1:
#                 eroded_mask = temp_eroded_mask
#                 erosion_iterations += 1  # Increment erosion counter
#             else:
#                 break  # Stop erosion if it would result in no voxels


#     # Label the connected components
#     labeled_mask, num_features = nd_label(eroded_mask, structure=struct)

#     # Calculate mean position of seeds
#     seeds_mean_position = np.mean(seeds, axis=0)

#     # If there are multiple features, select the one closest to the mean of seeds positions
#     if num_features > 1:
#         # Calculate center of mass for each feature
#         centers_of_mass = center_of_mass(eroded_mask, labeled_mask, range(1, num_features + 1))

#         # Calculate distances from each center of mass to the mean of seeds positions
#         distances = [np.linalg.norm(np.array(com) - seeds_mean_position) for com in centers_of_mass]

#         # Select the label with the minimum distance
#         closest_label = np.argmin(distances) + 1  # +1 because labels start from 1
#         cleaned_mask = (labeled_mask == closest_label)
#         msg += f"{log_level}Selected CC closest to seeds mean position. Num of CC: {len(centers_of_mass)}\n"
#     else:
#         cleaned_mask = eroded_mask

#     # Now, apply dilation to the cleaned_mask with the same number of iterations as erosion
#     closed_mask = binary_dilation(cleaned_mask, structure=struct, iterations=erosion_iterations)

#     # Metadata
#     metadata = {
#         'n_pixels': np.sum(closed_mask),
#         'tolerance_selected': selected_tolerance,
#         'tolerance_pixel_counts': len_list, 
#         'tolerances_inspected': len(len_list), # total number of tolerances visited
#         'elbow_i': knee_index, # index taken
#         'elbow2end_tol': len_list[knee_index-1:] # sizes of last elements
#     }
#     return cleaned_mask, metadata, msg

def calculate_sphericity(mask):
    """
    Calculates the sphericity of a 3D mask.

    Args:
        mask (numpy.ndarray): A 3D numpy array where the mask of the object is True (1) and the background is False (0).

    Returns:
        float: The sphericity index of the mask. Values closer to 1 indicate a more spherical shape.
    """
    # Volume calculation (V)
    volume = np.sum(mask)

    # Generate a structuring element for surface area calculation
    struct = generate_binary_structure(3, 1)

    # Erode mask to identify surface voxels
    eroded_mask = binary_erosion(mask, structure=struct)
    
    # Identify surface voxels using logical_xor
    surface_voxels = np.logical_xor(mask, eroded_mask)

    # Surface area calculation (A)
    surface_area = np.sum(surface_voxels)

    # Sphericity calculation (Ψ)
    sphericity = (np.pi ** (1/3) * (6 * volume) ** (2/3)) / surface_area

    return sphericity

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

    selected_tolerance_index = knee_index
    selected_mask = grown_regions[selected_tolerance_index]
    
    # -------------------- Cleaning of mask ----------------------------------
    
    # Define the structure for dilation and erosion based on connectivity
    struct = generate_binary_structure(volume.ndim, 1)
    
    # # Perform one final dilation and then erosion (closing)
    # closed_mask = binary_dilation(selected_mask, structure=struct, iterations=4)
    # closed_mask = binary_erosion(closed_mask, structure=struct, iterations=4)
    
    # # Fill holes in the mask to ensure it's solid
    # closed_mask = binary_fill_holes(closed_mask, structure=struct)

    # Label the connected components
    labeled_mask, num_features = nd_label(selected_mask, structure=struct)

    # If there are multiple features, select the largest one
    if num_features > 1:
        counts = np.bincount(labeled_mask.ravel())
        msg += f"{log_level}Found {num_features} CCs in one CMB. Counts: {counts[1:]}\n"
        max_label = 1 + np.argmax([np.sum(labeled_mask == i) for i in range(1, num_features + 1)])
        cleaned_mask = (labeled_mask == max_label)
    else:
        cleaned_mask = selected_mask

    # Sphericity
    sphere_index = round(calculate_sphericity(cleaned_mask), ndigits=3)

    # Metadata
    metadata = {
        'n_pixels': np.sum(cleaned_mask),
        'tolerance_selected': selected_tolerance,
        'tolerance_pixel_counts': len_list,
        'tolerances_inspected': len(len_list), 
        'elbow_i': knee_index,
        'elbow2end_tol': len_list[knee_index-1:],
        'sphericity_ind': sphere_index
    }

    return cleaned_mask, metadata, msg

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


def process_cmb_mask(label_im, msg, dataset_name="valdo", log_level="\t\t"):
    """
    Process a nibabel object containing a mask of cerebral microbleeds (CMBs).
    
    Extracts connected components and then returns list of centers of mass, radius and size for each CMB.

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

    # Fill holes in the mask
    data_filled = binary_fill_holes(data).astype(np.uint8)

    # Find connected components in the cleaned mask
    labeled_array, num_features = nd_label(data_filled)

    # Initialize an empty mask for combining processed CMBs
    processed_data = np.zeros_like(data_filled)

    # Perform erosion and dilation on each CMB separately execpt for the "valdo" dataset
    if dataset_name not in  ["valdo", "rodeja"]:
        struct_elem = generate_binary_structure(3, 1)
        for label_num in range(1, num_features + 1):
            cmb_mask = (labeled_array == label_num)  # Isolate the current CMB
            
            # Apply dilation and erosion
            cmb_mask_dilated = binary_dilation(cmb_mask, structure=struct_elem, iterations=2).astype(np.uint8)
            cmb_mask_processed = binary_erosion(cmb_mask_dilated, structure=struct_elem, iterations=2).astype(np.uint8)
            
            # Label the processed CMB to find connected components
            labeled_cmb, num_features_cmb = nd_label(cmb_mask_processed)
            
            if num_features_cmb > 1:
                # If more than one CC is found, keep only the largest one
                sizes = np.bincount(labeled_cmb.ravel())[1:]  # Exclude background size
                largest_cc_label = np.argmax(sizes) + 1  # +1 because np.argmax starts at 0
                cmb_mask_processed = (labeled_cmb == largest_cc_label).astype(np.uint8)
            
            # Combine the processed (and potentially filtered) CMB mask with the overall processed mask
            processed_data += cmb_mask_processed

        msg += f"{log_level}Applied closing operation to every CMB and kept only the largest CC if more than one was created.\n"

    else:
        # For the "valdo" dataset, use the filled data without further processing
        processed_data = data_filled

    # Recalculate centers of mass, pixel counts, and radii after processing ----
    labeled_array_processed, num_features_processed = nd_label(processed_data)

    # Calculate centers of mass for each labeled feature
    com_list = center_of_mass(processed_data, labels=labeled_array_processed, index=np.arange(1, num_features_processed + 1))
    com_list = [(int(coord[0]), int(coord[1]), int(coord[2])) for coord in com_list]

    # Count pixels in each labeled region
    pixel_counts = np.bincount(labeled_array_processed.ravel())[1:]  # Skip the background count

    # Calculate radii for each labeled feature
    radii = [(3 * count / (4 * np.pi))**(1/3) for count in pixel_counts]
    radii = [round(r, ndigits=2) for r in radii]

    # Identify indices where pixel_count is not 1, preserving only those entries
    valid_indices = [i for i, count in enumerate(pixel_counts) if count != 1]

    # Filter com_list and radii using valid_indices
    com_list_filtered = [com_list[i] for i in valid_indices]
    radii_filtered = [radii[i] for i in valid_indices]

    # Update com_list and radii to the filtered versions
    com_list = com_list_filtered
    radii = radii_filtered

    # Convert the processed mask data back to a nibabel object
    processed_mask_nib = nib.Nifti1Image(processed_data, label_im.affine, label_im.header)

    # Update the log message
    msg += f"{log_level}Number of processed CMBs: {num_features_processed}, Unique labels: {unique_labels}, Counts: {counts}\n"

    # Generate metadata
    metadata = {
        i: {"CM": com, "size": pixel_counts[i], "radius": radii[i]}
        for i, com in enumerate(com_list)
    }

    return processed_mask_nib, metadata, msg


def grow_3D_sphere(example_im, seeds, rad, size_threshold, max_dist_voxels, log_level, msg):
    """
    Grows a 3D spherical mask from a seed point within a NIfTI image volume, 
    based on a given radius in millimeters.

    Args:
        example_im (nib.Nifti1Image): The NIfTI image for extracting voxel dimensions.
        seeds (list): The seed points from which the sphere grows.
        rad (float): The radius of the sphere in millimeters.
        size_threshold (int): The maximum allowable size of the mask in voxels.
        max_dist_voxels (int): The maximum distance in voxels for adding new voxels to the mask.
        log_level (str): Logging level for messages.
        msg (str): Initial message string.

    Returns:
        processed_mask (numpy.ndarray): The final processed mask as a 3D numpy array.
        metadata (dict): Metadata about the generated mask.
        msg (str): Updated message string.
    """
    # Extract voxel dimensions from the nibabel object to convert radius from mm to voxels
    voxel_dims = example_im.header.get_zooms()
    
    # Calculate the average seed position in voxel coordinates
    seeds_mean = np.mean(seeds, axis=0)
    
    # Initialize an empty mask with the same shape as the example image
    mask_shape = example_im.shape
    processed_mask = np.zeros(mask_shape, dtype=np.uint8)
    
    # Convert the radius from millimeters to voxels for each dimension
    rad_voxels = rad / np.array(voxel_dims)
    
    # Generate spherical mask
    for i in range(mask_shape[0]):
        for j in range(mask_shape[1]):
            for k in range(mask_shape[2]):
                # Calculate distance from current voxel to seed mean in voxel units
                dist_voxels = np.sqrt(((i - seeds_mean[0]) / rad_voxels[0]) ** 2 +
                                      ((j - seeds_mean[1]) / rad_voxels[1]) ** 2 +
                                      ((k - seeds_mean[2]) / rad_voxels[2]) ** 2)
                if dist_voxels <= 1.0 and dist_voxels <= max_dist_voxels:
                    processed_mask[i, j, k] = 1

    # Decompose into connected components and keep the largest
    labeled_mask, num_features = nd_label(processed_mask)
    if num_features > 1:
        component_sizes = np.bincount(labeled_mask.ravel())[1:]  # Ignore background
        largest_component = component_sizes.argmax() + 1  # Label of largest component
        processed_mask = (labeled_mask == largest_component).astype(bool)
    
    # Fill holes in the largest component
    processed_mask = binary_fill_holes(processed_mask).astype(bool)
    
    # Check size threshold
    if np.sum(processed_mask) > size_threshold:
        msg += f"{log_level}Generated mask exceeds size threshold.\n"
        return processed_mask, {}, msg
    
    # Metadata
    metadata = {
        'rad_voxels': rad_voxels.tolist(),  # Convert numpy array to list for JSON-serializable metadata
        'n_pixels': np.sum(processed_mask),
    }
    
    return processed_mask, metadata, msg

def reprocess_study(study, processed_dir, mapping_file, dataset,
                    mri_im: nib.Nifti1Image, 
                    com_list: list, msg: str,
                    log_level="\t\t"):
    """
    Re-Process annotations for dataset subject by creating a sphere.
    Radius is derived from:
    - Manually set radius in "manual_fixes.csv" file 
    OR
    - Using Region Growing estimated radius from first run
    """
    # Get study metadata
    json_file = os.path.join(processed_dir, "Data", study, "Annotations_metadata", f"{study}_metadata.json")
    with open(json_file, 'r') as file:
        metadata_dict = json.load(file)

    # Get mapping list
    map_df = loading.get_sphere_df(mapping_file, study, dataset)
    if map_df.shape[0] > 0:
        msg += f"{log_level}---- Study found in manual fixes mapping CSV ----\n"

    # Compute size threshold and maximum distance in voxels
    size_th, max_dist_voxels = calculate_size_and_distance_thresholds(mri_im, max_dist_mm=10)

    # Initialize the final processed mask
    final_processed_mask = np.zeros_like(mri_im.get_fdata(), dtype=bool)
    rg_metadata = {}  # To collect metadata from region growing
    msg += f"{log_level}Applying Region Growing with max_distance={max_dist_voxels}, max_size={size_th}\n\n"

    # Process each CMB based on its center of mass
    for i, com in enumerate(com_list):
        msg += f"{log_level}\tCMB-{i}\n"
        seeds = [com]

        # Get from previous execution
        metadata_i = metadata_dict['CMBs_old'][str(i)]
        metadata_rg_i = metadata_i['region_growing']
        com_i = tuple(int(i) for i in metadata_i["CM"])
        assert com_i == com  # both are tuples
        map_temp = map_df[(map_df['x'] == com_i[0]) & (map_df['y'] == com_i[1]) & (map_df['z'] == com_i[2])]

        if map_temp.shape[0] == 1:
            radius = map_temp['radius'].item() / 2
            msg += f"{log_level}\t\tWill use pre-set radius of {radius}\n"

        elif map_temp.shape[0] > 1:
            raise ValueError(f"Found several mapping rows for study: {study}")
        else:
            radius =  metadata_i['radius']
            if dataset == "dou":
                radius = radius / 2
            msg += f"{log_level}\t\tWill use RG radius of {radius}\n"
            
        if dataset == "momeni-synth" and radius > 3:
            radius = 1
            msg += f"{log_level}\t\tHarcoded to radius of {radius}\n"

        processed_mask, metadata, msg = grow_3D_sphere(
            example_im=mri_im,  # nibabel object
            seeds=seeds,
            rad=radius,
            size_threshold=size_th,
            max_dist_voxels=max_dist_voxels,
            log_level=f"{log_level}\t\t",
            msg=msg
        )

        # Expand if only 1-4 voxels to increase signal
        if 1 <= metadata['n_pixels'] <= 4:
            struct = generate_binary_structure(3, 1)  # 3D dilation, connectivity=1
            dilated_mask = binary_dilation(processed_mask, structure=struct, iterations=1)
            n_pixels_dilated = np.sum(dilated_mask)
            
            if n_pixels_dilated > size_th:
                msg += f"{log_level}Dilated mask would exceed size threshold. Nothing done\n"
            else:
                processed_mask = dilated_mask
                msg += f"{log_level}\t\tMask expanded by one layer, new size={n_pixels_dilated} voxels.\n"
        radius = (3 * int(metadata['n_pixels']) / (4 * np.pi))**(1/3)
        met_sph = metadata
        msg += f"{log_level}\t\tSphere created with radius {radius}mm, size={np.sum(processed_mask)}\n"

        if np.any(final_processed_mask & processed_mask):
            msg += f"{log_level}\t\tCAUTION: Overlap detected at {com}\n" + \
                    f"{log_level}\t\t         Previously visited CMBs: {com_list[:i]}\n"

        final_processed_mask |= processed_mask.astype(bool)
        rg_metadata[i] = {
            "CM": com,
            "size": np.sum(processed_mask),
            "radius": round(radius, ndigits=2),
            "region_growing": metadata_rg_i,
            "sphere": met_sph
        }

    metadata_out = {
        "healthy": "no" if com_list else "yes",
        "CMBs_old": rg_metadata,
    }
    annotation_processed_nib = nib.Nifti1Image(final_processed_mask.astype(np.int16), mri_im.affine, mri_im.header)

    return annotation_processed_nib, metadata_out, msg

