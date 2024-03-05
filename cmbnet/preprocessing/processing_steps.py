#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Script containing preprocessing steps functions


{Long Description of Script}


@author: jorgedelpozolerida
@date: 05/03/2024
"""


import os

import numpy as np                                                                  
import time 
from nilearn.image import resample_to_img, resample_img
import threading
import subprocess
import nibabel as nib

# Define a lock for thread synchronization
csv_lock = threading.Lock()

# Utils
import cmbnet.preprocessing.process_masks as process_masks
import cmbnet.utils.utils_general as utils_gen



##############################################################################
###################            PROCESSING STEPS            ###################
##############################################################################

def fix_orientation_and_dtype(sequences_raw, labels_raw, prim_seq, msg, log_level='\t'):
    """
    Validates and fixes the orientation and data type of MRI scans and annotations.

    Args:
        sequences_raw (dict): Loaded MRI sequences.
        labels_raw (dict): Loaded annotations.
        prim_seq (str): Primary sequence name.
        msg (str): Logging message.
        log_level (str): Logging level for indentation.

    Returns:
        mris (dict): MRI scans with corrected orientation and data type.
        annotations (dict): Annotations with corrected orientation and data type or filled with zeros.
        msg (str): Updated log message.
    """
    start = time.time()
    msg += f'{log_level}Fixing orientation and data type...\n'
    mris = {}
    annotations = {}

    # Validate and fix MRIs
    for sequence_name in sequences_raw:
        mris[sequence_name] = sequences_raw[sequence_name]
        msg += f'{log_level}Found {sequence_name} MRI sequence of shape {mris[sequence_name].shape}\n'

        mris[sequence_name] = nib.as_closest_canonical(mris[sequence_name])
        mris[sequence_name].set_data_dtype(np.float32) 

        orientation = nib.aff2axcodes(mris[sequence_name].affine)
        if orientation != ('R', 'A', 'S'):
            raise ValueError(f"Image {sequence_name} does not have RAS orientation.")

    # Validate and fix annotations
    for sequence_name in sequences_raw:
        if sequence_name in labels_raw.keys():
            annotations[sequence_name] = labels_raw[sequence_name]
            msg += f'{log_level}Found {sequence_name} annotation of shape {annotations[sequence_name].shape}\n'
        else:
            annotations[sequence_name] = nib.Nifti1Image(np.zeros(shape=mris[prim_seq].shape),
                                                            affine=mris[prim_seq].affine,
                                                            header=mris[prim_seq].header)
            msg += f'{log_level}Missing {sequence_name} annotation, filling with 0s\n'

        annotations[sequence_name] = nib.as_closest_canonical(annotations[sequence_name])
        annotations[sequence_name].set_data_dtype(np.uint8)

        orientation = nib.aff2axcodes(annotations[sequence_name].affine)
        if orientation != ('R', 'A', 'S'):
            raise ValueError(f"Annotation {sequence_name} does not have RAS orientation.")

    end = time.time()
    msg += f'{log_level}Orientation and data type adjustment took {end - start} seconds.\n\n'

    return mris, annotations, msg

def resample(args, source_image, target_image, interpolation, is_annotation=False,
            isotropic=False, source_sequence=None, target_sequence=None, msg='', 
            log_level="\t\t"):
    '''
    Resamples source image to target image (no registration is performed).
    Args:
        source_image (nib.Nifti1Image): Source image being resampled.
        target_image (nib.Nifti1Image): Target image to which source is being resampled to.
        interpolation (str): Resampling method (one of nearest, linear and continuous).
        is_annotation (bool): Whether the source image is an annotation.
        isotropic (bool): Whether to resample to isotropic (uses only source image).
        source_sequence (str)(optional): Source sequence (for logging purposes only).
        target_sequence (str)(optional): Target sequence (for logging purposes only).
        msg (str)(optional): Log message.
    Returns:
        resampled_image (nib.Nifti1Image): Resampled source image.
        msg (str): Log message.
    '''
    if isotropic:
        msg += f'{log_level}Resampling {source_sequence} MRI to isotropic of voxel size {args.voxel_size} using {interpolation} interpolation...\n'
        msg += f'{log_level}\tShape before resampling: {source_image.shape}\n'

        desired_voxel_size = float(args.voxel_size)
        isotropic_affine = np.diag([desired_voxel_size, desired_voxel_size, desired_voxel_size])
        resampled_image = resample_img(source_image, target_affine=isotropic_affine,
                                        interpolation=interpolation,
                                        fill_value=np.min(source_image.get_fdata()),
                                        order='F')

    elif is_annotation:
        msg += f'{log_level}Resampling {source_sequence} annotation to {target_sequence} using {interpolation} interpolation...\n'
        msg += f'{log_level}\tShape before resampling: {source_image.shape}\n'

        if interpolation == 'nearest':
            resampled_image = resample_to_img(source_image, target_image,
                                                interpolation=interpolation,
                                                fill_value=0)

        elif interpolation == 'linear':
            resampled_image = np.zeros(target_image.shape, dtype=np.float32)

            unique_labels = np.rint(np.unique(source_image.get_fdata()))
            for unique_label in unique_labels:

                annotation_binary = nib.Nifti1Image(
                    (np.rint(source_image.get_fdata()) == unique_label).astype(np.float32),
                    source_image.affine, source_image.header)

                annotation_binary = resample_to_img(annotation_binary, target_image,
                                                    interpolation=interpolation, fill_value=0)

                resampled_image[annotation_binary.get_fdata() >= 0.5] = unique_label

            resampled_image = nib.Nifti1Image(resampled_image, affine=annotation_binary.affine,
                                                header=annotation_binary.header)

    else:
        msg += f'{log_level}Resampling {source_sequence} MRI to {target_sequence} using {interpolation} interpolation...\n'
        msg += f'{log_level}\tShape before resampling: {source_image.shape}\n'

        resampled_image = resample_to_img(source_image, target_image, interpolation=interpolation,
                                            fill_value=np.min(source_image.get_fdata()))

    msg += f'{log_level}Shape after resampling: {resampled_image.shape}\n'

    return resampled_image, msg

def resample_mris_and_annotations(args, mris, annotations, primary_sequence, isotropic, msg=''):
    '''
    Resamples MRIs and annotations to primary sequence space.
    Args:
        mris (dict): Dictionary of MRIs.
        annotations (dict): Dictionary of annotations.
        primary_sequence (str): Sequence to which other sequences are being resampled to.
        isotropic (bool): Whether to resample to isotropic (uses only source image).
        msg (str)(optional): Log message.
    Returns:
        mris (dict): Dictionary of resampled MRIs.
        annotations (dict): Dictionary of resampled annotations.
        msg (str): Log message.
    '''
    msg += '\tResampling MRIs and annotations maps...\n'

    mris_copy = mris.copy()
    annotations_copy = annotations.copy()
    start = time.time()

    if isotropic:
        mris_copy[primary_sequence], msg = resample(args, source_image=mris_copy[primary_sequence],
                                                target_image=None,
                                                interpolation='linear',
                                                isotropic=True,
                                                source_sequence=primary_sequence,
                                                target_sequence=primary_sequence, msg=msg)
    for sequence in mris_copy:
        # resample MRI
        if sequence != primary_sequence:
            mris_copy[sequence], msg = resample(args, source_image=mris_copy[sequence],
                                            target_image=mris_copy[primary_sequence],
                                            interpolation='continuous',
                                            source_sequence=sequence,
                                            target_sequence=primary_sequence, msg=msg)
        # resample annotation
        annotations_copy[sequence], msg = resample(args, source_image=annotations_copy[sequence],
                                                target_image=mris_copy[primary_sequence],
                                                interpolation='nearest', # bcs binary mask
                                                is_annotation=True,
                                                source_sequence=sequence,
                                                target_sequence=primary_sequence, msg=msg)

    end = time.time()
    msg += f'\t\tResampling of MRIs and annotations took {end - start} seconds!\n\n'

    return mris_copy, annotations_copy, msg


def crop_and_concatenate(mris, annotations, primary_sequence, save_sequence_order, msg=''):
    '''
    Crops and concatenates MRIs and annotations to non-zero region.
    Args:
        mris (nib.Nifti1Image): Input MRIs.
        annotations (nib.Nifti1Image): Input annotations.
        primary_sequence (str): Sequence to which other sequences are being resampled to.
        save_sequence_order ([str]): Save sequence order.
        msg (str)(optional): Log message.
    Returns:
        cropped_mris (np.ndarray): Cropped MRIs array.
        cropped_annotations (np.ndarray): Cropped annotations array.
        msg (str): Log message.
    '''
    mris_copy = mris.copy()
    annotations_copy = annotations.copy()
    msg += '\tCropping and concatenating MRIs and annotations...\n'

    start = time.time()

    # get brain mask from primary sequence
    mask = process_masks.get_brain_mask(image=mris_copy[primary_sequence])

    x, y, z = np.where(mask == 1)
    coordinates = {'x': [np.min(x), np.max(x)], 'y': [np.min(y), np.max(y)],
                    'z': [np.min(z), np.max(z)]}

    # concatenate MRIs and annotations
    mris_array, annotations_array = [], []

    for sequence in save_sequence_order:
        mris_array.append(mris_copy[sequence].get_fdata()[..., None])
        annotations_array.append(annotations_copy[sequence].get_fdata()[..., None])

    mris_array = np.concatenate(mris_array, axis=-1)
    annotations_array = np.concatenate(annotations_array, axis=-1)

    msg += f'\t\tMRIs shape after concatenation: {mris_array.shape}\n'
    msg += f'\t\tAnnotations shape after concatenation: {annotations_array.shape}\n'

    # crop MRIs and annotations by applying brain mask
    cropped_mris = mris_array[coordinates['x'][0]:coordinates['x'][1],
                                coordinates['y'][0]:coordinates['y'][1],
                                coordinates['z'][0]:coordinates['z'][1], :]

    cropped_annotations = annotations_array[coordinates['x'][0]:coordinates['x'][1],
                                            coordinates['y'][0]:coordinates['y'][1],
                                            coordinates['z'][0]:coordinates['z'][1], :]

    msg += f'\t\tMRIs shape after cropping: {cropped_mris.shape}\n'
    msg += f'\t\tAnnotations shape after cropping: {cropped_annotations.shape}\n'

    end = time.time()
    msg += f'\t\tCropping and concatenation of MRIs and annotations took {end - start} seconds!\n\n'

    return cropped_mris, cropped_annotations, msg

def apply_synthstrip(synthstrip_docker, image_path, tmp_dir_path):
    '''
    Applies SynthStrip algorithm on MRI image.
    Args:
        image_path (str): Path to the MRI image.
        tmp_dir_path (str): Path to directory where temporary files will be saved.
    Returns:
        skull_stripped_image (nib.Nifti1Image): Skullstripped MRI image.
        mask (nib.Nifti1Image): Brain mask.
    '''
    image_filename = os.path.abspath(image_path).split(os.sep)[-1]

    tmp_image_mask_path = os.path.join(
        tmp_dir_path, image_filename.replace('.nii.gz', '_brain_mask.nii.gz'))
    tmp_image_synthstripped_path = os.path.join(
        tmp_dir_path, image_filename.replace('.nii.gz', '_skull_stripped.nii.gz'))

    image = nib.load(image_path)
    image_data = image.get_fdata()
    image_data = (image_data - np.mean(image_data)) / np.std(image_data)
    image_normalized = utils_gen.create_nifti(image_data, affine=image.affine, header=image.header, is_annotation=False)
    tmp_image_normalized_path = os.path.join(tmp_dir_path, image_filename)
    nib.save(image_normalized, tmp_image_normalized_path)

    command_list = [synthstrip_docker, '-i', tmp_image_normalized_path, '-o',
                    tmp_image_synthstripped_path, '-m', tmp_image_mask_path]
    print(" ".join(command_list))
    subprocess.run(command_list,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    check=True, universal_newlines=True)

    skull_stripped_image = nib.load(tmp_image_synthstripped_path)
    mask = nib.load(tmp_image_mask_path)
    mask_data = mask.get_fdata()

    # create skull-stripped image
    image_data = image.get_fdata()
    image_data[mask_data == 0] = 0 # set to 0 background
    skull_stripped_image = utils_gen.create_nifti(image_data, affine=image.affine, header=image.header, is_annotation=False)

    return skull_stripped_image, mask

def skull_strip(synthstrip_docker, subject, mris, tmp_dir_path, msg=''):
    '''
    Skull-strips MRIs using SynthStrip algorithm.
    Args:
        mris (dict): Dictionary of MRIs.
        mri_paths (dict): Dictionary of MRI paths.
        tmp_dir_path (str): Path to directory where temporary files will be saved.
        msg (str)(optional): Log message.
    Returns:
        mris (dict): Dictionary of skull-stripped MRIs.
        msg (str): Log message.
    '''
    msg += '\tPerforming skull-stripping using SynthStrip algorithm...\n'

    start = time.time()
    mris_copy = mris.copy()
    brain_masks = {}
    for sequence in mris_copy:
        
        mri_path = os.path.join(tmp_dir_path, f"{subject}_{sequence}.nii.gz")
        nib.save(mris_copy[sequence], mri_path)

        msg += '\t\tPerforming skull-stripping on {}...\n'.format(sequence)

        skull_stripped_image, brain_mask = apply_synthstrip(synthstrip_docker, mri_path, tmp_dir_path)
        mris_copy[sequence] = skull_stripped_image
        brain_masks[sequence] = brain_mask

    end = time.time()
    msg += '\t\tSkull-stripping of MRIs took {} seconds!\n\n'.format(end-start)

    return mris_copy, brain_masks, msg


def combine_annotations(annotations, priorities, msg=''):
    '''
    Combines multi-channel annotations to single-channel according to label priotiries.
    Args:
        annotations (np.array): Annotations array.
        priorities ([int]): Label priorities.
        msg (str)(optional): Log message.
    Returns:
        combined_annotations (np.array): Combined annotations array.
        msg (optional): Log message.
    '''
    
    # TODO: if with future datasets several labels, combine here. 
    
    # For now let's just take first channel (T2S)
    combined_annotations = annotations[:, :, :, 0]
    
    return combined_annotations, msg

def crop_mris_and_annotations(mris, annotations, brain_masks, prim_seq, msg, log_level="\t"):
    """
    Crops MRIs and annotations to brain-only regions based on the provided brain masks,
    correctly updating the header to reflect the new image dimensions.

    Args:
        mris (dict): Dictionary of MRI scans keyed by sequence type.
        annotations (dict): Dictionary of annotations keyed by sequence type.
        brain_masks (dict): Dictionary of brain masks keyed by sequence type.
        prim_seq (str): The primary sequence key to use for cropping dimensions.
        msg (str): Log message for tracking.
        log_level (str): Indentation level for log messages.

    Returns:
        mris_cropped (dict): Dictionary of cropped MRI scans.
        annotations_cropped (dict): Dictionary of cropped annotations.
        msg (str): Updated log message.
    """
    msg += f'{log_level}Cropping MRIs and annotations to brain-only regions using nibabel slicer...\n'
    start = time.time()

    # Use the primary sequence's mask to define cropping coordinates
    mask_array = brain_masks[prim_seq].get_fdata().astype(bool)
    x, y, z = np.where(mask_array)
    coordinates = {
        'x': slice(np.min(x), np.max(x) + 1),
        'y': slice(np.min(y), np.max(y) + 1),
        'z': slice(np.min(z), np.max(z) + 1)
    }

    # Crop MRIs and annotations by applying the brain mask coordinates
    mris_cropped = {}
    annotations_cropped = {}
    for seq in mris:
        # The slice object allows us to use slicing directly on nibabel images
        mris_cropped[seq] = mris[seq].slicer[coordinates['x'], coordinates['y'], coordinates['z']]

    for seq in annotations:
        annotations_cropped[seq] = annotations[seq].slicer[coordinates['x'], coordinates['y'], coordinates['z']]

    end = time.time()
    msg += f'{log_level}Cropping completed in {end - start:.2f} seconds.\n'

    return mris_cropped, annotations_cropped, msg