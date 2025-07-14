'''
author: zyx
date: 2025-05-23
last modified: 2025-05-25
description: 
    Functions for morphological operations
'''
import numpy as np

from skimage.morphology import erosion, ball, disk
from scipy import ndimage
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

def erode_binary_mask_2D(mask: np.ndarray,
                         radius: int) -> np.ndarray:
    '''
    Fast erosion for 2D binary masks (single label).
    
    Parameters:
    -----------
    mask : np.ndarray
        Input binary mask (will be binarized with mask > 0)
    radius : int
        Radius of the disk structuring element
    
    Returns:
    --------
    np.ndarray (bool)
        Eroded binary mask
    
    Note:
    -----
    This function is optimized for speed on binary masks.
    For multi-label masks, use erode_mask_2D_with_ball instead.
    '''
    binary_mask = mask > 0
    
    if radius == 0:
        return binary_mask
    
    if HAS_OPENCV:
        # OpenCV is typically 5-10x faster than skimage
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
        return cv2.erode(binary_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    else:
        # SciPy's binary_erosion is still 2-3x faster than skimage
        struct_elem = disk(radius)
        return ndimage.binary_erosion(binary_mask, structure=struct_elem)


def erode_mask_2D_with_ball(mask: np.ndarray,
                            radius: int) -> np.ndarray:
    '''
    Erode a 2D multi-label mask by a disk of given radius.
    Each label is eroded separately to preserve object identities.
    
    Parameters:
    -----------
    mask : np.ndarray
        Input mask with multiple labels (0 is background)
    radius : int
        Radius of the disk structuring element
    
    Returns:
    --------
    np.ndarray
        Eroded mask with preserved labels
    
    Note:
    -----
    For single-label binary masks, use erode_binary_mask_2D for better performance.
    '''
    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels > 0]  # Exclude background
    
    if len(unique_labels) == 0:
        return mask
    
    # If only one label, use fast binary erosion
    if len(unique_labels) == 1:
        binary_result = erode_binary_mask_2D(mask, radius)
        result = np.zeros_like(mask)
        result[binary_result] = unique_labels[0]
        return result
    
    # Multi-label erosion
    result = np.zeros_like(mask)
    
    # Create structuring element once
    if HAS_OPENCV and radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    else:
        struct_elem = disk(radius)
    
    for label in unique_labels:
        binary_label = (mask == label)
        
        if HAS_OPENCV and radius > 0:
            eroded = cv2.erode(binary_label.astype(np.uint8), kernel, iterations=1).astype(bool)
        else:
            eroded = ndimage.binary_erosion(binary_label, structure=struct_elem)
        
        result[eroded] = label
    
    return result

def generate_binary_donut_2D(mask: np.ndarray,
                            radius: int) -> np.ndarray:
    '''
    Fast generation of a 2D binary donut mask.
    
    Parameters:
    -----------
    mask : np.ndarray
        Input binary mask (will be binarized with mask > 0)
    radius : int
        Radius of erosion for the inner boundary
    
    Returns:
    --------
    np.ndarray (bool)
        Binary donut mask (True in the donut region)
    
    Note:
    -----
    This function is optimized for speed on binary masks.
    For multi-label masks, use generate_2D_donut instead.
    '''
    binary_mask = mask > 0
    eroded_mask = erode_binary_mask_2D(mask, radius)
    return binary_mask & ~eroded_mask


def generate_2D_donut(mask: np.ndarray,
                      radius: int) -> np.ndarray:
    '''
    Generate a 2D donut mask by eroding the input mask with a disk of given radius.
    Preserves multiple labels if present.
    
    Parameters:
    -----------
    mask : np.ndarray
        Input mask (can be binary or multi-label)
    radius : int
        Radius of erosion for the inner boundary
    
    Returns:
    --------
    np.ndarray
        Donut mask with preserved labels
    
    Note:
    -----
    For single-label binary masks, use generate_binary_donut_2D for better performance.
    '''
    eroded_mask = erode_mask_2D_with_ball(mask, radius)
    
    # Check if mask has multiple labels
    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels > 0]
    
    if len(unique_labels) <= 1:
        # Binary mask case
        return generate_binary_donut_2D(mask, radius)
    else:
        # Multi-label case: preserve original labels in the donut region
        donut_mask = mask.copy()
        donut_mask[eroded_mask > 0] = 0
        return donut_mask
