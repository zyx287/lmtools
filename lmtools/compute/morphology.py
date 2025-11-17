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
from scipy.ndimage import distance_transform_edt
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

# GPU support with CuPy
try:
    import cupy as cp
    from cupyx.scipy import ndimage as cp_ndimage
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    cp = None
    cp_ndimage = None

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


def erode_mask_2D_with_dt(mask: np.ndarray, radius: int) -> np.ndarray:
    '''
    Fast erosion for 2D masks using Euclidean distance transform.
    
    Parameters:
    -----------
    mask : np.ndarray
        Input mask (will be binarized with mask > 0)
    radius : int
        Erosion radius in pixels
    
    Returns:
    --------
    np.ndarray (bool)
        Eroded binary mask where distance from edge > radius
    
    Note:
    -----
    This method is often faster than morphological erosion for large radii.
    It computes the distance transform and thresholds at the specified radius.
    '''
    # Binarize the mask
    binary_mask = mask > 0
    
    if radius == 0:
        return binary_mask
    
    # Compute Euclidean distance transform
    dist_transform = distance_transform_edt(binary_mask)
    
    # Return mask where distance from edge is greater than radius
    return dist_transform > radius
    # Usage: eroded = erode_mask_2D_with_dt(tissue_mask, erosion_radius=10)


def generate_2D_donut_dt(mask: np.ndarray, radius: int) -> np.ndarray:
    '''
    Generate a 2D donut mask using distance transform erosion.
    
    Parameters:
    -----------
    mask : np.ndarray
        Input mask (will be binarized with mask > 0)
    radius : int
        Radius of erosion for the inner boundary
    
    Returns:
    --------
    np.ndarray (bool)
        Binary donut mask (True in the shell region)
    
    Note:
    -----
    Returns the "shell" between the original mask and the eroded mask.
    This is the region within 'radius' pixels of the mask boundary.
    '''
    # Get original binary mask
    binary_mask = mask > 0
    
    if radius == 0:
        return np.zeros_like(binary_mask, dtype=bool)
    
    # Get eroded mask using distance transform
    eroded_mask = erode_mask_2D_with_dt(mask, radius)
    
    # Return the shell (original minus eroded)
    return binary_mask & ~eroded_mask
    # Usage: donut = generate_2D_donut_dt(tissue_mask, radius=10)


def check_gpu_available() -> bool:
    '''
    Check if GPU acceleration is available for morphological operations.
    
    Returns:
    --------
    bool
        True if CuPy is installed and GPU is available
    '''
    if not HAS_GPU:
        return False
    
    try:
        # Try to create a small array on GPU
        test = cp.array([1, 2, 3])
        del test
        return True
    except Exception:
        return False


def erode_mask_2D_gpu(mask: np.ndarray, radius: int) -> np.ndarray:
    '''
    GPU-accelerated erosion for 2D masks using CuPy.
    
    Parameters:
    -----------
    mask : np.ndarray
        Input mask (will be binarized with mask > 0)
    radius : int
        Erosion radius in pixels
    
    Returns:
    --------
    np.ndarray (bool)
        Eroded binary mask
    
    Note:
    -----
    Requires CuPy and CUDA-capable GPU. Falls back to EDT if GPU unavailable.
    For best performance, use with large images and radii.
    '''
    if not check_gpu_available():
        # Fallback to EDT method
        return erode_mask_2D_with_dt(mask, radius)
    
    # Binarize the mask
    binary_mask = mask > 0
    
    if radius == 0:
        return binary_mask
    
    try:
        # Transfer to GPU
        gpu_mask = cp.asarray(binary_mask)
        
        # Create structuring element on GPU
        y, x = cp.ogrid[-radius:radius+1, -radius:radius+1]
        struct_elem = (x**2 + y**2 <= radius**2)
        
        # Perform erosion on GPU
        gpu_eroded = cp_ndimage.binary_erosion(gpu_mask, structure=struct_elem)
        
        # Transfer back to CPU
        result = cp.asnumpy(gpu_eroded)
        
        # Clean up GPU memory
        del gpu_mask, gpu_eroded, struct_elem
        cp.get_default_memory_pool().free_all_blocks()
        
        return result
        
    except Exception as e:
        # Fallback to EDT on error
        print(f"GPU erosion failed: {e}, falling back to EDT")
        return erode_mask_2D_with_dt(mask, radius)
    # Usage: eroded = erode_mask_2D_gpu(tissue_mask, radius=100)


def erode_mask_2D_gpu_edt(mask: np.ndarray, radius: int) -> np.ndarray:
    '''
    GPU-accelerated erosion using Euclidean distance transform with CuPy.
    
    Parameters:
    -----------
    mask : np.ndarray
        Input mask (will be binarized with mask > 0)
    radius : int
        Erosion radius in pixels
    
    Returns:
    --------
    np.ndarray (bool)
        Eroded binary mask where distance from edge > radius
    
    Note:
    -----
    Combines the speed of EDT with GPU acceleration for maximum performance.
    Particularly effective for very large images.
    '''
    if not check_gpu_available():
        # Fallback to CPU EDT
        return erode_mask_2D_with_dt(mask, radius)
    
    # Binarize the mask
    binary_mask = mask > 0
    
    if radius == 0:
        return binary_mask
    
    try:
        # Transfer to GPU
        gpu_mask = cp.asarray(binary_mask)
        
        # Compute EDT on GPU
        gpu_dist = cp_ndimage.distance_transform_edt(gpu_mask)
        
        # Threshold on GPU
        gpu_result = gpu_dist > radius
        
        # Transfer back to CPU
        result = cp.asnumpy(gpu_result)
        
        # Clean up GPU memory
        del gpu_mask, gpu_dist, gpu_result
        cp.get_default_memory_pool().free_all_blocks()
        
        return result
        
    except Exception as e:
        # Fallback to CPU EDT on error
        print(f"GPU EDT failed: {e}, falling back to CPU EDT")
        return erode_mask_2D_with_dt(mask, radius)
    # Usage: eroded = erode_mask_2D_gpu_edt(tissue_mask, radius=100)


def generate_2D_donut_gpu(mask: np.ndarray, radius: int) -> np.ndarray:
    '''
    Generate a 2D donut mask using GPU-accelerated erosion.
    
    Parameters:
    -----------
    mask : np.ndarray
        Input mask (will be binarized with mask > 0)
    radius : int
        Radius of erosion for the inner boundary
    
    Returns:
    --------
    np.ndarray (bool)
        Binary donut mask (True in the shell region)
    
    Note:
    -----
    Uses GPU EDT erosion for maximum speed on large images.
    '''
    # Get original binary mask
    binary_mask = mask > 0
    
    if radius == 0:
        return np.zeros_like(binary_mask, dtype=bool)
    
    # Get eroded mask using GPU
    eroded_mask = erode_mask_2D_gpu_edt(mask, radius)
    
    # Return the shell (original minus eroded)
    return binary_mask & ~eroded_mask
    # Usage: donut = generate_2D_donut_gpu(tissue_mask, radius=50)
