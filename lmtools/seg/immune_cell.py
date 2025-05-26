'''
author: zyx
date: 2025-05-25
last modified: 2025-05-25
description: 
    Integrated functions for immune cell segmentation filtering
    Yupu pipeline
'''
import argparse
from pathlib import Path
import numpy as np
import tifffile

from scipy.ndimage import binary_erosion, find_objects
from skimage.morphology import ball
from skimage.filters import threshold_otsu
from skimage.measure import label as sk_label
from skimage.segmentation import relabel_sequential

from morphology import erode_mask_2D_with_ball, generate_2D_donut

import matplotlib.pyplot as plt

def load_segmentation(path: Path) -> np.ndarray:
    """Load a .npy segmentation mask."""
    return np.load(path, allow_pickle=True)

def load_image(path: Path) -> np.ndarray:
    """Load an image volume or slice (TIFF)."""
    return tifffile.imread(path)

class DataPaths:
    """
    Encapsulate file paths for segmentation masks and raw images for each channel.
    """
    def __init__(
        self,
        cy5_seg: Path,
        cy5_img: Path,
        dapi_seg: Path,
        dapi_img: Path,
        cd11b_img: Path,
        qupath_seg: Path
    ):
        # segmentation masks:
        self.cy5_seg = Path(cy5_seg)
        self.dapi_seg = Path(dapi_seg)
        self.qupath_seg = Path(qupath_seg)
        # raw images:
        self.cy5_img = Path(cy5_img)
        self.dapi_img = Path(dapi_img)
        self.cd11b_img = Path(cd11b_img)

    def load_segs(self):
        """
        Returns tuple of segmentation masks: (CD45, DAPI, QuPath).
        """
        seg_cy5 = load_segmentation(self.cy5_seg)
        seg_dapi = load_segmentation(self.dapi_seg)
        seg_qupath = load_segmentation(self.qupath_seg)
        return seg_cy5, seg_dapi, seg_qupath

    def load_imgs(self):
        """
        Returns tuple of raw image arrays: (CD45, DAPI, CD11b).
        """
        img_cy5 = load_image(self.cy5_img)
        img_dapi = load_image(self.dapi_img)
        img_cd11b = load_image(self.cd11b_img)
        return img_cy5, img_dapi, img_cd11b

def compute_statistics():
    pass

def filter_by_overlap(
    seg_mask: np.ndarray,
    ref_mask: np.ndarray,
    min_overlap_ratio: float
) -> np.ndarray:
    """
    Keep only objects with overlap(ref_mask)/area >= min_overlap_ratio.
    """
    filtered = seg_mask.copy()
    objects = find_objects(seg_mask)
    labels = np.unique(seg_mask)[1:]
    for lab in labels:
        slc = objects[lab-1]
        if slc is None:
            continue
        region = (seg_mask[slc]==lab)
        overlap = (region & (ref_mask[slc]>0)).sum()
        area = region.sum()
        if overlap/area < min_overlap_ratio:
            filtered[slc][region] = 0
    return filtered


def size_and_dapi_filter(
    seg_mask: np.ndarray,
    dapi_mask: np.ndarray,
    dapi_img: np.ndarray,
    min_size: int,
    min_overlap_ratio: float,
    min_dapi_intensity: float = None
) -> np.ndarray:
    """
    1) Remove objects smaller than `min_size` always.
    2) Remove objects with DAPI-overlap < `min_overlap_ratio`, unless the object's
       mean DAPI intensity >= `min_dapi_intensity` (if provided).
    """
    filtered = seg_mask.copy()
    objects = find_objects(seg_mask)
    labels = np.unique(seg_mask)[1:]

    for lab in labels:
        slc = objects[lab - 1]
        if slc is None:
            continue
        region = (seg_mask[slc] == lab)
        size = region.sum()

        # always enforce min_size
        if size < min_size:
            filtered[slc][region] = 0
            continue

        # compute overlap ratio
        overlap = (region & (dapi_mask[slc] > 0)).sum()
        overlap_ratio = overlap / size if size > 0 else 0.0

        # compute mean DAPI intensity if threshold is set
        mean_intensity = None
        if min_dapi_intensity is not None:
            pix = dapi_img[slc][region]
            mean_intensity = pix.mean() if pix.size > 0 else 0.0

        # remove by overlap ratio unless intensity safeguard applies
        if overlap_ratio < min_overlap_ratio:
            if min_dapi_intensity is None or mean_intensity < min_dapi_intensity:
                filtered[slc][region] = 0

    return filtered

def compute_average_intensity(
    seg_mask: np.ndarray,
    intensity_img: np.ndarray,
    use_donut: bool=False,
    erode_radius: int=1
) -> dict:
    """
    Returns dict {label: mean intensity} per object; ring if use_donut.
    """
    avgs = {}
    objs = find_objects(seg_mask)
    labs = np.unique(seg_mask)[1:]
    for lab in labs:
        slc = objs[lab-1]
        if slc is None:
            continue
        region = (seg_mask[slc]==lab)
        if use_donut:
            region_ring = generate_2D_donut(region, erode_radius)
            pix = intensity_img[slc][region_ring]
        else:
            pix = intensity_img[slc][region]
        if pix.size>0:
            avgs[lab] = pix.mean()
    return avgs

def threshold_otsu_values(values: np.ndarray) -> float:
    return threshold_otsu(values)

def intensity_filter(
    seg_mask: np.ndarray,
    avg_int: dict,
    upper_thresh: float,
    lower_thresh: float=0,
) -> np.ndarray:
    filt = seg_mask.copy()
    objs = find_objects(seg_mask)
    for lab, mean in avg_int.items():
        slc = objs[lab-1]
        if slc is None:
            continue
        region = (seg_mask[slc]==lab)
        if mean>upper_thresh or mean<lower_thresh:
            filt[slc][region]=0
    return filt

def reassign_labels(mask: np.ndarray) -> np.ndarray:
    return sk_label(mask>0)

def relabel_sequential_labels(mask: np.ndarray) -> np.ndarray:
    nm, _, _ = relabel_sequential(mask)
    return nm

def count_cells(mask: np.ndarray) -> int:
    return len(np.unique(mask)[1:])


#### Visualize
def display_example(
    image: np.ndarray,
    seg_mask: np.ndarray,
    slice_idx: int=0
):
    fig, axes = plt.subplots(1,2,figsize=(10,5))
    img = image[slice_idx] if image.ndim==3 else image
    seg = seg_mask[slice_idx] if seg_mask.ndim==3 else seg_mask
    axes[0].imshow(img, cmap='gray'); axes[0].set_title(f'Image slice {slice_idx}')
    axes[1].imshow(seg, cmap='nipy_spectral'); axes[1].set_title(f'Seg slice {slice_idx}')
    plt.tight_layout()
    plt.show()

def plot_dirc_distribution(intensity_dict):
    """
    Plot the distribution of intensity values from a dictionary
    
    Args:
        intensity_dict: Dictionary with intensity values
    """
    # Extract intensity values from the dictionary
    intensity_values = list(intensity_dict.values())
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(intensity_values, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel('Mean Intensity')
    plt.ylabel('Frequency')
    plt.title('Distribution of Intensity Values')
    plt.grid(alpha=0.3)
    
    # Add vertical line for mean and median
    mean_intensity = np.mean(intensity_values)
    median_intensity = np.median(intensity_values)
    plt.axvline(mean_intensity, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_intensity:.2f}')
    plt.axvline(median_intensity, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median_intensity:.2f}')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return mean_intensity, median_intensity