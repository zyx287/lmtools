'''
author: zyx
date: 2025-05-25
last modified: 2025-06-10
description: 
    Integrated functions for immune cell segmentation filtering
    Yupu pipeline
'''
import argparse
from pathlib import Path
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any
from scipy.ndimage import binary_erosion, find_objects
from skimage.morphology import ball
from skimage.filters import threshold_otsu
from skimage.measure import label as sk_label
from skimage.segmentation import relabel_sequential

from lmtools.compute.morphology import erode_mask_2D_with_ball, generate_2D_donut
from lmtools.compute.intensity_threshold import compute_gmm_component
from lmtools.io.metadata_tracking import (
    ProcessingStep,
    DataPaths,
    load_segmentation,
    load_image
)

import matplotlib.pyplot as plt


def filter_by_overlap(
    seg_mask: np.ndarray,
    ref_mask: np.ndarray,
    min_overlap_ratio: float,
    data_paths: Optional[DataPaths] = None,
    step_name: str = "overlap_filter"
) -> np.ndarray:
    '''
    Keep only objects with overlap(ref_mask)/area >= min_overlap_ratio.
    '''
    filtered = seg_mask.copy()
    objects = find_objects(seg_mask)
    labels = np.unique(seg_mask)[1:]
    removed_count = 0
    
    for lab in labels:
        slc = objects[lab-1]
        if slc is None:
            continue
        region = (seg_mask[slc]==lab)
        overlap = (region & (ref_mask[slc]>0)).sum()
        area = region.sum()
        if area > 0 and overlap/area < min_overlap_ratio:
            filtered[slc][region] = 0
            removed_count += 1
    
    # Track processing step if data_paths provided
    if data_paths:
        step = ProcessingStep(
            step_name=step_name,
            timestamp=datetime.now().isoformat(),
            parameters={
                "min_overlap_ratio": min_overlap_ratio,
                "removed_objects": removed_count,
                "remaining_objects": len(labels) - removed_count
            },
            input_data=["segmentation_mask", "reference_mask"],
            notes=f"Filtered objects with overlap ratio < {min_overlap_ratio}"
        )
        data_paths.metadata.add_step(step)
    
    return filtered


def size_and_dapi_filter(
    seg_mask: np.ndarray,
    dapi_mask: np.ndarray,
    dapi_img: np.ndarray,
    min_size: int,
    min_overlap_ratio: float,
    min_dapi_intensity: float = None
) -> np.ndarray:
    '''
    1) Remove objects smaller than `min_size` always.
    2) Remove objects with DAPI-overlap < `min_overlap_ratio`, unless the object's
       mean DAPI intensity >= `min_dapi_intensity` (if provided).
    '''
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
    '''
    Returns dict {label: mean intensity} per object; ring if use_donut.
    '''
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

def intensity_filter(
    seg_mask: np.ndarray,
    avg_int: dict,
    upper_thresh: float,
    lower_thresh: float=0,
    data_paths: Optional[DataPaths] = None,
    intensity_channel: Optional[str] = None,
    step_name: str = "intensity_filter"
) -> np.ndarray:
    filt = seg_mask.copy()
    objs = find_objects(seg_mask)
    removed_count = 0
    
    for lab, mean in avg_int.items():
        slc = objs[lab-1]
        if slc is None:
            continue
        region = (seg_mask[slc]==lab)
        if mean>upper_thresh or mean<lower_thresh:
            filt[slc][region]=0
            removed_count += 1
    
    # Track processing step and intensity source
    if data_paths:
        if intensity_channel:
            data_paths.metadata.intensity_filter_sources[step_name] = intensity_channel
        
        step = ProcessingStep(
            step_name=step_name,
            timestamp=datetime.now().isoformat(),
            parameters={
                "upper_threshold": upper_thresh,
                "lower_threshold": lower_thresh,
                "intensity_channel": intensity_channel,
                "removed_objects": removed_count,
                "remaining_objects": len(avg_int) - removed_count
            },
            input_data=["segmentation_mask", f"intensity_image_{intensity_channel}" if intensity_channel else "intensity_image"],
            notes=f"Filtered objects with intensity outside [{lower_thresh}, {upper_thresh}]"
        )
        data_paths.metadata.add_step(step)
    
    return filt

def compute_gmm_threshold(
    intensity_dict: Dict[int, float],
    n_components: int = 3,
    exclude_components: int = 1,
    n_delta: float = 2.0,
    data_paths: Optional[DataPaths] = None,
    intensity_channel: Optional[str] = None,
    step_name: str = "gmm_threshold"
) -> Tuple[float, Dict[str, Any]]:
    '''
    Compute threshold using Gaussian Mixture Model (GMM) on intensity values.
    
    Parameters
    ----------
    intensity_dict : Dict[int, float]
        Dictionary mapping cell labels to mean intensities (from compute_average_intensity)
    n_components : int, default=3
        Number of GMM components to fit
    exclude_components : int, default=1
        Number of lowest components to exclude as noise/background
    n_delta : float, default=2.0
        Number of standard deviations to add to the mean of the excluded component
    data_paths : Optional[DataPaths]
        DataPaths instance for metadata tracking
    intensity_channel : Optional[str]
        Name of the intensity channel being analyzed
    step_name : str, default="gmm_threshold"
        Name for this processing step
    
    Returns
    -------
    threshold : float
        Computed threshold value (mean + n_delta * std of the highest excluded component)
    gmm_info : Dict[str, Any]
        Dictionary containing GMM parameters and statistics
    '''
    # Extract intensity values from dictionary
    intensity_values = np.array(list(intensity_dict.values()))
    
    # Fit GMM
    gmm_model = compute_gmm_component(intensity_values, n_components=n_components)
    
    # Get component parameters (means and standard deviations)
    means = gmm_model.means_.flatten()
    covariances = gmm_model.covariances_.flatten()
    stds = np.sqrt(covariances)
    weights = gmm_model.weights_
    
    # Sort components by mean value
    sorted_indices = np.argsort(means)
    sorted_means = means[sorted_indices]
    sorted_stds = stds[sorted_indices]
    sorted_weights = weights[sorted_indices]
    
    # Calculate threshold based on the highest excluded component
    if exclude_components > 0 and exclude_components <= n_components:
        # Use the highest excluded component (0-indexed)
        exclude_idx = exclude_components - 1
        threshold_mean = sorted_means[exclude_idx]
        threshold_std = sorted_stds[exclude_idx]
        threshold = threshold_mean + n_delta * threshold_std
    else:
        raise ValueError(f"exclude_components must be between 1 and {n_components}")
    
    # Prepare GMM info
    gmm_info = {
        "n_components": n_components,
        "exclude_components": exclude_components,
        "n_delta": n_delta,
        "threshold": threshold,
        "components": []
    }
    
    # Add info for each component
    for i in range(n_components):
        idx = sorted_indices[i]
        component_info = {
            "component": i + 1,
            "mean": float(sorted_means[i]),
            "std": float(sorted_stds[i]),
            "weight": float(sorted_weights[i]),
            "excluded": i < exclude_components
        }
        gmm_info["components"].append(component_info)
    
    # Add threshold calculation details
    gmm_info["threshold_calculation"] = {
        "component_used": exclude_components,
        "mean": float(threshold_mean),
        "std": float(threshold_std),
        "formula": f"mean + {n_delta} * std = {threshold_mean:.2f} + {n_delta} * {threshold_std:.2f} = {threshold:.2f}"
    }
    
    # Track in metadata if provided
    if data_paths:
        step = ProcessingStep(
            step_name=step_name,
            timestamp=datetime.now().isoformat(),
            parameters={
                "n_components": n_components,
                "exclude_components": exclude_components,
                "n_delta": n_delta,
                "intensity_channel": intensity_channel,
                "threshold": float(threshold),
                "n_cells_analyzed": len(intensity_dict),
                "gmm_components": gmm_info["components"],
                "threshold_calculation": gmm_info["threshold_calculation"]
            },
            input_data=[f"intensity_dict_{intensity_channel}" if intensity_channel else "intensity_dict"],
            notes=f"GMM threshold computed: {threshold:.2f} from component {exclude_components} ({threshold_mean:.2f} + {n_delta}*{threshold_std:.2f})"
        )
        data_paths.metadata.add_step(step)
    
    return threshold, gmm_info

def reassign_labels(mask: np.ndarray) -> np.ndarray:
    '''
    Relabel images based on connectivity (might hugely reduce the label count).
    '''
    return sk_label(mask>0)

def relabel_sequential_labels(mask: np.ndarray) -> np.ndarray:
    '''
    Relabel sequentially to ensure labels are contiguous and start from 1.
    Would not generate/remove any labels, just reassign them.
    '''
    nm, _, _ = relabel_sequential(mask)
    return nm

def count_cells(mask: np.ndarray) -> int:
    '''
    Count the number of unique cell labels in a segmentation mask, excluding background (0).
    This implementation uses find_objects for better performance with large masks.
    Parameters
    ----------
    mask : np.ndarray
        Segmentation mask with integer labels
    Returns
    -------
    int
        Count of unique cell labels (excluding background)
    '''
    if mask.size == 0 or mask.max() == 0:  # No cells
        return 0
    
    # find_objects returns a list of slices for each label
    objects = find_objects(mask)
    
    # Count non-None entries (meaning the label exists in the image)
    return sum(obj is not None for obj in objects)


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
    '''
    Plot the distribution of intensity values from a dictionary
    
    Args:
        intensity_dict: Dictionary with intensity values
    '''
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


def visualize_gmm_threshold(
    intensity_dict: Dict[int, float],
    gmm_info: Dict[str, Any],
    threshold: float,
    bins: int = 100,
    save_path: Optional[Path] = None
):
    '''
    Visualize GMM components and threshold on intensity distribution.
    
    Parameters
    ----------
    intensity_dict : Dict[int, float]
        Dictionary mapping cell labels to mean intensities
    gmm_info : Dict[str, Any]
        GMM information from compute_gmm_threshold
    threshold : float
        Computed threshold value
    bins : int, default=100
        Number of histogram bins
    save_path : Optional[Path]
        If provided, save figure to this path
    '''
    from scipy.stats import norm
    
    # Extract intensity values
    intensity_values = np.array(list(intensity_dict.values()))
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot histogram
    n, bins_edges, _ = plt.hist(intensity_values, bins=bins, density=True, 
                                alpha=0.6, color='skyblue', edgecolor='black',
                                label='Data histogram')
    
    # Plot individual GMM components
    x = np.linspace(intensity_values.min(), intensity_values.max(), 1000)
    colors = ['red', 'green', 'blue', 'orange', 'purple']
    
    for i, comp in enumerate(gmm_info['components']):
        mean = comp['mean']
        std = comp['std']
        weight = comp['weight']
        color = colors[i % len(colors)]
        
        # Plot Gaussian curve
        y = weight * norm.pdf(x, mean, std)
        label = f'Component {i+1}: μ={mean:.1f}, σ={std:.1f}, w={weight:.2f}'
        if comp['excluded']:
            label += ' (excluded)'
        plt.plot(x, y, color=color, linewidth=2, label=label)
        
        # Add vertical line at component mean
        plt.axvline(mean, color=color, linestyle='--', alpha=0.5)
    
    # Plot combined GMM
    combined_y = np.zeros_like(x)
    for comp in gmm_info['components']:
        combined_y += comp['weight'] * norm.pdf(x, comp['mean'], comp['std'])
    plt.plot(x, combined_y, 'k-', linewidth=2, label='Combined GMM')
    
    # Plot threshold
    plt.axvline(threshold, color='red', linestyle='-', linewidth=2, 
                label=f'Threshold: {threshold:.1f}')
    
    # Add shaded region for excluded cells
    plt.axvspan(intensity_values.min(), threshold, alpha=0.2, color='red',
                label='Excluded region')
    
    # Labels and title
    plt.xlabel('Mean Intensity', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.title('GMM Analysis of Intensity Distribution', fontsize=14)
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    
    # Add text with threshold calculation
    calc_info = gmm_info['threshold_calculation']
    text = f"Threshold = {calc_info['formula']}"
    plt.text(0.02, 0.98, text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()



# Example usage function
def example_usage():
    '''
    Example of how to use the enhanced DataPaths class with metadata tracking.
    
    Note: Import DataPaths and create_data_paths from lmtools.io.metadata_tracking
    '''
    # Import the necessary classes
    from lmtools.io.metadata_tracking import create_data_paths, ProcessingStep
    
    # Create data paths with metadata
    data_paths = create_data_paths(
        base_dir="/path/to/data",
        base_name="Sample01",
        experiment_name="Immune Cell Analysis",
        sample_id="Mouse_01_Brain_Section_03",
        acquisition_date="2024-01-15",
        notes="60x oil immersion, Z-stack 50 slices"
    )
    
    # Load images and segmentations
    img_cy5, img_dapi, img_cd11b = data_paths.load_imgs()
    seg_cy5, seg_dapi, seg_qupath = data_paths.load_segs()
    
    # Process with overlap filter
    filtered_cy5 = filter_by_overlap(
        seg_cy5, 
        seg_dapi, 
        min_overlap_ratio=0.5,
        data_paths=data_paths,
        step_name="cy5_dapi_overlap_filter"
    )
    
    # Save processed mask
    data_paths.save_processed_mask(
        filtered_cy5, 
        "cy5_dapi_filtered",
        ProcessingStep(
            step_name="save_cy5_filtered",
            timestamp=datetime.now().isoformat(),
            parameters={"format": "numpy"},
            input_data=["cy5_dapi_overlap_filter"],
            notes="Saved CY5 cells with >50% DAPI overlap"
        )
    )
    
    # Compute intensities and filter
    avg_cd11b = compute_average_intensity(filtered_cy5, img_cd11b, use_donut=True)
    
    # Apply intensity filter
    final_mask = intensity_filter(
        filtered_cy5,
        avg_cd11b,
        upper_thresh=1000,
        lower_thresh=100,
        data_paths=data_paths,
        intensity_channel="cd11b",
        step_name="cd11b_intensity_filter"
    )
    
    # Save final result and metadata
    data_paths.save_processed_mask(final_mask, "final_immune_cells")
    metadata_path = data_paths.save_metadata()
    
    print(f"Processing complete. Metadata saved to: {metadata_path}")
    print(f"All paths: {data_paths.get_all_paths_dict()}")
    
    return data_paths