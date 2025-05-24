'''
author: zyx
date: 2025-05-09
last_modified: 2025-05-09
description: 
    Functions for cell-specific filtering in segmentation masks based on multiple channel data
'''
import os
import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, Dict, Any, List, Literal
from pathlib import Path
import logging
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from scipy.ndimage import find_objects
from sklearn.mixture import GaussianMixture
from scipy import stats
import tifffile

# Set up logging
logger = logging.getLogger(__name__)

def overlap_filter(
    segmentation_mask: Union[str, Path, np.ndarray],
    reference_mask: Union[str, Path, np.ndarray],
    output_path: Optional[Union[str, Path]] = None,
    overlap_threshold: float = 0.2,
    relabel: bool = False
) -> np.ndarray:
    """
    Filter segmentation mask based on overlap with a reference mask.
    
    Parameters
    ----------
    segmentation_mask : str, Path, or numpy.ndarray
        Segmentation mask with labeled objects to be filtered
    reference_mask : str, Path, or numpy.ndarray
        Reference mask to check for overlap (e.g., DAPI mask)
    output_path : str or Path, optional
        Path to save the filtered segmentation mask
    overlap_threshold : float
        Minimum fraction of overlap required to keep an object (0-1)
    relabel : bool
        Whether to relabel objects in sequence after filtering
        
    Returns
    -------
    numpy.ndarray
        Filtered segmentation mask
    """
    # Load segmentation mask if path is provided
    if isinstance(segmentation_mask, (str, Path)):
        seg_path = Path(segmentation_mask)
        if not seg_path.exists():
            raise FileNotFoundError(f"Segmentation mask not found: {seg_path}")
        try:
            if seg_path.suffix.lower() in ['.tif', '.tiff']:
                seg_data = tifffile.imread(seg_path)
            else:
                seg_data = np.load(seg_path, allow_pickle=True)
            logger.info(f"Loaded segmentation mask from {seg_path}")
        except Exception as e:
            logger.error(f"Error loading segmentation mask: {e}")
            raise
    else:
        seg_data = segmentation_mask
    
    # Load reference mask if path is provided
    if isinstance(reference_mask, (str, Path)):
        ref_path = Path(reference_mask)
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference mask not found: {ref_path}")
        try:
            if ref_path.suffix.lower() in ['.tif', '.tiff']:
                ref_data = tifffile.imread(ref_path)
            else:
                ref_data = np.load(ref_path, allow_pickle=True)
            logger.info(f"Loaded reference mask from {ref_path}")
        except Exception as e:
            logger.error(f"Error loading reference mask: {e}")
            raise
    else:
        ref_data = reference_mask
    
    # Verify inputs have the same shape
    if seg_data.shape != ref_data.shape:
        raise ValueError(f"Segmentation mask shape {seg_data.shape} does not match reference mask shape {ref_data.shape}")

    # Create a copy for filtering
    filtered_mask = seg_data.copy()
    
    # Get unique labels and their objects
    labels = np.unique(seg_data)
    labels = labels[labels > 0]
    
    # Make sure the labels are sequential to avoid issues with find_objects
    label_map = {old_label: i+1 for i, old_label in enumerate(labels)}
    inverse_map = {i+1: old_label for i, old_label in enumerate(labels)}
    
    # Create a sequential mask for processing
    sequential_mask = np.zeros_like(seg_data)
    for old_label, new_label in label_map.items():
        sequential_mask[seg_data == old_label] = new_label
    
    # Find objects in the sequential mask
    objects = find_objects(sequential_mask)
    
    # Track which labels to keep
    labels_to_keep = []
    
    # For each object, check overlap with reference mask
    for i, obj_slice in enumerate(objects):
        if obj_slice is None:
            continue
        
        # Get the current label
        current_label = i + 1  # Labels start at 1
        
        # Create a mask for this object
        region = sequential_mask[obj_slice]
        mask = region == current_label
        
        # Count total pixels in the mask
        total_mask_pixels = np.sum(mask)
        
        if total_mask_pixels == 0:
            continue
        
        # Get the reference region corresponding to the current slice
        ref_region = ref_data[obj_slice]
        
        # Count pixels that overlap with any reference signal
        overlap_pixels = np.sum((ref_region > 0) & mask)
        
        # If overlap is sufficient, keep the object
        if overlap_pixels / total_mask_pixels >= overlap_threshold:
            labels_to_keep.append(inverse_map[current_label])
        else:
            # Remove the object from the filtered mask
            filtered_mask[seg_data == inverse_map[current_label]] = 0
    
    # Relabel objects if requested
    if relabel and labels_to_keep:
        from scipy import ndimage
        # Create a binary mask of objects to keep
        keep_mask = np.isin(seg_data, labels_to_keep)
        # Label connected components
        labeled_mask, num_features = ndimage.label(keep_mask)
        filtered_mask = labeled_mask
    
    # Save output if path is provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.save(output_path, filtered_mask)
        logger.info(f"Saved filtered mask to {output_path}")
    
    return filtered_mask

def intensity_channel_filter(
    segmentation_mask: Union[str, Path, np.ndarray],
    intensity_image: Union[str, Path, np.ndarray],
    output_path: Optional[Union[str, Path]] = None,
    threshold: Optional[float] = None,
    threshold_method: str = 'otsu',
    min_intensity: Optional[float] = None,
    gmm_components: Optional[int] = None,
    plot_histogram: bool = True,
    figure_path: Optional[Union[str, Path]] = None,
    relabel: bool = False
) -> np.ndarray:
    """
    Filter segmentation mask based on average intensity of objects in another channel.
    
    Parameters
    ----------
    segmentation_mask : str, Path, or numpy.ndarray
        Segmentation mask with labeled objects to be filtered
    intensity_image : str, Path, or numpy.ndarray
        Intensity image to measure (e.g., CY3 channel)
    output_path : str or Path, optional
        Path to save the filtered segmentation mask
    threshold : float, optional
        Intensity threshold for filtering. If None, threshold is determined automatically.
    threshold_method : str
        Method for automatic thresholding: 'otsu', 'gmm'
    min_intensity : float, optional
        Minimum intensity required (useful for removing background objects)
    gmm_components : int, optional
        Number of components for Gaussian Mixture Model. If None, determined automatically.
    plot_histogram : bool
        Whether to generate a histogram of object intensities
    figure_path : str or Path, optional
        Path to save the histogram figure
    relabel : bool
        Whether to relabel objects in sequence after filtering
        
    Returns
    -------
    numpy.ndarray
        Filtered segmentation mask
    """
    # Load segmentation mask if path is provided
    if isinstance(segmentation_mask, (str, Path)):
        seg_path = Path(segmentation_mask)
        if not seg_path.exists():
            raise FileNotFoundError(f"Segmentation mask not found: {seg_path}")
        try:
            if seg_path.suffix.lower() in ['.tif', '.tiff']:
                seg_data = tifffile.imread(seg_path)
            else:
                seg_data = np.load(seg_path, allow_pickle=True)
            logger.info(f"Loaded segmentation mask from {seg_path}")
        except Exception as e:
            logger.error(f"Error loading segmentation mask: {e}")
            raise
    else:
        seg_data = segmentation_mask
    
    # Load intensity image if path is provided
    if isinstance(intensity_image, (str, Path)):
        img_path = Path(intensity_image)
        if not img_path.exists():
            raise FileNotFoundError(f"Intensity image not found: {img_path}")
        try:
            intensity_data = tifffile.imread(img_path)
            logger.info(f"Loaded intensity image from {img_path}")
        except Exception as e:
            logger.error(f"Error loading intensity image: {e}")
            raise
    else:
        intensity_data = intensity_image
    
    # Handle multi-channel intensity image
    if len(intensity_data.shape) > 2:
        logger.info(f"Intensity data has multiple channels, extracting the first channel.")
        if intensity_data.shape[2] > 1:
            logger.info(f"Original shape: {intensity_data.shape}")
            intensity_data = intensity_data[:, :, 0]
    
    # Normalize intensity values to 0-1 range
    if intensity_data.dtype == np.uint16:
        intensity_float = intensity_data.astype(np.float32) / 65535.0
        logger.info("Converted uint16 to float32.")
    elif intensity_data.dtype == np.uint8:
        intensity_float = intensity_data.astype(np.float32) / 255.0
        logger.info("Converted uint8 to float32.")
    else:
        intensity_float = intensity_data.astype(np.float32)
    
    # Get unique labels and their objects
    labels = np.unique(seg_data)
    labels = labels[labels > 0]
    
    # Make sure the labels are sequential to avoid issues with find_objects
    label_map = {old_label: i+1 for i, old_label in enumerate(labels)}
    inverse_map = {i+1: old_label for i, old_label in enumerate(labels)}
    
    # Create a sequential mask for processing
    sequential_mask = np.zeros_like(seg_data)
    for old_label, new_label in label_map.items():
        sequential_mask[seg_data == old_label] = new_label
    
    # Find objects in the sequential mask
    objects = find_objects(sequential_mask)
    
    # Calculate average intensity for each object
    average_intensities = []
    valid_labels = []
    
    for i, obj_slice in enumerate(objects):
        if obj_slice is None:
            continue
        
        # Get the current label
        current_label = i + 1  # Labels start at 1
        original_label = inverse_map[current_label]
        
        # Get the intensity region and mask for this object
        region = intensity_float[obj_slice]
        mask = sequential_mask[obj_slice] == current_label
        
        # If mask is empty, skip this object
        if np.sum(mask) == 0:
            continue
        
        # Calculate average intensity for this object
        average_intensity = np.mean(region[mask])
        average_intensities.append(average_intensity)
        valid_labels.append(original_label)
    
    # Convert to numpy array
    average_intensities = np.array(average_intensities)
    valid_labels = np.array(valid_labels)
    
    # Determine threshold if not provided
    if threshold is None:
        if threshold_method == 'otsu':
            threshold = threshold_otsu(average_intensities)
            logger.info(f"Calculated Otsu threshold: {threshold}")
        
        elif threshold_method == 'gmm':
            # Determine optimal number of components if not specified
            if gmm_components is None:
                bic_scores = []
                components_range = range(1, 6)
                for n_components in components_range:
                    gmm = GaussianMixture(n_components=n_components, random_state=42)
                    gmm.fit(average_intensities.reshape(-1, 1))
                    bic_scores.append(gmm.bic(average_intensities.reshape(-1, 1)))
                
                # Choose optimal number of components (lowest BIC)
                optimal_components = components_range[np.argmin(bic_scores)]
                logger.info(f"Optimal number of components: {optimal_components}")
                gmm_components = optimal_components
            
            # Fit GMM with optimal number of components
            gmm = GaussianMixture(n_components=gmm_components, random_state=42)
            gmm.fit(average_intensities.reshape(-1, 1))
            
            # Find threshold from GMM
            if gmm_components >= 2:
                sorted_means = np.sort(gmm.means_.flatten())
                # Calculate potential thresholds between consecutive components
                thresholds = [(sorted_means[i] + sorted_means[i+1])/2 for i in range(len(sorted_means)-1)]
                # Use the first threshold (between lowest and second lowest means)
                threshold = thresholds[0]
                logger.info(f"Calculated GMM threshold: {threshold}")
            else:
                # Fallback to Otsu if only one component
                threshold = threshold_otsu(average_intensities)
                logger.info(f"Using Otsu threshold (only one GMM component): {threshold}")
        
        else:
            raise ValueError(f"Unknown threshold method: {threshold_method}")
    
    # Plot histogram if requested
    if plot_histogram:
        plt.figure(figsize=(12, 6))
        plt.hist(average_intensities, bins=100, color='blue', alpha=0.7)
        plt.axvline(x=threshold, color='red', linestyle='--', 
                   label=f'Threshold: {threshold:.6f}')
        
        if min_intensity is not None:
            plt.axvline(x=min_intensity, color='green', linestyle='--', 
                      label=f'Min threshold: {min_intensity:.6f}')
        
        plt.axvline(x=max(average_intensities), color='purple', linestyle='--', 
                   label=f'Max: {max(average_intensities):.6f}')
        plt.axvline(x=min(average_intensities), color='orange', linestyle='--', 
                   label=f'Min: {min(average_intensities):.6f}')
        
        # If GMM was used, plot the mixture components
        if threshold_method == 'gmm' and 'gmm' in locals():
            x = np.linspace(min(average_intensities), max(average_intensities), 1000).reshape(-1, 1)
            logprob = gmm.score_samples(x)
            pdf = np.exp(logprob)
            plt.plot(x, pdf * len(average_intensities) * (max(average_intensities) - min(average_intensities)) / 100, 
                    '-k', label='GMM PDF')
            
            # Plot individual components
            colors = ['red', 'blue', 'green', 'purple', 'orange'][:gmm_components]
            for i, (weight, mean, covar) in enumerate(zip(gmm.weights_, gmm.means_, gmm.covariances_)):
                pdf_individual = stats.norm.pdf(x.flatten(), mean.flatten()[0], np.sqrt(covar.flatten()[0]))
                plt.plot(x.flatten(), weight * pdf_individual * len(average_intensities) * 
                        (max(average_intensities) - min(average_intensities)) / 100, 
                        color=colors[i], 
                        label=f'Component {i+1}: μ={mean[0]:.6f}, σ={np.sqrt(covar.flatten()[0]):.6f}, w={weight:.3f}')
        
        plt.xlabel('Average Intensity')
        plt.ylabel('Frequency')
        plt.title('Histogram of Average Intensities')
        plt.legend()
        plt.grid(True)
        
        if figure_path is not None:
            fig_path = Path(figure_path)
            fig_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved histogram to {fig_path}")
        
        plt.show()
    
    # Filter objects based on intensity
    # Create a copy of segmentation mask for filtering
    filtered_mask = seg_data.copy()
    
    # Apply intensity filtering
    labels_to_keep = []
    for i, label in enumerate(valid_labels):
        # Skip if out of bounds
        if i >= len(average_intensities):
            continue
        
        keep_object = True
        
        # Apply threshold
        if average_intensities[i] > threshold:
            keep_object = False
        
        # Apply minimum intensity threshold if provided
        if min_intensity is not None and average_intensities[i] < min_intensity:
            keep_object = False
        
        # Remove object if needed
        if not keep_object:
            filtered_mask[seg_data == label] = 0
        else:
            labels_to_keep.append(label)
    
    # Relabel objects if requested
    if relabel and labels_to_keep:
        from scipy import ndimage
        # Create a binary mask of objects to keep
        keep_mask = np.isin(seg_data, labels_to_keep)
        # Label connected components
        labeled_mask, num_features = ndimage.label(keep_mask)
        filtered_mask = labeled_mask
    
    # Save output if path is provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.save(output_path, filtered_mask)
        logger.info(f"Saved filtered mask to {output_path}")
    
    return filtered_mask

def multi_channel_filter(
    segmentation_mask: Union[str, Path, np.ndarray],
    dapi_mask: Optional[Union[str, Path, np.ndarray]] = None,
    dapi_image: Optional[Union[str, Path, np.ndarray]] = None,
    channel_images: Optional[Dict[str, Union[str, Path, np.ndarray]]] = None,
    output_path: Optional[Union[str, Path]] = None,
    dapi_overlap_threshold: float = 0.2,
    channel_thresholds: Optional[Dict[str, float]] = None,
    channel_min_intensities: Optional[Dict[str, float]] = None,
    threshold_method: str = 'otsu',
    plot_histograms: bool = True,
    figure_dir: Optional[Union[str, Path]] = None,
    relabel_final: bool = True
) -> np.ndarray:
    """
    Multi-channel filtering pipeline for segmentation masks.
    
    Parameters
    ----------
    segmentation_mask : str, Path, or numpy.ndarray
        Segmentation mask with labeled objects to be filtered
    dapi_mask : str, Path, or numpy.ndarray, optional
        DAPI segmentation mask for overlap filtering
    dapi_image : str, Path, or numpy.ndarray, optional
        DAPI intensity image for intensity-based filtering
    channel_images : Dict[str, Union[str, Path, np.ndarray]], optional
        Dictionary of channel names to intensity images for filtering
    output_path : str or Path, optional
        Path to save the final filtered segmentation mask
    dapi_overlap_threshold : float
        Minimum fraction of overlap with DAPI required to keep an object
    channel_thresholds : Dict[str, float], optional
        Manual thresholds for each channel (if None, calculated automatically)
    channel_min_intensities : Dict[str, float], optional
        Minimum intensity thresholds for each channel
    threshold_method : str
        Method for automatic thresholding: 'otsu', 'gmm'
    plot_histograms : bool
        Whether to generate histograms for intensity distributions
    figure_dir : str or Path, optional
        Directory to save histogram figures
    relabel_final : bool
        Whether to relabel objects in the final mask
        
    Returns
    -------
    numpy.ndarray
        Final filtered segmentation mask
    """
    # Load segmentation mask if path is provided
    if isinstance(segmentation_mask, (str, Path)):
        seg_path = Path(segmentation_mask)
        if not seg_path.exists():
            raise FileNotFoundError(f"Segmentation mask not found: {seg_path}")
        try:
            if seg_path.suffix.lower() in ['.tif', '.tiff']:
                seg_data = tifffile.imread(seg_path)
            else:
                seg_data = np.load(seg_path, allow_pickle=True)
            logger.info(f"Loaded segmentation mask from {seg_path}")
        except Exception as e:
            logger.error(f"Error loading segmentation mask: {e}")
            raise
    else:
        seg_data = segmentation_mask
    
    # Create output directory if needed
    if figure_dir is not None:
        figure_dir = Path(figure_dir)
        figure_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize current mask with the original segmentation
    current_mask = seg_data.copy()
    
    # Step 1: DAPI overlap filtering
    if dapi_mask is not None:
        logger.info("Performing DAPI overlap filtering")
        figure_path = None
        if figure_dir is not None:
            figure_path = figure_dir / "dapi_overlap_filtering.png"
        
        current_mask = overlap_filter(
            segmentation_mask=current_mask,
            reference_mask=dapi_mask,
            overlap_threshold=dapi_overlap_threshold,
            relabel=False
        )
        
        logger.info(f"After DAPI overlap filtering: {len(np.unique(current_mask)) - 1} objects")
    
    # Step 2: DAPI intensity filtering
    if dapi_image is not None:
        logger.info("Performing DAPI intensity filtering")
        figure_path = None
        if figure_dir is not None and plot_histograms:
            figure_path = figure_dir / "dapi_intensity_filtering.png"
        
        dapi_threshold = None
        if channel_thresholds is not None and 'dapi' in channel_thresholds:
            dapi_threshold = channel_thresholds['dapi']
        
        dapi_min = None
        if channel_min_intensities is not None and 'dapi' in channel_min_intensities:
            dapi_min = channel_min_intensities['dapi']
        
        current_mask = intensity_channel_filter(
            segmentation_mask=current_mask,
            intensity_image=dapi_image,
            threshold=dapi_threshold,
            threshold_method=threshold_method,
            min_intensity=dapi_min,
            plot_histogram=plot_histograms,
            figure_path=figure_path,
            relabel=False
        )
        
        logger.info(f"After DAPI intensity filtering: {len(np.unique(current_mask)) - 1} objects")
    
    # Step 3: Additional channel filtering
    if channel_images is not None:
        for channel_name, channel_image in channel_images.items():
            logger.info(f"Performing {channel_name} intensity filtering")
            figure_path = None
            if figure_dir is not None and plot_histograms:
                figure_path = figure_dir / f"{channel_name}_intensity_filtering.png"
            
            channel_threshold = None
            if channel_thresholds is not None and channel_name in channel_thresholds:
                channel_threshold = channel_thresholds[channel_name]
            
            channel_min = None
            if channel_min_intensities is not None and channel_name in channel_min_intensities:
                channel_min = channel_min_intensities[channel_name]
            
            current_mask = intensity_channel_filter(
                segmentation_mask=current_mask,
                intensity_image=channel_image,
                threshold=channel_threshold,
                threshold_method=threshold_method,
                min_intensity=channel_min,
                plot_histogram=plot_histograms,
                figure_path=figure_path,
                relabel=False
            )
            
            logger.info(f"After {channel_name} filtering: {len(np.unique(current_mask)) - 1} objects")
    
    # Relabel objects in final mask if requested
    if relabel_final:
        from scipy import ndimage
        # Create a binary mask of non-zero pixels
        binary_mask = current_mask > 0
        # Label connected components
        labeled_mask, num_features = ndimage.label(binary_mask)
        logger.info(f"Relabeled mask: {num_features} objects")
        current_mask = labeled_mask
    
    # Save output if path is provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.save(output_path, current_mask)
        logger.info(f"Saved final filtered mask to {output_path}")
    
    return current_mask