'''
author: zyx
date: 2025-04-16
last_modified: 2025-04-16
description: 
    Functions for filtering segmentation objects based on intensity measurements
'''
import os
import numpy as np
from typing import Union, Tuple, Optional, Dict, Any, List, Literal
from pathlib import Path
import logging
import tifffile
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure, morphology, segmentation
import pandas as pd

# Set up logging
logger = logging.getLogger(__name__)

def intensity_filter(
    segmentation: Union[str, Path, np.ndarray],
    intensity_image: Union[str, Path, np.ndarray],
    output_path: Optional[Union[str, Path]] = None,
    threshold: Optional[float] = None,
    threshold_method: str = 'otsu',
    percentile: float = 25.0,
    region_type: str = 'whole',
    membrane_width: int = 2,
    plot_histogram: bool = True,
    figure_path: Optional[Union[str, Path]] = None,
    return_measurements: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, pd.DataFrame]]:
    """
    Filter segmented objects based on their intensity in a corresponding image.
    
    Parameters
    ----------
    segmentation : str, Path, or numpy.ndarray
        Segmentation mask with labeled objects (non-zero integers)
    intensity_image : str, Path, or numpy.ndarray
        Image to measure intensity from
    output_path : str or Path, optional
        Path to save the filtered segmentation
    threshold : float, optional
        Intensity threshold for filtering. Objects with mean intensity below this value
        will be removed. If None, threshold is determined automatically.
    threshold_method : str
        Method for automatic thresholding: 'otsu', 'percentile'
    percentile : float
        Percentile to use for thresholding when method is 'percentile'
    region_type : str
        Region within objects to consider for intensity calculation:
        - 'whole': Entire object
        - 'membrane': Membrane/border region only
        - 'inner': Inner region excluding border
        - 'outer': Outer region only
    membrane_width : int
        Width of membrane/border in pixels (for membrane region_type)
    plot_histogram : bool
        Whether to generate a histogram of object intensities
    figure_path : str or Path, optional
        Path to save the histogram figure
    return_measurements : bool
        Whether to return intensity measurements as a DataFrame
        
    Returns
    -------
    numpy.ndarray or Tuple[numpy.ndarray, pandas.DataFrame]
        Filtered segmentation mask, and optionally a DataFrame with measurements
    """
    # Load segmentation if path is provided
    if isinstance(segmentation, (str, Path)):
        seg_path = Path(segmentation)
        if not seg_path.exists():
            raise FileNotFoundError(f"Segmentation file not found: {seg_path}")
        try:
            seg_data = tifffile.imread(seg_path)
            logger.info(f"Loaded segmentation from {seg_path}")
        except Exception as e:
            logger.error(f"Error loading segmentation: {e}")
            raise
    else:
        seg_data = segmentation
    
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
    
    # Verify inputs
    if seg_data.shape[:2] != intensity_data.shape[:2]:
        raise ValueError(f"Segmentation shape {seg_data.shape} does not match intensity image shape {intensity_data.shape}")
    
    # Convert intensity image to float and handle multi-channel images
    if len(intensity_data.shape) > 2:
        if intensity_data.shape[2] > 1:
            logger.info(f"Using first channel of {intensity_data.shape[2]}-channel intensity image")
            intensity_data = intensity_data[:, :, 0]
    
    # Handle different bit depths
    if intensity_data.dtype == np.uint16:
        intensity_float = intensity_data.astype(np.float32) / 65535.0
    elif intensity_data.dtype == np.uint8:
        intensity_float = intensity_data.astype(np.float32) / 255.0
    else:
        intensity_float = intensity_data.astype(np.float32)
    
    # Get unique labels (excluding background which is 0)
    labels = np.unique(seg_data)
    labels = labels[labels > 0]
    
    # Create region masks based on region_type
    region_masks = {}
    
    for label in labels:
        # Create binary mask for this object
        obj_mask = (seg_data == label)
        
        if region_type == 'whole':
            # Use the whole object
            region_masks[label] = obj_mask
            
        elif region_type == 'membrane':
            # Create a membrane/border region
            eroded = morphology.erosion(obj_mask, morphology.disk(membrane_width))
            membrane = obj_mask & ~eroded
            region_masks[label] = membrane
            
        elif region_type == 'inner':
            # Create an inner region
            eroded = morphology.erosion(obj_mask, morphology.disk(membrane_width))
            region_masks[label] = eroded
            
        elif region_type == 'outer':
            # Create an outer region including the object and its boundary
            dilated = morphology.dilation(obj_mask, morphology.disk(membrane_width))
            outer = dilated & ~obj_mask
            region_masks[label] = outer
        
        else:
            raise ValueError(f"Unknown region type: {region_type}")
    
    # Calculate mean intensity for each object
    measurements = {}
    
    for label in labels:
        mask = region_masks[label]
        pixels = intensity_float[mask]
        
        if len(pixels) == 0:
            mean_intensity = 0
        else:
            mean_intensity = np.mean(pixels)
        
        measurements[label] = {
            'label': int(label),
            'mean_intensity': float(mean_intensity),
            'area': int(np.sum(mask)),
            'region_type': region_type
        }
    
    # Create DataFrame from measurements
    df = pd.DataFrame.from_dict(measurements, orient='index')
    
    # Determine threshold if not provided
    if threshold is None:
        if threshold_method == 'otsu':
            try:
                from skimage.filters import threshold_otsu
                intensities = df['mean_intensity'].values
                threshold = threshold_otsu(intensities)
                logger.info(f"Calculated Otsu threshold: {threshold}")
            except Exception as e:
                # Fallback to percentile method
                logger.warning(f"Failed to calculate Otsu threshold: {e}")
                threshold = np.percentile(df['mean_intensity'].values, percentile)
                logger.info(f"Using {percentile}th percentile threshold: {threshold}")
        
        elif threshold_method == 'percentile':
            threshold = np.percentile(df['mean_intensity'].values, percentile)
            logger.info(f"Using {percentile}th percentile threshold: {threshold}")
        
        else:
            raise ValueError(f"Unknown threshold method: {threshold_method}")
    
    # Apply threshold to filter objects
    filtered_labels = df[df['mean_intensity'] >= threshold]['label'].values
    filtered_mask = np.zeros_like(seg_data)
    
    for label in filtered_labels:
        filtered_mask[seg_data == label] = label
    
    logger.info(f"Filtered from {len(labels)} to {len(filtered_labels)} objects (threshold: {threshold})")
    
    # Plot histogram if requested
    if plot_histogram:
        plt.figure(figsize=(10, 6))
        plt.hist(df['mean_intensity'].values, bins=30, alpha=0.7)
        plt.axvline(x=threshold, color='r', linestyle='--', 
                    label=f'Threshold: {threshold:.4f}')
        plt.xlabel('Mean Intensity')
        plt.ylabel('Number of Objects')
        plt.title(f'Object Intensity Distribution ({region_type} region)')
        plt.legend()
        plt.grid(alpha=0.3)
        
        if figure_path is not None:
            fig_path = Path(figure_path)
            fig_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved histogram to {fig_path}")
        
        plt.show()
    
    # Save output if path is provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        tifffile.imwrite(output_path, filtered_mask)
        logger.info(f"Saved filtered segmentation to {output_path}")
    
    # Return results
    if return_measurements:
        return filtered_mask, df
    else:
        return filtered_mask


def visualize_intensity_regions(
    segmentation: Union[str, Path, np.ndarray],
    intensity_image: Union[str, Path, np.ndarray],
    output_path: Optional[Union[str, Path]] = None,
    label_id: Optional[int] = None,
    membrane_width: int = 2,
    show_all_regions: bool = True
) -> np.ndarray:
    """
    Visualize the different regions used for intensity calculations.
    
    Parameters
    ----------
    segmentation : str, Path, or numpy.ndarray
        Segmentation mask with labeled objects
    intensity_image : str, Path, or numpy.ndarray
        Intensity image
    output_path : str or Path, optional
        Path to save the visualization
    label_id : int, optional
        Specific label to visualize. If None, a random label will be chosen.
    membrane_width : int
        Width of membrane/border in pixels
    show_all_regions : bool
        Whether to show all region types (whole, membrane, inner, outer)
        
    Returns
    -------
    numpy.ndarray
        Visualization image
    """
    # Load segmentation if path is provided
    if isinstance(segmentation, (str, Path)):
        seg_path = Path(segmentation)
        if not seg_path.exists():
            raise FileNotFoundError(f"Segmentation file not found: {seg_path}")
        try:
            seg_data = tifffile.imread(seg_path)
            logger.info(f"Loaded segmentation from {seg_path}")
        except Exception as e:
            logger.error(f"Error loading segmentation: {e}")
            raise
    else:
        seg_data = segmentation
    
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
        if intensity_data.shape[2] > 1:
            logger.info(f"Using first channel of {intensity_data.shape[2]}-channel intensity image")
            intensity_data = intensity_data[:, :, 0]
    
    # Normalize intensity for visualization
    if intensity_data.dtype == np.uint16:
        intensity_vis = intensity_data.astype(np.float32) / 65535.0
    elif intensity_data.dtype == np.uint8:
        intensity_vis = intensity_data.astype(np.float32) / 255.0
    else:
        intensity_vis = intensity_data.astype(np.float32)
        intensity_vis = (intensity_vis - intensity_vis.min()) / (intensity_vis.max() - intensity_vis.min() + 1e-8)
    
    # Convert to 3-channel RGB
    intensity_rgb = np.stack([intensity_vis] * 3, axis=-1)
    
    # Select a label to visualize
    if label_id is None:
        labels = np.unique(seg_data)
        labels = labels[labels > 0]
        if len(labels) == 0:
            raise ValueError("No objects found in segmentation mask")
        label_id = labels[np.random.randint(0, len(labels))]
    
    # Check if label exists
    if np.sum(seg_data == label_id) == 0:
        raise ValueError(f"Label {label_id} not found in segmentation mask")
    
    logger.info(f"Visualizing regions for label {label_id}")
    
    # Create binary mask for this object
    obj_mask = (seg_data == label_id)
    
    # Create region masks
    if show_all_regions:
        # Prepare a figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        axs = axs.flatten()
        
        # Original object
        eroded = morphology.erosion(obj_mask, morphology.disk(membrane_width))
        dilated = morphology.dilation(obj_mask, morphology.disk(membrane_width))
        membrane = obj_mask & ~eroded
        outer = dilated & ~obj_mask
        
        # Show whole object
        overlay = intensity_rgb.copy()
        overlay[obj_mask, 0] = 1.0  # Red channel
        axs[0].imshow(overlay)
        axs[0].set_title('Whole Object')
        axs[0].axis('off')
        
        # Show membrane region
        overlay = intensity_rgb.copy()
        overlay[membrane, 1] = 1.0  # Green channel
        axs[1].imshow(overlay)
        axs[1].set_title('Membrane Region')
        axs[1].axis('off')
        
        # Show inner region
        overlay = intensity_rgb.copy()
        overlay[eroded, 2] = 1.0  # Blue channel
        axs[2].imshow(overlay)
        axs[2].set_title('Inner Region')
        axs[2].axis('off')
        
        # Show outer region
        overlay = intensity_rgb.copy()
        overlay[outer, 0] = 1.0  # Red channel
        overlay[outer, 1] = 1.0  # Green channel
        axs[3].imshow(overlay)
        axs[3].set_title('Outer Region')
        axs[3].axis('off')
        
        plt.tight_layout()
        
        # Save the figure if output path is provided
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved region visualization to {output_path}")
        
        plt.show()
        
        # Return a combined visualization
        vis_img = np.zeros((*intensity_rgb.shape[:2], 3), dtype=np.float32)
        vis_img[obj_mask] = [1.0, 0.7, 0.7]  # Light red for whole object
        vis_img[membrane] = [0.0, 1.0, 0.0]  # Green for membrane
        vis_img[eroded] = [0.7, 0.7, 1.0]    # Light blue for inner region
        vis_img[outer] = [1.0, 1.0, 0.0]     # Yellow for outer region
        
        # Overlay on grayscale intensity
        alpha = 0.7
        vis_img = alpha * vis_img + (1 - alpha) * np.stack([intensity_vis] * 3, axis=-1)
        
    else:
        # Just show one overlay with all regions
        vis_img = intensity_rgb.copy()
        
        # Create region masks
        eroded = morphology.erosion(obj_mask, morphology.disk(membrane_width))
        dilated = morphology.dilation(obj_mask, morphology.disk(membrane_width))
        membrane = obj_mask & ~eroded
        outer = dilated & ~obj_mask
        
        # Color the regions
        vis_img[eroded, 2] = 1.0          # Blue for inner region
        vis_img[membrane, 1] = 1.0        # Green for membrane
        vis_img[outer, 0] = 1.0           # Red for outer region
        
        # Display the image
        plt.figure(figsize=(8, 8))
        plt.imshow(vis_img)
        plt.title(f'Regions for Object {label_id}')
        plt.axis('off')
        
        # Save the figure if output path is provided
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved region visualization to {output_path}")
        
        plt.show()
    
    return vis_img


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Filter segmentation objects based on intensity")
    parser.add_argument("segmentation", type=str, help="Path to segmentation mask")
    parser.add_argument("intensity", type=str, help="Path to intensity image")
    parser.add_argument("output", type=str, help="Path to save filtered segmentation")
    parser.add_argument("--threshold", type=float, help="Intensity threshold (if not specified, calculated automatically)")
    parser.add_argument("--threshold-method", type=str, default="otsu", choices=["otsu", "percentile"], 
                       help="Method for automatic threshold calculation")
    parser.add_argument("--percentile", type=float, default=25.0, 
                       help="Percentile for threshold calculation (used with percentile method)")
    parser.add_argument("--region", type=str, default="whole", choices=["whole", "membrane", "inner", "outer"],
                       help="Region to consider for intensity calculation")
    parser.add_argument("--membrane-width", type=int, default=2,
                       help="Width of membrane/border in pixels")
    parser.add_argument("--no-histogram", action="store_false", dest="histogram",
                       help="Don't generate histogram")
    parser.add_argument("--figure", type=str, help="Path to save histogram figure")
    parser.add_argument("--visualize-regions", action="store_true",
                       help="Visualize the different regions")
    parser.add_argument("--label", type=int, help="Label ID to visualize (for --visualize-regions)")
    parser.add_argument("--vis-output", type=str, help="Path to save region visualization")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Visualize regions if requested
    if args.visualize_regions:
        visualize_intensity_regions(
            args.segmentation,
            args.intensity,
            output_path=args.vis_output,
            label_id=args.label,
            membrane_width=args.membrane_width
        )
    
    # Filter segmentation based on intensity
    intensity_filter(
        args.segmentation,
        args.intensity,
        output_path=args.output,
        threshold=args.threshold,
        threshold_method=args.threshold_method,
        percentile=args.percentile,
        region_type=args.region,
        membrane_width=args.membrane_width,
        plot_histogram=args.histogram,
        figure_path=args.figure
    )