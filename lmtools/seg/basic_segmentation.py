'''
author: zyx
date: 2025-04-05
last_modified: 2025-04-05
description: 
    Basic segmentation techniques for microscopy images without deep learning
'''
import os
import numpy as np
from typing import Union, Tuple, Optional, Dict, Any, List, Literal
from pathlib import Path
import logging
import tifffile
from scipy import ndimage
import skimage.filters as filters
import skimage.morphology as morphology
import skimage.segmentation as segmentation
import skimage.feature as feature
import skimage.measure as measure
from skimage.color import rgb2gray

# Set up logging
logger = logging.getLogger(__name__)

def threshold_segment(
    image: Union[str, Path, np.ndarray],
    output_path: Optional[Union[str, Path]] = None,
    method: str = 'otsu',
    threshold_value: Optional[float] = None,
    block_size: int = 35,
    offset: float = 0.0,
    remove_small_objects: bool = True,
    min_size: int = 50,
    fill_holes: bool = True,
    connectivity: int = 1,
    return_labels: bool = True
) -> np.ndarray:
    '''
    Segment image using thresholding techniques
    
    Parameters
    ----------
    image : str, Path, or numpy.ndarray
        Input image to segment (path or array)
    output_path : str or Path, optional
        Path to save the segmentation result
    method : str
        Thresholding method: 'simple', 'otsu', 'adaptive', 'local', 'yen', 'li', 'triangle'
    threshold_value : float, optional
        Manual threshold value (0-1) to use with 'simple' method
    block_size : int
        Size of local region for adaptive methods (must be odd)
    offset : float
        Offset from the local mean/median for adaptive methods
    remove_small_objects : bool
        Whether to remove small objects from the segmentation
    min_size : int
        Minimum size of objects to keep (in pixels)
    fill_holes : bool
        Whether to fill holes in the segmentation
    connectivity : int
        Connectivity for determining connected components (1 or 2)
    return_labels : bool
        Whether to return labeled objects (True) or binary mask (False)
        
    Returns
    -------
    numpy.ndarray
        Segmentation result (labeled objects or binary mask)
    '''
    # Load image if path is provided
    if isinstance(image, (str, Path)):
        image_path = Path(image)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        try:
            img = tifffile.imread(image_path)
            logger.info(f"Loaded image from {image_path}")
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            raise
    else:
        img = image
    
    # Convert to grayscale if needed
    if len(img.shape) > 2:
        if img.shape[2] == 3 or img.shape[2] == 4:  # RGB or RGBA
            img = rgb2gray(img)
            logger.info("Converted color image to grayscale")
        else:
            # Use first channel for multichannel images
            img = img[:, :, 0]
            logger.warning(f"Using only first channel of {img.shape[2]}-channel image")
    
    # Handle 16-bit images by scaling to 0-1
    if img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    elif img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    
    # Apply thresholding
    if method == 'simple':
        if threshold_value is None:
            raise ValueError("For 'simple' method, threshold_value must be provided")
        threshold = threshold_value
        logger.info(f"Using simple threshold: {threshold}")
    
    elif method == 'otsu':
        threshold = filters.threshold_otsu(img)
        logger.info(f"Computed Otsu threshold: {threshold}")
    
    elif method == 'yen':
        threshold = filters.threshold_yen(img)
        logger.info(f"Computed Yen threshold: {threshold}")
    
    elif method == 'li':
        threshold = filters.threshold_li(img)
        logger.info(f"Computed Li threshold: {threshold}")
    
    elif method == 'triangle':
        threshold = filters.threshold_triangle(img)
        logger.info(f"Computed Triangle threshold: {threshold}")
    
    elif method == 'adaptive':
        # Check that block_size is odd
        if block_size % 2 == 0:
            block_size += 1
            logger.warning(f"Adjusted block size to odd value: {block_size}")
        
        binary = filters.threshold_adaptive(
            img, 
            block_size=block_size,
            offset=offset
        )
        logger.info(f"Applied adaptive thresholding with block size: {block_size}, offset: {offset}")
    
    elif method == 'local':
        # Check that block_size is odd
        if block_size % 2 == 0:
            block_size += 1
            logger.warning(f"Adjusted block size to odd value: {block_size}")
        
        binary = filters.threshold_local(
            img,
            block_size=block_size,
            offset=offset
        ) > 0
        logger.info(f"Applied local thresholding with block size: {block_size}, offset: {offset}")
    
    else:
        raise ValueError(f"Unknown thresholding method: {method}")
    
    # Create binary mask for non-adaptive methods
    if method not in ['adaptive', 'local']:
        binary = img > threshold
    
    # Post-processing
    if fill_holes:
        binary = ndimage.binary_fill_holes(binary)
        logger.info("Filled holes in binary mask")
    
    if remove_small_objects and min_size > 0:
        binary = morphology.remove_small_objects(
            binary, 
            min_size=min_size, 
            connectivity=connectivity
        )
        logger.info(f"Removed small objects (min size: {min_size})")
    
    # Label objects if requested
    if return_labels:
        labeled, num_objects = ndimage.label(binary, structure=np.ones((3, 3)))
        logger.info(f"Found {num_objects} objects in segmentation")
        result = labeled
    else:
        result = binary.astype(np.uint8) * 255
    
    # Save output if path is provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        tifffile.imwrite(output_path, result)
        logger.info(f"Saved segmentation to {output_path}")
    
    return result


def watershed_segment(
    image: Union[str, Path, np.ndarray],
    output_path: Optional[Union[str, Path]] = None,
    threshold_method: str = 'otsu',
    threshold_value: Optional[float] = None,
    distance_transform: bool = True,
    distance_threshold: Optional[float] = None,
    min_distance: int = 10,
    watershed_connectivity: int = 1,
    remove_small_objects: bool = True,
    min_size: int = 50,
    fill_holes: bool = True
) -> np.ndarray:
    '''
    Segment image using the watershed algorithm
    
    Parameters
    ----------
    image : str, Path, or numpy.ndarray
        Input image to segment (path or array)
    output_path : str or Path, optional
        Path to save the segmentation result
    threshold_method : str
        Method for initial thresholding: 'simple', 'otsu', 'yen', 'li', 'triangle'
    threshold_value : float, optional
        Manual threshold value for 'simple' method
    distance_transform : bool
        Whether to use distance transform for watershed markers
    distance_threshold : float, optional
        Threshold for distance transform (fraction of max distance)
    min_distance : int
        Minimum distance between peaks (used if distance_transform is False)
    watershed_connectivity : int
        Connectivity for watershed algorithm (1 or 2)
    remove_small_objects : bool
        Whether to remove small objects from the segmentation
    min_size : int
        Minimum size of objects to keep (in pixels)
    fill_holes : bool
        Whether to fill holes in the segmentation
        
    Returns
    -------
    numpy.ndarray
        Segmentation result (labeled objects)
    '''
    # Load image if path is provided
    if isinstance(image, (str, Path)):
        image_path = Path(image)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        try:
            img = tifffile.imread(image_path)
            logger.info(f"Loaded image from {image_path}")
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            raise
    else:
        img = image
    
    # Convert to grayscale if needed
    if len(img.shape) > 2:
        if img.shape[2] == 3 or img.shape[2] == 4:  # RGB or RGBA
            img = rgb2gray(img)
            logger.info("Converted color image to grayscale")
        else:
            # Use first channel for multichannel images
            img = img[:, :, 0]
            logger.warning(f"Using only first channel of {img.shape[2]}-channel image")
    
    # Handle 16-bit images by scaling to 0-1
    if img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    elif img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    
    # Create initial binary mask using thresholding
    binary = threshold_segment(
        img,
        method=threshold_method,
        threshold_value=threshold_value,
        remove_small_objects=remove_small_objects,
        min_size=min_size,
        fill_holes=fill_holes,
        return_labels=False
    )
    
    # Binary should be boolean for watershed
    binary = binary > 0
    
    # Generate markers for watershed
    if distance_transform:
        # Use distance transform to generate markers
        distance = ndimage.distance_transform_edt(binary)
        
        # Determine threshold for peak detection
        if distance_threshold is None:
            # Use 70% of max distance as default
            dist_threshold = 0.7 * distance.max()
        else:
            dist_threshold = distance_threshold * distance.max()
        
        # Create markers from local maxima of distance transform
        local_max = feature.peak_local_max(
            distance,
            min_distance=min_distance,
            threshold_abs=dist_threshold,
            labels=binary
        )
        
        # Create marker array
        markers = np.zeros_like(distance, dtype=np.int32)
        markers[tuple(local_max.T)] = 1
        markers = measure.label(markers)
        
        logger.info(f"Created {markers.max()} watershed markers using distance transform")
    
    else:
        # Use intensity peaks as markers
        local_max = feature.peak_local_max(
            img,
            min_distance=min_distance,
            labels=binary
        )
        
        # Create marker array
        markers = np.zeros_like(img, dtype=np.int32)
        markers[tuple(local_max.T)] = 1
        markers = measure.label(markers)
        
        logger.info(f"Created {markers.max()} watershed markers using intensity peaks")
    
    # Apply watershed
    if markers.max() > 0:
        # Invert the image for watershed (watershed works on high values as boundaries)
        if distance_transform:
            # Use the negative distance transform
            elevation_map = -distance
        else:
            # Use the inverted image
            elevation_map = filters.rank.gradient(img.astype(np.uint8), morphology.disk(2))
        
        # Apply watershed
        labels = segmentation.watershed(
            elevation_map,
            markers,
            mask=binary,
            connectivity=watershed_connectivity
        )
        
        logger.info(f"Watershed segmentation found {labels.max()} objects")
    else:
        # If no markers were found, use the binary image with connected components
        labels, num_objects = ndimage.label(binary)
        logger.warning(f"No watershed markers found, using connected components instead: {num_objects} objects")
    
    # Save output if path is provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        tifffile.imwrite(output_path, labels)
        logger.info(f"Saved watershed segmentation to {output_path}")
    
    return labels


def region_growing_segment(
    image: Union[str, Path, np.ndarray],
    output_path: Optional[Union[str, Path]] = None,
    seed_method: str = 'intensity',
    num_seeds: int = 10,
    threshold_method: str = 'otsu',
    compactness: float = 0.001,
    mask: Optional[np.ndarray] = None,
    remove_small_objects: bool = True,
    min_size: int = 50
) -> np.ndarray:
    '''
    Segment image using region growing (SLIC superpixels or random walker)
    
    Parameters
    ----------
    image : str, Path, or numpy.ndarray
        Input image to segment (path or array)
    output_path : str or Path, optional
        Path to save the segmentation result
    seed_method : str
        Method to generate seeds: 'intensity', 'grid', 'random'
    num_seeds : int
        Number of seed points or regions
    threshold_method : str
        Method for initial thresholding (for intensity seed method)
    compactness : float
        Compactness parameter for SLIC algorithm (higher = more compact regions)
    mask : numpy.ndarray, optional
        Binary mask to restrict segmentation
    remove_small_objects : bool
        Whether to remove small objects from the segmentation
    min_size : int
        Minimum size of objects to keep (in pixels)
        
    Returns
    -------
    numpy.ndarray
        Segmentation result (labeled objects)
    '''
    # Load image if path is provided
    if isinstance(image, (str, Path)):
        image_path = Path(image)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        try:
            img = tifffile.imread(image_path)
            logger.info(f"Loaded image from {image_path}")
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            raise
    else:
        img = image
    
    # Convert to grayscale if needed
    if len(img.shape) > 2:
        if img.shape[2] == 3 or img.shape[2] == 4:  # RGB or RGBA
            img_gray = rgb2gray(img)
            logger.info("Converted color image to grayscale")
        else:
            # Use first channel for multichannel images
            img_gray = img[:, :, 0]
            logger.warning(f"Using only first channel of {img.shape[2]}-channel image")
    else:
        img_gray = img
    
    # Handle 16-bit images by scaling to 0-1
    if img_gray.dtype == np.uint16:
        img_float = img_gray.astype(np.float32) / 65535.0
    elif img_gray.dtype == np.uint8:
        img_float = img_gray.astype(np.float32) / 255.0
    else:
        img_float = img_gray.astype(np.float32)
    
    # Apply SLIC superpixel segmentation
    segments = segmentation.slic(
        img_float,
        n_segments=num_seeds,
        compactness=compactness,
        mask=mask,
        start_label=1
    )
    
    logger.info(f"Generated {segments.max()} superpixels using SLIC")
    
    # For intensity-based segmentation, merge regions based on intensity
    if seed_method == 'intensity':
        # Get mean intensity for each region
        region_means = np.zeros(segments.max() + 1)
        for i in range(1, segments.max() + 1):
            region_means[i] = np.mean(img_float[segments == i])
        
        # Determine threshold
        if threshold_method == 'otsu':
            threshold = filters.threshold_otsu(img_float)
        elif threshold_method == 'yen':
            threshold = filters.threshold_yen(img_float)
        elif threshold_method == 'li':
            threshold = filters.threshold_li(img_float)
        elif threshold_method == 'triangle':
            threshold = filters.threshold_triangle(img_float)
        else:
            raise ValueError(f"Unknown threshold method: {threshold_method}")
        
        logger.info(f"Using {threshold_method} threshold: {threshold}")
        
        # Create new segmentation by merging regions
        merged_segments = np.zeros_like(segments)
        for i in range(1, segments.max() + 1):
            if region_means[i] > threshold:
                merged_segments[segments == i] = 1
        
        # Label connected components
        labeled, num_objects = ndimage.label(merged_segments)
        logger.info(f"Merged superpixels into {num_objects} objects based on intensity")
        
        # Remove small objects if requested
        if remove_small_objects and min_size > 0:
            labeled = morphology.remove_small_objects(
                labeled, 
                min_size=min_size
            )
            # Re-label to ensure consecutive labels
            labeled, num_objects = ndimage.label(labeled > 0)
            logger.info(f"Removed small objects, {num_objects} objects remaining")
        
        result = labeled
    
    else:
        # Just use the SLIC segments directly
        result = segments
    
    # Save output if path is provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        tifffile.imwrite(output_path, result)
        logger.info(f"Saved region growing segmentation to {output_path}")
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Basic segmentation methods for microscopy images")
    subparsers = parser.add_subparsers(dest="command", help="Segmentation method")
    
    # Threshold segmentation
    threshold_parser = subparsers.add_parser("threshold", help="Threshold-based segmentation")
    threshold_parser.add_argument("input", type=str, help="Input image path")
    threshold_parser.add_argument("output", type=str, help="Output segmentation path")
    threshold_parser.add_argument("--method", type=str, default="otsu",
                                choices=["simple", "otsu", "adaptive", "local", "yen", "li", "triangle"],
                                help="Thresholding method")
    threshold_parser.add_argument("--threshold", type=float, help="Manual threshold value (0-1)")
    threshold_parser.add_argument("--block-size", type=int, default=35, 
                                help="Block size for adaptive methods")
    threshold_parser.add_argument("--offset", type=float, default=0.0,
                                help="Offset for adaptive methods")
    threshold_parser.add_argument("--no-remove-small", action="store_false", dest="remove_small",
                                help="Don't remove small objects")
    threshold_parser.add_argument("--min-size", type=int, default=50,
                                help="Minimum object size in pixels")
    threshold_parser.add_argument("--no-fill-holes", action="store_false", dest="fill_holes",
                                help="Don't fill holes")
    threshold_parser.add_argument("--connectivity", type=int, default=1, choices=[1, 2],
                                help="Connectivity for determining connected components")
    threshold_parser.add_argument("--binary", action="store_false", dest="return_labels",
                                help="Return binary mask instead of labeled objects")
    
    # Watershed segmentation
    watershed_parser = subparsers.add_parser("watershed", help="Watershed segmentation")
    watershed_parser.add_argument("input", type=str, help="Input image path")
    watershed_parser.add_argument("output", type=str, help="Output segmentation path")
    watershed_parser.add_argument("--threshold-method", type=str, default="otsu",
                                choices=["simple", "otsu", "yen", "li", "triangle"],
                                help="Method for initial thresholding")
    watershed_parser.add_argument("--threshold", type=float, help="Manual threshold value (0-1)")
    watershed_parser.add_argument("--no-distance-transform", action="store_false", dest="distance_transform",
                                help="Don't use distance transform for markers")
    watershed_parser.add_argument("--distance-threshold", type=float,
                                help="Threshold for distance transform (fraction of max)")
    watershed_parser.add_argument("--min-distance", type=int, default=10,
                                help="Minimum distance between peaks")
    watershed_parser.add_argument("--connectivity", type=int, default=1, choices=[1, 2],
                                help="Connectivity for watershed")
    watershed_parser.add_argument("--no-remove-small", action="store_false", dest="remove_small",
                                help="Don't remove small objects")
    watershed_parser.add_argument("--min-size", type=int, default=50,
                                help="Minimum object size in pixels")
    watershed_parser.add_argument("--no-fill-holes", action="store_false", dest="fill_holes",
                                help="Don't fill holes")
    
    # Region growing segmentation
    region_parser = subparsers.add_parser("region", help="Region growing segmentation")
    region_parser.add_argument("input", type=str, help="Input image path")
    region_parser.add_argument("output", type=str, help="Output segmentation path")
    region_parser.add_argument("--seed-method", type=str, default="intensity",
                             choices=["intensity", "grid", "random"],
                             help="Method to generate seeds")
    region_parser.add_argument("--num-seeds", type=int, default=100,
                             help="Number of seed points or regions")
    region_parser.add_argument("--threshold-method", type=str, default="otsu",
                             choices=["otsu", "yen", "li", "triangle"],
                             help="Method for intensity thresholding")
    region_parser.add_argument("--compactness", type=float, default=0.001,
                             help="Compactness parameter for SLIC")
    region_parser.add_argument("--no-remove-small", action="store_false", dest="remove_small",
                             help="Don't remove small objects")
    region_parser.add_argument("--min-size", type=int, default=50,
                             help="Minimum object size in pixels")
    
    # Common options
    for p in [threshold_parser, watershed_parser, region_parser]:
        p.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if args.command == "threshold":
        threshold_segment(
            args.input,
            args.output,
            method=args.method,
            threshold_value=args.threshold,
            block_size=args.block_size,
            offset=args.offset,
            remove_small_objects=args.remove_small,
            min_size=args.min_size,
            fill_holes=args.fill_holes,
            connectivity=args.connectivity,
            return_labels=args.return_labels
        )
    
    elif args.command == "watershed":
        watershed_segment(
            args.input,
            args.output,
            threshold_method=args.threshold_method,
            threshold_value=args.threshold,
            distance_transform=args.distance_transform,
            distance_threshold=args.distance_threshold,
            min_distance=args.min_distance,
            watershed_connectivity=args.connectivity,
            remove_small_objects=args.remove_small,
            min_size=args.min_size,
            fill_holes=args.fill_holes
        )
    
    elif args.command == "region":
        region_growing_segment(
            args.input,
            args.output,
            seed_method=args.seed_method,
            num_seeds=args.num_seeds,
            threshold_method=args.threshold_method,
            compactness=args.compactness,
            remove_small_objects=args.remove_small,
            min_size=args.min_size
        )
    
    else:
        parser.print_help()