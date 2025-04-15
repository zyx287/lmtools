'''
author: zyx
date: 2025-04-14
last_modified: 2025-04-14
description: 
    Functions for downsampling microscopy images with different algorithms
'''
import os
import numpy as np
from typing import Union, Tuple, Optional, Literal
from pathlib import Path
import logging
import tifffile
from PIL import Image

# For OpenCV-based methods
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

# For scikit-image based methods
try:
    from skimage import transform
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

# Set up logging
logger = logging.getLogger(__name__)

# Define downsampling method types
DownsamplingMethod = Literal[
    'nearest', 'bilinear', 'bicubic', 'lanczos', 
    'area', 'gaussian', 'median'
]

def downsample_image(
    image_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    scale_factor: Union[float, Tuple[float, float]] = 0.5,
    method: DownsamplingMethod = 'bicubic',
    preserve_range: bool = True,
    library: Literal['auto', 'pillow', 'opencv', 'skimage'] = 'auto'
) -> np.ndarray:
    """
    Downsample a TIFF image using various algorithms
    
    Parameters
    ----------
    image_path : str or Path
        Path to the input TIFF image
    output_path : str or Path, optional
        Path to save the downsampled image. If None, the image is not saved.
    scale_factor : float or tuple of float, optional
        Scale factor for downsampling. If a single float, the same factor is applied
        to both dimensions. If a tuple (y_scale, x_scale), different factors are 
        applied to each dimension. Default is 0.5 (half size).
    method : str, optional
        Downsampling method to use. Options:
        - 'nearest': Nearest neighbor interpolation (fastest, lowest quality)
        - 'bilinear': Bilinear interpolation (good balance)
        - 'bicubic': Bicubic interpolation (better quality, slower)
        - 'lanczos': Lanczos filter (high quality, slowest)
        - 'area': Area-based downsampling (good for significant downsampling)
        - 'gaussian': Gaussian blur followed by subsampling
        - 'median': Median filter followed by subsampling
    preserve_range : bool, optional
        Whether to preserve the intensity range of the input image. 
        If False, the output is scaled to the 0-1 range. Default is True.
    library : str, optional
        Which library to use for downsampling. Options:
        - 'auto': Automatically select the best library for the method
        - 'pillow': Use PIL/Pillow
        - 'opencv': Use OpenCV
        - 'skimage': Use scikit-image
        
    Returns
    -------
    numpy.ndarray
        The downsampled image as a NumPy array
        
    Raises
    ------
    ValueError
        If the method or library is not supported
    ImportError
        If the requested library is not available
    """
    # Convert paths to Path objects for easier handling
    image_path = Path(image_path)
    if output_path is not None:
        output_path = Path(output_path)
    
    # Validate input
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    if not image_path.suffix.lower() in ['.tif', '.tiff']:
        logger.warning(f"Input may not be a TIFF file: {image_path}")
    
    # Convert single scale factor to tuple
    if isinstance(scale_factor, (int, float)):
        scale_factor = (scale_factor, scale_factor)
    
    # Load the input image
    try:
        # Use tifffile to handle complex TIFF formats
        image = tifffile.imread(image_path)
        logger.info(f"Loaded image with shape {image.shape} and dtype {image.dtype}")
    except Exception as e:
        logger.error(f"Error loading image: {e}")
        raise
    
    # Check if image is 2D
    if len(image.shape) > 2 and image.shape[2] > 4:
        raise ValueError(f"Image has {image.shape[2]} channels. Only 2D grayscale or RGB/RGBA images are supported.")
    
    # Automatically choose the library if set to 'auto'
    if library == 'auto':
        if method in ['nearest', 'bilinear', 'bicubic', 'lanczos']:
            library = 'pillow'
        elif method in ['area'] and OPENCV_AVAILABLE:
            library = 'opencv'
        elif method in ['gaussian', 'median'] and SKIMAGE_AVAILABLE:
            library = 'skimage'
        else:
            library = 'pillow'  # Default to Pillow as it's the most widely available
    
    # Perform downsampling based on the selected library
    if library == 'pillow':
        # Map method names to PIL resampling filters
        method_map = {
            'nearest': Image.NEAREST,
            'bilinear': Image.BILINEAR,
            'bicubic': Image.BICUBIC,
            'lanczos': Image.LANCZOS
        }
        
        if method not in method_map:
            raise ValueError(f"Method '{method}' not supported with Pillow. Choose from: {', '.join(method_map.keys())}")
        
        # Convert to PIL Image
        pil_img = Image.fromarray(image)
        
        # Calculate new dimensions
        new_height = int(image.shape[0] * scale_factor[0])
        new_width = int(image.shape[1] * scale_factor[1])
        
        # Resize the image
        resized_img = pil_img.resize((new_width, new_height), method_map[method])
        
        # Convert back to numpy array
        downsampled = np.array(resized_img)
    
    elif library == 'opencv':
        if not OPENCV_AVAILABLE:
            raise ImportError("OpenCV (cv2) is not available. Install it with 'pip install opencv-python'")
        
        # Map method names to OpenCV interpolation methods
        method_map = {
            'nearest': cv2.INTER_NEAREST,
            'bilinear': cv2.INTER_LINEAR,
            'bicubic': cv2.INTER_CUBIC,
            'lanczos': cv2.INTER_LANCZOS4,
            'area': cv2.INTER_AREA
        }
        
        if method not in method_map:
            raise ValueError(f"Method '{method}' not supported with OpenCV. Choose from: {', '.join(method_map.keys())}")
        
        # Calculate new dimensions
        new_height = int(image.shape[0] * scale_factor[0])
        new_width = int(image.shape[1] * scale_factor[1])
        
        # Resize the image
        downsampled = cv2.resize(
            image, 
            (new_width, new_height), 
            interpolation=method_map[method]
        )
    
    elif library == 'skimage':
        if not SKIMAGE_AVAILABLE:
            raise ImportError("scikit-image is not available. Install it with 'pip install scikit-image'")
        
        # Calculate new dimensions
        new_height = int(image.shape[0] * scale_factor[0])
        new_width = int(image.shape[1] * scale_factor[1])
        
        # For special methods
        if method in ['gaussian', 'median']:
            # For significant downsampling, we use a two-step approach:
            # 1. Apply filter to reduce aliasing
            # 2. Subsample the image
            
            # Determine the filter size based on downsampling factor
            # Higher downsampling requires larger filter to prevent aliasing
            min_scale = min(scale_factor)
            filter_size = max(3, int(1 / min_scale))
            
            if method == 'gaussian':
                from skimage.filters import gaussian
                # Apply Gaussian blur to prevent aliasing
                sigma = (1 / min_scale) / 3  # Sigma based on downsampling factor
                filtered = gaussian(image, sigma=sigma, preserve_range=True)
            
            elif method == 'median':
                from skimage.filters import median
                # Apply median filter to prevent aliasing while preserving edges
                from skimage.morphology import disk
                selem = disk(filter_size // 2)
                filtered = median(image, selem=selem)
            
            # Subsample the filtered image
            if len(image.shape) == 2:  # Grayscale
                downsampled = filtered[::int(1/scale_factor[0]), ::int(1/scale_factor[1])]
            else:  # Color image
                downsampled = filtered[::int(1/scale_factor[0]), ::int(1/scale_factor[1]), :]
        else:
            # Map method names to skimage interpolation orders
            method_map = {
                'nearest': 0,
                'bilinear': 1,
                'bicubic': 3,
                'lanczos': 1  # Not directly supported, use bilinear instead
            }
            
            if method not in method_map:
                raise ValueError(f"Method '{method}' not supported with scikit-image. Choose from: {', '.join(method_map.keys())}")
            
            # Use transform.resize for standard methods
            downsampled = transform.resize(
                image,
                (new_height, new_width),
                order=method_map[method],
                preserve_range=preserve_range,
                anti_aliasing=(method != 'nearest')
            )
    
    else:
        raise ValueError(f"Library '{library}' not supported. Choose from: 'auto', 'pillow', 'opencv', 'skimage'")
    
    # Convert data type to match input if preserve_range is True
    if preserve_range and downsampled.dtype != image.dtype:
        if np.issubdtype(image.dtype, np.integer):
            downsampled = np.clip(downsampled, 0, np.iinfo(image.dtype).max)
            downsampled = downsampled.astype(image.dtype)
        else:
            downsampled = downsampled.astype(image.dtype)
    
    # Save the downsampled image if output_path is provided
    if output_path is not None:
        # Create directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the image
        try:
            tifffile.imwrite(output_path, downsampled)
            logger.info(f"Saved downsampled image to {output_path}")
        except Exception as e:
            logger.error(f"Error saving downsampled image: {e}")
            raise
    
    return downsampled


def batch_downsample(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    scale_factor: Union[float, Tuple[float, float]] = 0.5,
    method: DownsamplingMethod = 'bicubic',
    file_pattern: str = "*.tif*",
    preserve_range: bool = True,
    library: str = 'auto',
    recursive: bool = False
) -> int:
    """
    Batch downsample multiple TIFF images in a directory
    
    Parameters
    ----------
    input_dir : str or Path
        Directory containing input TIFF images
    output_dir : str or Path
        Directory to save downsampled images
    scale_factor : float or tuple of float, optional
        Scale factor for downsampling, default 0.5
    method : str, optional
        Downsampling method, default 'bicubic'
    file_pattern : str, optional
        Glob pattern to match files, default "*.tif*"
    preserve_range : bool, optional
        Whether to preserve intensity range, default True
    library : str, optional
        Library to use for downsampling, default 'auto'
    recursive : bool, optional
        Whether to search subdirectories recursively, default False
        
    Returns
    -------
    int
        Number of images successfully processed
    """
    # Convert to Path objects
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of files to process
    if recursive:
        # Use rglob for recursive search
        files = list(input_dir.rglob(file_pattern))
    else:
        # Use glob for non-recursive search
        files = list(input_dir.glob(file_pattern))
    
    logger.info(f"Found {len(files)} files to process")
    
    # Process each file
    successful = 0
    for file_path in files:
        try:
            # Create relative path to maintain structure in output directory
            rel_path = file_path.relative_to(input_dir)
            out_path = output_dir / rel_path
            
            # Create parent directories if needed
            out_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Downsample image
            downsample_image(
                file_path,
                out_path,
                scale_factor=scale_factor,
                method=method,
                preserve_range=preserve_range,
                library=library
            )
            
            successful += 1
            logger.info(f"Processed {file_path.name} -> {out_path}")
        
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
    
    logger.info(f"Successfully processed {successful} out of {len(files)} files")
    return successful


if __name__ == "__main__":
    # Example usage when run as a script
    import argparse
    
    parser = argparse.ArgumentParser(description="Downsample TIFF images")
    parser.add_argument("input", type=str, help="Input TIFF image or directory")
    parser.add_argument("output", type=str, help="Output TIFF image or directory")
    parser.add_argument("--scale", type=float, default=0.5, help="Scale factor (default: 0.5)")
    parser.add_argument("--method", type=str, default="bicubic", 
                        choices=["nearest", "bilinear", "bicubic", "lanczos", "area", "gaussian", "median"],
                        help="Downsampling method (default: bicubic)")
    parser.add_argument("--library", type=str, default="auto",
                        choices=["auto", "pillow", "opencv", "skimage"],
                        help="Library to use for downsampling (default: auto)")
    parser.add_argument("--no-preserve-range", action="store_false", dest="preserve_range",
                        help="Do not preserve intensity range")
    parser.add_argument("--recursive", action="store_true",
                        help="Process subdirectories recursively")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Check if input is a file or directory
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if input_path.is_file():
        # Process single file
        try:
            downsample_image(
                input_path,
                output_path,
                scale_factor=args.scale,
                method=args.method,
                preserve_range=args.preserve_range,
                library=args.library
            )
            print(f"Image downsampled successfully: {output_path}")
        except Exception as e:
            print(f"Error: {e}")
    
    elif input_path.is_dir():
        # Process directory
        try:
            num_processed = batch_downsample(
                input_path,
                output_path,
                scale_factor=args.scale,
                method=args.method,
                preserve_range=args.preserve_range,
                library=args.library,
                recursive=args.recursive
            )
            print(f"Processed {num_processed} images")
        except Exception as e:
            print(f"Error: {e}")
    
    else:
        print(f"Error: Input path {input_path} does not exist")