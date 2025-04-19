'''
author: zyx
date: 2025-04-19
last_modified: 2025-04-19
description: 
    Functions for handling and transforming unusual microscopy data formats
'''
import os
import numpy as np
from typing import Union, List, Optional, Dict, Tuple, Literal
from pathlib import Path
import logging
import tifffile
import json

# Set up logging
logger = logging.getLogger(__name__)

def transform_and_split(
    image_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    channel_names: Optional[List[str]] = None,
    channel_axis: Optional[int] = None,
    suffix_separator: str = "_",
    transpose_axes: Optional[List[int]] = None,
    normalize: bool = False,
    save_metadata: bool = True
) -> Dict[str, Path]:
    """
    Transform and split a multi-dimensional image into separate channel images.
    Handles unusual data layouts where channels might not be in the usual RGB position.
    
    Parameters
    ----------
    image_path : str or Path
        Path to the input multi-channel image
    output_dir : str or Path, optional
        Directory to save the split channel images. If None, will use the same directory as the input.
    channel_names : List[str], optional
        List of names for the channels. If None, will use ["C1", "C2", ...].
        Must match the number of channels in the image.
    channel_axis : int, optional
        Index of the dimension that represents channels. If None, will attempt to detect automatically.
    suffix_separator : str, optional
        Separator to use between the base filename and the channel name, default is "_"
    transpose_axes : List[int], optional
        List of axes indices to reorder dimensions before splitting channels.
        For example, [2, 0, 1] would transform (z, y, x) to (x, z, y).
    normalize : bool, optional
        Whether to normalize each channel to 0-1 range, default is False
    save_metadata : bool, optional
        Whether to save metadata about the original image shape and transformation
        
    Returns
    -------
    Dict[str, Path]
        Dictionary mapping channel names to output file paths
    
    Raises
    ------
    ValueError
        If channel detection fails or if channel_names length doesn't match
    FileNotFoundError
        If the input file doesn't exist
    """
    # Convert to Path
    image_path = Path(image_path)
    
    # Check if file exists
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Determine output directory
    if output_dir is None:
        output_dir = image_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the image
    try:
        image = tifffile.imread(image_path)
        logger.info(f"Loaded image with shape {image.shape} and dtype {image.dtype}")
    except Exception as e:
        logger.error(f"Error loading image: {e}")
        raise
    
    # Store original shape for metadata
    original_shape = image.shape
    
    # Transpose axes if specified
    if transpose_axes is not None:
        if len(transpose_axes) != len(image.shape):
            raise ValueError(f"transpose_axes length ({len(transpose_axes)}) must match image dimensions ({len(image.shape)})")
        
        try:
            image = np.transpose(image, transpose_axes)
            logger.info(f"Transposed image to shape {image.shape}")
        except Exception as e:
            logger.error(f"Error transposing image: {e}")
            raise
    
    # Auto-detect channel axis if not specified
    if channel_axis is None:
        # Heuristic: For unusual microscopy formats, usually the smallest dimension is the channel axis
        # For common 3D formats like (C, H, W), the first dimension is usually channels
        
        if len(image.shape) >= 3:
            # Find the dimension with the smallest size (likely channels)
            channel_axis = np.argmin(image.shape)
            
            # Default to first dimension if no clear candidate (common for microscopy formats)
            if len(image.shape) == 3 and image.shape[0] < 10:  # Typical channels are less than 10
                channel_axis = 0
                
        elif len(image.shape) == 2:
            # Single channel image
            channel_axis = None
        else:
            raise ValueError(f"Could not detect channel axis for image with shape {image.shape}")
    
    # Get number of channels
    if channel_axis is not None:
        num_channels = image.shape[channel_axis]
    else:
        num_channels = 1
    
    logger.info(f"Using channel axis {channel_axis} with {num_channels} channels")
    
    # Set default channel names if not provided
    if channel_names is None:
        channel_names = [f"C{i+1}" for i in range(num_channels)]
    elif len(channel_names) != num_channels:
        raise ValueError(f"Number of channel names ({len(channel_names)}) doesn't match number of channels ({num_channels})")
    
    # Prepare output paths
    base_name = image_path.stem
    extension = image_path.suffix
    
    output_paths = {}
    metadata = {
        "original_file": str(image_path),
        "original_shape": original_shape,
        "channel_axis": channel_axis,
        "transpose_axes": transpose_axes,
        "channels": {}
    }
    
    # Split channels and save individual files
    for i, name in enumerate(channel_names):
        output_filename = f"{base_name}{suffix_separator}{name}{extension}"
        output_path = output_dir / output_filename
        
        # Extract the channel
        if channel_axis is None or num_channels == 1:
            # Single channel image
            channel_img = image
        else:
            # Select the appropriate channel
            # Create a tuple of slices for the correct dimension
            slices = tuple(slice(None) if dim != channel_axis else slice(i, i+1) for dim in range(len(image.shape)))
            channel_img = image[slices]
            
            # Squeeze out the singleton dimension
            channel_img = np.squeeze(channel_img)
        
        # Normalize if requested
        if normalize:
            min_val = channel_img.min()
            max_val = channel_img.max()
            if max_val > min_val:  # Avoid division by zero
                channel_img = (channel_img - min_val) / (max_val - min_val)
                logger.info(f"Normalized channel {name} to range [0, 1]")
            
            # Convert to 32-bit float for normalized data
            channel_img = channel_img.astype(np.float32)
        
        # Save the channel image
        try:
            tifffile.imwrite(output_path, channel_img)
            logger.info(f"Saved channel {name} to {output_path}")
            output_paths[name] = output_path
            
            # Store metadata for this channel
            metadata["channels"][name] = {
                "output_path": str(output_path),
                "shape": channel_img.shape,
                "dtype": str(channel_img.dtype),
                "min": float(channel_img.min()),
                "max": float(channel_img.max()),
                "index": i
            }
            
        except Exception as e:
            logger.error(f"Error saving channel {name}: {e}")
            raise
    
    # Save metadata if requested
    if save_metadata:
        metadata_path = output_dir / f"{base_name}_metadata.json"
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Saved metadata to {metadata_path}")
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    return output_paths


def batch_transform_and_split(
    input_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    channel_names: Optional[List[str]] = None,
    channel_axis: Optional[int] = None,
    file_pattern: str = "*.tif*",
    suffix_separator: str = "_",
    transpose_axes: Optional[List[int]] = None,
    normalize: bool = False,
    save_metadata: bool = True,
    recursive: bool = False
) -> int:
    """
    Transform and split channels for multiple images in a directory
    
    Parameters
    ----------
    input_dir : str or Path
        Directory containing input images
    output_dir : str or Path, optional
        Directory to save split channel images. If None, will use the input directory.
    channel_names : List[str], optional
        List of names for the channels. If None, will use ["C1", "C2", ...].
    channel_axis : int, optional
        Index of the dimension that represents channels. If None, will attempt to detect automatically.
    file_pattern : str, optional
        Glob pattern to match files, default "*.tif*"
    suffix_separator : str, optional
        Separator between base filename and channel name, default "_"
    transpose_axes : List[int], optional
        List of axes indices to reorder dimensions before splitting channels.
    normalize : bool, optional
        Whether to normalize each channel to 0-1 range, default is False
    save_metadata : bool, optional
        Whether to save metadata about the original image shape and transformation
    recursive : bool, optional
        Whether to search subdirectories recursively, default False
        
    Returns
    -------
    int
        Number of images successfully processed
    """
    # Convert to Path objects
    input_dir = Path(input_dir)
    
    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = Path(output_dir)
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
            if output_dir != input_dir:
                rel_path = file_path.relative_to(input_dir)
                out_dir = output_dir / rel_path.parent
                out_dir.mkdir(parents=True, exist_ok=True)
            else:
                out_dir = output_dir
            
            # Transform and split channels
            transform_and_split(
                file_path,
                output_dir=out_dir,
                channel_names=channel_names,
                channel_axis=channel_axis,
                suffix_separator=suffix_separator,
                transpose_axes=transpose_axes,
                normalize=normalize,
                save_metadata=save_metadata
            )
            
            successful += 1
            logger.info(f"Processed {file_path.name}")
        
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
    
    logger.info(f"Successfully processed {successful} out of {len(files)} files")
    return successful


if __name__ == "__main__":
    # Example usage when run as a script
    import argparse
    
    parser = argparse.ArgumentParser(description="Transform and split channels in microscopy images")
    parser.add_argument("input", type=str, help="Input image or directory")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--channel-axis", type=int, help="Channel axis (dimension index)")
    parser.add_argument("--channels", type=str, nargs="+", help="Channel names")
    parser.add_argument("--transpose", type=int, nargs="+", help="Transpose axes order")
    parser.add_argument("--separator", type=str, default="_", help="Suffix separator (default: _)")
    parser.add_argument("--normalize", action="store_true", help="Normalize channel intensities")
    parser.add_argument("--no-metadata", action="store_false", dest="metadata", help="Don't save metadata")
    parser.add_argument("--recursive", action="store_true", help="Process subdirectories recursively")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Check if input is a file or directory
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Process single file
        try:
            transform_and_split(
                input_path,
                output_dir=args.output,
                channel_names=args.channels,
                channel_axis=args.channel_axis,
                suffix_separator=args.separator,
                transpose_axes=args.transpose,
                normalize=args.normalize,
                save_metadata=args.metadata
            )
            print(f"Image processed successfully")
        except Exception as e:
            print(f"Error: {e}")
    
    elif input_path.is_dir():
        # Process directory
        try:
            num_processed = batch_transform_and_split(
                input_path,
                output_dir=args.output,
                channel_names=args.channels,
                channel_axis=args.channel_axis,
                suffix_separator=args.separator,
                transpose_axes=args.transpose,
                normalize=args.normalize,
                save_metadata=args.metadata,
                recursive=args.recursive
            )
            print(f"Processed {num_processed} images")
        except Exception as e:
            print(f"Error: {e}")
    
    else:
        print(f"Error: Input path {input_path} does not exist")