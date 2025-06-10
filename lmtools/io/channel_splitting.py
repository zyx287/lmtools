'''
author: zyx
date: 2025-04-19
last_modified: 2025-04-19
description: 
    Functions for splitting multi-channel microscopy images into separate channel files
'''
import os
import numpy as np
from typing import Union, List, Optional, Dict, Tuple
from pathlib import Path
import logging
import tifffile
from glob import glob

# Set up logging
logger = logging.getLogger(__name__)

def split_channels(
    image_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    channel_names: Optional[List[str]] = None,
    suffix_separator: str = "_"
) -> Dict[str, Path]:
    '''
    Split a multi-channel image into separate single-channel images.
    
    Parameters
    ----------
    image_path : str or Path
        Path to the input multi-channel image
    output_dir : str or Path, optional
        Directory to save the split channel images. If None, will use the same directory as the input.
    channel_names : List[str], optional
        List of names for the channels. If None, will use ["C1", "C2", ...].
        Must match the number of channels in the image.
    suffix_separator : str, optional
        Separator to use between the base filename and the channel name, default is "_"
        
    Returns
    -------
    Dict[str, Path]
        Dictionary mapping channel names to output file paths
    
    Raises
    ------
    ValueError
        If the image is not multi-channel or if channel_names length doesn't match
    FileNotFoundError
        If the input file doesn't exist
    '''
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
    
    # Check image dimensions
    if len(image.shape) < 3:
        raise ValueError(f"Image has only {len(image.shape)} dimensions. Need at least 3 dimensions to split channels.")
    
    # Determine the number of channels
    # Typically channels are in the 3rd dimension (index 2) for RGB images, 
    # but could be in other dimensions for microscopy formats
    
    # Assume the last dimension is channels for RGB-like images
    if len(image.shape) == 3 and image.shape[2] <= 4:
        num_channels = image.shape[2]
        channel_axis = 2
    # For multi-dimensional microscopy images, assume one of the dimensions is channels
    elif len(image.shape) > 3:
        # Heuristic: Smallest dimension is likely to be channels
        # This might need adjustment for specific microscopy formats
        channel_axis = np.argmin(image.shape)
        num_channels = image.shape[channel_axis]
    else:
        # Default assumption for 3D image stacks without clear channel dimension
        num_channels = 1
        channel_axis = None
    
    logger.info(f"Detected {num_channels} channels in dimension {channel_axis}")
    
    # Set default channel names if not provided
    if channel_names is None:
        channel_names = [f"C{i+1}" for i in range(num_channels)]
    elif len(channel_names) != num_channels:
        raise ValueError(f"Number of channel names ({len(channel_names)}) doesn't match number of channels ({num_channels})")
    
    # Prepare output paths
    base_name = image_path.stem
    extension = image_path.suffix
    
    output_paths = {}
    
    # Split channels and save individual files
    for i, name in enumerate(channel_names):
        output_filename = f"{base_name}{suffix_separator}{name}{extension}"
        output_path = output_dir / output_filename
        
        # Extract the channel
        if channel_axis is None or num_channels == 1:
            # Single channel image or unclear channel dimension
            channel_img = image
        else:
            # Select the appropriate channel
            # Create a tuple of slices for the correct dimension
            slices = tuple(slice(None) if dim != channel_axis else slice(i, i+1) for dim in range(len(image.shape)))
            channel_img = image[slices]
            
            # Squeeze out the singleton dimension
            channel_img = np.squeeze(channel_img)
        
        # Save the channel image
        try:
            tifffile.imwrite(output_path, channel_img)
            logger.info(f"Saved channel {name} to {output_path}")
            output_paths[name] = output_path
        except Exception as e:
            logger.error(f"Error saving channel {name}: {e}")
            raise
    
    return output_paths


def batch_split_channels(
    input_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    channel_names: Optional[List[str]] = None,
    file_pattern: str = "*.tif*",
    suffix_separator: str = "_",
    recursive: bool = False
) -> int:
    '''
    Split channels for multiple images in a directory
    
    Parameters
    ----------
    input_dir : str or Path
        Directory containing input images
    output_dir : str or Path, optional
        Directory to save split channel images. If None, will use the input directory.
    channel_names : List[str], optional
        List of names for the channels. If None, will use ["C1", "C2", ...].
    file_pattern : str, optional
        Glob pattern to match files, default "*.tif*"
    suffix_separator : str, optional
        Separator between base filename and channel name, default "_"
    recursive : bool, optional
        Whether to search subdirectories recursively, default False
        
    Returns
    -------
    int
        Number of images successfully processed
    '''
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
            
            # Split channels
            split_channels(
                file_path,
                output_dir=out_dir,
                channel_names=channel_names,
                suffix_separator=suffix_separator
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
    
    parser = argparse.ArgumentParser(description="Split channels in microscopy images")
    parser.add_argument("input", type=str, help="Input image or directory")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--channels", type=str, nargs="+", help="Channel names")
    parser.add_argument("--separator", type=str, default="_", help="Suffix separator (default: _)")
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
            split_channels(
                input_path,
                output_dir=args.output,
                channel_names=args.channels,
                suffix_separator=args.separator
            )
            print(f"Image processed successfully")
        except Exception as e:
            print(f"Error: {e}")
    
    elif input_path.is_dir():
        # Process directory
        try:
            num_processed = batch_split_channels(
                input_path,
                output_dir=args.output,
                channel_names=args.channels,
                suffix_separator=args.separator,
                recursive=args.recursive
            )
            print(f"Processed {num_processed} images")
        except Exception as e:
            print(f"Error: {e}")
    
    else:
        print(f"Error: Input path {input_path} does not exist")