'''
author: zyx
date: 2025-04-04
last_modified: 2025-04-05
description: 
    CellPose segmentation pipeline for light microscopy images with enhanced GPU cache management
'''
import os
import numpy as np
import torch
import logging
from typing import List, Dict, Tuple, Union, Optional, Any
import yaml
from glob import glob
from pathlib import Path
import gc

# Set up logging
logger = logging.getLogger(__name__)

def clear_gpu_cache(force: bool = False) -> None:
    """
    Clear GPU cache and run garbage collection
    
    Parameters
    ----------
    force : bool
        If True, forces synchronization before clearing cache
    """
    try:
        if torch.cuda.is_available():
            if force:
                torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()
            logger.debug("GPU cache cleared successfully")
    except Exception as e:
        logger.warning(f"Error clearing GPU cache: {e}")

def check_gpu() -> bool:
    """
    Check if GPU is available for use with cellpose
    
    Returns
    -------
    bool
        True if GPU is available, False otherwise
    """
    try:
        from cellpose import core
        
        # Clear cache before checking
        clear_gpu_cache()
        
        use_GPU = core.use_gpu()
        logger.info(f'GPU activated: {["NO", "YES"][use_GPU]}')
        
        if use_GPU:
            # Log GPU memory status
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            logger.info(f'GPU memory: {allocated:.2f}/{gpu_mem:.2f} GB allocated')
        
        return use_GPU
    except Exception as e:
        logger.warning(f"Error checking GPU availability: {e}")
        return False

def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file
    
    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file
        
    Returns
    -------
    Dict
        Configuration parameters
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate essential configuration parameters
        required_keys = ['model', 'input']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration key: {key}")
        
        return config
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        raise

def process_directory(
    directory: str,
    model: Any,
    channels: List[int] = [0, 0],
    diameter: Optional[float] = None,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    exclude_pattern: str = '_masks',
    output_suffix: str = '_masks',
    clear_cache: bool = True,
    clear_cache_every_n: int = 5
) -> List[str]:
    """
    Process all images in a directory with cellpose
    
    Parameters
    ----------
    directory : str
        Directory containing images to process
    model : cellpose.models.CellposeModel
        Pretrained cellpose model
    channels : List[int]
        Channel indices [channel_to_segment, optional_nuclear_channel]
    diameter : float, optional
        Cell diameter for segmentation, by default None (use model default)
    flow_threshold : float
        Flow threshold for cellpose, by default 0.4
    cellprob_threshold : float
        Cell probability threshold, by default 0.0
    exclude_pattern : str
        Skip files containing this pattern, by default '_masks'
    output_suffix : str
        Suffix to add to output files, by default '_masks'
    clear_cache : bool
        Whether to clear GPU cache after each image, by default True
    clear_cache_every_n : int
        Clear cache every N images for better performance, by default 5
        
    Returns
    -------
    List[str]
        List of paths to the generated mask files
    """
    try:
        from cellpose import io
        
        # Clear cache at the start
        if torch.cuda.is_available():
            clear_gpu_cache(force=True)
            logger.info("GPU cache cleared before processing directory")
        
        # Get image files, excluding any that match the exclude pattern
        files = io.get_image_files(directory, exclude_pattern)
        logger.info(f"Found {len(files)} images in {directory}")
        
        if not files:
            logger.warning(f"No image files found in {directory}")
            return []
        
        output_files = []
        
        for idx, file in enumerate(files):
            try:
                # Load image
                img = io.imread(file)
                logger.info(f"Processing {file} ({idx+1}/{len(files)}), shape: {img.shape}")
                
                # Extract channel to segment (default to channel 1 - green)
                if len(img.shape) == 3 and img.shape[2] >= 3:
                    # Multi-channel image (RGB or more)
                    channel_idx = channels[0] if channels[0] < img.shape[2] else 0
                    image_to_segment = img[:, :, channel_idx]
                else:
                    # Grayscale image
                    image_to_segment = img
                
                # Monitor GPU memory before segmentation
                if torch.cuda.is_available():
                    mem_before = torch.cuda.memory_allocated(0) / 1024**3
                    logger.debug(f"GPU memory before segmentation: {mem_before:.2f} GB")
                
                # Run segmentation
                masks, flows, styles = model.eval(
                    image_to_segment,
                    diameter=diameter if diameter is not None else model.diam_labels,
                    channels=channels,
                    flow_threshold=flow_threshold,
                    cellprob_threshold=cellprob_threshold,
                )
                
                # Create output file path
                output_path = f"{file}{output_suffix}.npy"
                
                # Save masks
                np.save(output_path, masks)
                logger.info(f"Detected {masks.max()} cells in {file}")
                logger.info(f"Saved masks to {output_path}")
                
                output_files.append(output_path)
                
                # Clean up variables
                del masks, flows, styles, img, image_to_segment
                
                # Clear cache based on settings
                if clear_cache and torch.cuda.is_available():
                    # Always clear after every nth image or if memory usage is high
                    should_clear = (idx + 1) % clear_cache_every_n == 0
                    
                    # Check memory usage
                    mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
                    mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    mem_percent = (mem_allocated / mem_total) * 100
                    
                    # Force clear if memory usage exceeds 80%
                    if mem_percent > 80:
                        should_clear = True
                        logger.warning(f"GPU memory usage high ({mem_percent:.1f}%), forcing cache clear")
                    
                    if should_clear:
                        clear_gpu_cache()
                        logger.debug(f"GPU cache cleared after processing image {idx+1}")
                
            except Exception as e:
                logger.error(f"Error processing file {file}: {e}")
                # Clear cache on error to prevent memory issues
                if torch.cuda.is_available():
                    clear_gpu_cache(force=True)
        
        # Final cache clear after processing all images
        if torch.cuda.is_available():
            clear_gpu_cache(force=True)
            logger.info("GPU cache cleared after processing all images")
        
        logger.info(f"Completed processing all images in {directory}")
        return output_files
    
    except Exception as e:
        logger.error(f"Error processing directory {directory}: {e}")
        # Ensure cache is cleared on error
        if torch.cuda.is_available():
            clear_gpu_cache(force=True)
        return []

def run_pipeline(config_path: str) -> List[str]:
    """
    Run the cellpose segmentation pipeline using configuration from a YAML file
    
    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file
        
    Returns
    -------
    List[str]
        List of paths to all generated mask files
    """
    # Load configuration
    config = load_config(config_path)
    
    try:
        from cellpose import models
        
        # Check for GPU and clear cache
        use_gpu = check_gpu()
        force_gpu = config.get('force_gpu', False)
        
        # Configure model
        model_config = config['model']
        model_path = model_config.get('path')
        pretrained_model = model_config.get('pretrained_model', 'cyto')
        
        # Load the model
        model = models.CellposeModel(
            gpu=use_gpu or force_gpu,
            pretrained_model=model_path if model_path else pretrained_model
        )
        
        # Get segmentation parameters
        seg_params = config.get('segmentation_params', {})
        channels = seg_params.get('channels', [0, 0])
        diameter = seg_params.get('diameter', None)
        flow_threshold = seg_params.get('flow_threshold', 0.4)
        cellprob_threshold = seg_params.get('cellprob_threshold', 0.0)
        
        # Get output parameters
        output_params = config.get('output', {})
        exclude_pattern = output_params.get('exclude_pattern', '_masks')
        output_suffix = output_params.get('suffix', '_masks')
        clear_cache = output_params.get('clear_cache', True)
        clear_cache_every_n = output_params.get('clear_cache_every_n', 5)
        
        # Process all directories
        all_output_files = []
        
        # Handle input directories
        input_config = config['input']
        
        # Single directory as string
        if isinstance(input_config, str):
            directories = [input_config]
        # List of directories
        elif isinstance(input_config, list):
            directories = input_config
        # Dict with 'directories' key
        elif isinstance(input_config, dict) and 'directories' in input_config:
            directories = input_config['directories']
        else:
            raise ValueError("Invalid input configuration format")
        
        # Process each directory
        for directory in directories:
            # Skip if directory doesn't exist
            if not os.path.isdir(directory):
                logger.warning(f"Directory {directory} does not exist, skipping")
                continue
            
            # Clear cache before processing each directory
            if torch.cuda.is_available():
                clear_gpu_cache()
                logger.info(f"GPU cache cleared before processing directory: {directory}")
            
            # Process the directory
            output_files = process_directory(
                directory=directory,
                model=model,
                channels=channels,
                diameter=diameter,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
                exclude_pattern=exclude_pattern,
                output_suffix=output_suffix,
                clear_cache=clear_cache,
                clear_cache_every_n=clear_cache_every_n
            )
            
            all_output_files.extend(output_files)
        
        # Final cleanup
        del model
        if torch.cuda.is_available():
            clear_gpu_cache(force=True)
            logger.info("Final GPU cache clear completed")
        
        logger.info(f"Pipeline completed. Generated {len(all_output_files)} mask files.")
        return all_output_files
    
    except ImportError:
        logger.error("Could not import cellpose. Please install it with: pip install cellpose")
        return []
    except Exception as e:
        logger.error(f"Error running pipeline: {e}")
        # Ensure cache is cleared on error
        if torch.cuda.is_available():
            clear_gpu_cache(force=True)
        return []

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Run cellpose segmentation pipeline")
    parser.add_argument("config", type=str, help="Path to YAML configuration file")
    args = parser.parse_args()
    
    run_pipeline(args.config)