'''
author: zyx
date: 2025-04-04
last_modified: 2025-06-06
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
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial

# Set up logging
logger = logging.getLogger(__name__)

def clear_gpu_cache(force: bool = False, gpu_id: Optional[int] = None) -> None:
    '''
    Clear GPU cache and run garbage collection
    
    Parameters
    ----------
    force : bool
        If True, forces synchronization before clearing cache
    gpu_id : int, optional
        Physical GPU ID for logging purposes (actual GPU being cleared depends on CUDA_VISIBLE_DEVICES)
    '''
    try:
        if torch.cuda.is_available():
            if force:
                torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()
            
            # Log which GPU's cache was cleared
            if gpu_id is not None:
                logger.info(f"GPU {gpu_id} cache cleared successfully")
            else:
                # When gpu_id not provided, show current visible devices
                visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'all')
                logger.info(f"GPU cache cleared successfully (visible devices: {visible_devices})")
    except Exception as e:
        gpu_str = f"GPU {gpu_id}" if gpu_id is not None else "GPU"
        logger.warning(f"Error clearing {gpu_str} cache: {e}")

def check_gpu() -> bool:
    '''
    Check if GPU is available for use with cellpose
    
    Returns
    -------
    bool
        True if GPU is available, False otherwise
    '''
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

def _process_files_on_gpu(
    gpu_id: int,
    files: List[str],
    model_params: Dict,
    seg_params: Dict,
    output_suffix: str = '_masks',
    clear_cache_every_n: int = 5
) -> List[str]:
    '''
    Helper function to process files on a specific GPU
    '''
    try:
        # Set this process to only see the specified physical GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        torch.cuda.set_device(0)  # 0 is the only visible device in this context
        # All GPU operations in this process will use physical GPU {gpu_id}
        
        from cellpose import models, io
        
        # Clear cache on the assigned GPU (physical GPU {gpu_id})
        clear_gpu_cache(gpu_id=gpu_id)
        
        model = models.CellposeModel(
            gpu=True,
            pretrained_model=model_params.get('pretrained_model', 'cpsam')
        )
        
        output_files = []
        
        for idx, file in enumerate(files):
            try:
                img = io.imread(file)
                
                if len(img.shape) == 3 and img.shape[2] >= 3:
                    channel_idx = seg_params['channels'][0] if seg_params['channels'][0] < img.shape[2] else 0
                    image_to_segment = img[:, :, channel_idx]
                else:
                    image_to_segment = img
                
                masks, flows, styles = model.eval(
                    image_to_segment,
                    diameter=seg_params.get('diameter', model.diam_labels),
                    channels=seg_params['channels'],
                    flow_threshold=seg_params['flow_threshold'],
                    cellprob_threshold=seg_params['cellprob_threshold'],
                )
                
                output_path = f"{file}{output_suffix}.npy"
                np.save(output_path, masks)
                output_files.append(output_path)
                
                del masks, flows, styles, img, image_to_segment
                
                if (idx + 1) % clear_cache_every_n == 0:
                    clear_gpu_cache(gpu_id=gpu_id)
                    
            except Exception as e:
                logger.error(f"GPU {gpu_id}: Error processing {file}: {e}")
                clear_gpu_cache(force=True, gpu_id=gpu_id)
        
        del model
        clear_gpu_cache(force=True, gpu_id=gpu_id)
        return output_files
        
    except Exception as e:
        logger.error(f"Error in GPU {gpu_id} processing: {e}")
        return []

def load_config(config_path: str) -> Dict:
    '''
    Load configuration from YAML file
    
    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file
        
    Returns
    -------
    Dict
        Configuration parameters
    '''
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
    '''
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
    '''
    try:
        from cellpose import io
        
        # Check if multi-GPU mode is enabled
        is_multi_gpu = isinstance(model, dict) and model.get('multi_gpu', False)
        
        if is_multi_gpu:
            # Multi-GPU processing
            num_gpus = model.get('num_gpus', 1)
            model_config = model.get('model_config', {})
            pretrained_model = model.get('pretrained_model', 'cpsam')
            
            # Get image files
            files = io.get_image_files(directory, exclude_pattern)
            logger.info(f"Found {len(files)} images in {directory}")
            
            if not files:
                logger.warning(f"No image files found in {directory}")
                return []
            
            # Determine actual number of GPUs to use
            available_gpus = torch.cuda.device_count()
            actual_gpus = min(num_gpus, available_gpus, len(files))
            
            if actual_gpus < num_gpus:
                logger.warning(f"Requested {num_gpus} GPUs, but only {actual_gpus} available/needed")
            
            # Split files into batches
            files_per_gpu = len(files) // actual_gpus
            remainder = len(files) % actual_gpus
            
            file_batches = []
            start_idx = 0
            
            for gpu_id in range(actual_gpus):
                batch_size = files_per_gpu + (1 if gpu_id < remainder else 0)
                end_idx = start_idx + batch_size
                file_batches.append((gpu_id, files[start_idx:end_idx]))
                start_idx = end_idx
            
            logger.info(f"Distributing files across {actual_gpus} GPUs")
            
            # Prepare parameters for multi-GPU processing
            model_params = {
                'pretrained_model': pretrained_model,
                'path': model_config.get('path')
            }
            seg_params = {
                'channels': channels,
                'diameter': diameter,
                'flow_threshold': flow_threshold,
                'cellprob_threshold': cellprob_threshold
            }
            
            # Process using multiple GPUs
            all_output_files = []
            ctx = mp.get_context('spawn')
            
            with ProcessPoolExecutor(max_workers=actual_gpus, mp_context=ctx) as executor:
                futures = {}
                for gpu_id, file_batch in file_batches:
                    future = executor.submit(
                        _process_files_on_gpu,
                        gpu_id,
                        file_batch,
                        model_params,
                        seg_params,
                        output_suffix,
                        clear_cache_every_n
                    )
                    futures[future] = gpu_id
                
                for future in as_completed(futures):
                    gpu_id = futures[future]
                    try:
                        output_files = future.result()
                        all_output_files.extend(output_files)
                        logger.info(f"GPU {gpu_id} completed: {len(output_files)} files processed")
                    except Exception as e:
                        logger.error(f"GPU {gpu_id} failed: {e}")
            
            return all_output_files
            
        else:
            # Single GPU processing (original code)
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
                        logger.info(f"GPU memory before segmentation: {mem_before:.2f} GB")
                    
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
    '''
    Run the cellpose segmentation pipeline using configuration from a YAML file
    
    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file
        
    Returns
    -------
    List[str]
        List of paths to all generated mask files
    '''
    # Load configuration
    config = load_config(config_path)
    
    try:
        from cellpose import models
        
        # Check for GPU and clear cache
        use_gpu = check_gpu()
        force_gpu = config.get('force_gpu', False)
        
        # Check for multi-GPU configuration
        multi_gpu_config = config.get('multi_gpu', {})
        use_multi_gpu = multi_gpu_config.get('enabled', False)
        num_gpus = multi_gpu_config.get('num_gpus', 1)
        
        # Configure model
        model_config = config['model']
        model_path = model_config.get('path')
        pretrained_model = model_config.get('pretrained_model', 'cpsam')
        
        # For multi-GPU, we'll create models inside the process_directory function
        # For single GPU, create model here
        model = None
        if not use_multi_gpu:
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
            if use_multi_gpu:
                # For multi-GPU, pass configuration as model parameter
                model_info = {
                    'multi_gpu': True,
                    'num_gpus': num_gpus,
                    'model_config': model_config,
                    'pretrained_model': pretrained_model
                }
                output_files = process_directory(
                    directory=directory,
                    model=model_info,
                    channels=channels,
                    diameter=diameter,
                    flow_threshold=flow_threshold,
                    cellprob_threshold=cellprob_threshold,
                    exclude_pattern=exclude_pattern,
                    output_suffix=output_suffix,
                    clear_cache=clear_cache,
                    clear_cache_every_n=clear_cache_every_n
                )
            else:
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
        if model is not None:
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
            current_gpu_id = torch.cuda.current_device()
            clear_gpu_cache(force=True, gpu_id=current_gpu_id)
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