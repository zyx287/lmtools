"""
LMTools - Tools and scripts for processing and visualizing light microscopy data.
"""

__version__ = "0.0.1"

# Import main functionality for easier access
from lmtools.io import (
    load_nd2, 
    downsample_image, 
    batch_downsample, 
    split_channels, 
    batch_split_channels,
    transform_and_split,
    batch_transform_and_split
)
from lmtools.seg import (
    generate_segmentation_mask, 
    maskExtract,
    analyze_segmentation,
    summarize_segmentation,
    get_bounding_boxes,
    run_pipeline,
    threshold_segment,
    watershed_segment,
    region_growing_segment,
    intensity_filter,
    visualize_intensity_regions,
    overlap_filter,
    intensity_channel_filter,
)

__all__ = [
    'load_nd2',
    'downsample_image',
    'batch_downsample',
    'split_channels',
    'batch_split_channels',
    'transform_and_split',
    'batch_transform_and_split',
    'generate_segmentation_mask',
    'maskExtract',
    'analyze_segmentation',
    'summarize_segmentation',
    'get_bounding_boxes',
    'run_pipeline',
    'threshold_segment',
    'watershed_segment',
    'region_growing_segment',
    'intensity_filter',
    'visualize_intensity_regions',
    'overlap_filter',
    'intensity_channel_filter',
]