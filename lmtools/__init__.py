"""
LMTools - Tools and scripts for processing and visualizing light microscopy data.
"""

__version__ = "0.0.1"

# Import main functionality for easier access
from lmtools.io import load_nd2
from lmtools.seg import (
    generate_segmentation_mask, 
    maskExtract,
    analyze_segmentation,
    summarize_segmentation,
    get_bounding_boxes,
    run_pipeline
)

__all__ = [
    'load_nd2',
    'generate_segmentation_mask',
    'maskExtract',
    'analyze_segmentation',
    'summarize_segmentation',
    'get_bounding_boxes',
    'run_pipeline',
]