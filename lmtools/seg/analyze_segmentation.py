'''
author: zyx
date: 2025-04-04
last_modified: 2025-04-04
description: 
    Functions for analyzing segmentation masks and extracting statistics
'''
import numpy as np
from typing import Dict, Union, List, Optional, Tuple
import os
from scipy import ndimage
import pandas as pd

def analyze_segmentation(
    mask: Union[str, np.ndarray],
    compute_object_stats: bool = True,
    min_size: int = 0
) -> Dict:
    """
    Analyze a segmentation mask and compute various statistics
    
    Parameters
    ----------
    mask : str or numpy.ndarray
        Path to a .npy file containing a segmentation mask or the mask array directly.
        Mask should contain integer labels where 0 is background and positive integers
        represent different objects.
    compute_object_stats : bool, optional
        Whether to compute statistics for individual objects, by default True
    min_size : int, optional
        Minimum object size (in pixels) to include in the analysis, by default 0
        
    Returns
    -------
    Dict
        Dictionary containing segmentation statistics:
        - 'dimensions': tuple of mask dimensions (height, width, [depth])
        - 'shape': tuple matching the input array shape
        - 'num_objects': number of distinct objects (excluding background)
        - 'object_ids': list of object IDs
        - 'total_mask_area': total area of all objects in pixels
        - 'object_stats': DataFrame with per-object statistics (if compute_object_stats=True)
            - 'id': object ID
            - 'area': object area in pixels
            - 'centroid': object centroid coordinates
            - 'bbox': bounding box (min_row, min_col, max_row, max_col)
    """
    # Load the mask if it's a file path
    if isinstance(mask, str):
        if not os.path.exists(mask):
            raise FileNotFoundError(f"Mask file not found: {mask}")
        
        try:
            mask_data = np.load(mask)
        except Exception as e:
            raise ValueError(f"Error loading mask file: {e}")
    else:
        mask_data = mask
    
    # Validate the mask
    if not isinstance(mask_data, np.ndarray):
        raise TypeError("Mask must be a numpy array")
    
    if mask_data.size == 0:
        raise ValueError("Mask is empty")
    
    # Get basic information
    shape = mask_data.shape
    dimensions = shape if len(shape) <= 3 else shape[1:]  # Handle channel dimension if present
    
    # Find unique object IDs (excluding background which is 0)
    object_ids = np.unique(mask_data)
    object_ids = object_ids[object_ids > 0]  # Exclude background
    num_objects = len(object_ids)
    
    # Calculate total mask area (non-zero pixels)
    total_mask_area = np.sum(mask_data > 0)
    
    # Initialize result dictionary
    result = {
        'dimensions': dimensions,
        'shape': shape,
        'num_objects': num_objects,
        'object_ids': object_ids.tolist(),
        'total_mask_area': int(total_mask_area)
    }
    
    # Compute per-object statistics if requested
    if compute_object_stats and num_objects > 0:
        stats = []
        
        # For each object ID, compute statistics
        for obj_id in object_ids:
            # Create binary mask for this object
            obj_mask = (mask_data == obj_id)
            
            # Calculate area
            area = np.sum(obj_mask)
            
            # Skip if object is too small
            if area < min_size:
                continue
                
            # Calculate centroid
            centroid = ndimage.center_of_mass(obj_mask)
            
            # Calculate bounding box
            if len(shape) == 2:
                rows, cols = np.where(obj_mask)
                if len(rows) > 0 and len(cols) > 0:
                    bbox = (int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max()))
                else:
                    bbox = (0, 0, 0, 0)
            elif len(shape) == 3:
                planes, rows, cols = np.where(obj_mask)
                if len(rows) > 0 and len(cols) > 0 and len(planes) > 0:
                    bbox = (int(planes.min()), int(rows.min()), int(cols.min()), 
                            int(planes.max()), int(rows.max()), int(cols.max()))
                else:
                    bbox = (0, 0, 0, 0, 0, 0)
            else:
                bbox = None
            
            # Store object statistics
            stats.append({
                'id': int(obj_id),
                'area': int(area),
                'centroid': tuple(float(c) for c in centroid),
                'bbox': bbox
            })
        
        # Convert stats to DataFrame for easier analysis
        result['object_stats'] = pd.DataFrame(stats) if stats else pd.DataFrame()
    
    return result


def summarize_segmentation(
    results: Dict,
    print_summary: bool = True
) -> str:
    """
    Create a human-readable summary of segmentation analysis results
    
    Parameters
    ----------
    results : Dict
        Results dictionary from analyze_segmentation()
    print_summary : bool, optional
        Whether to print the summary to stdout, by default True
        
    Returns
    -------
    str
        Summary text
    """
    dimensions = 'x'.join(str(d) for d in results['dimensions'])
    
    summary = [
        f"Segmentation Summary:",
        f"  Dimensions: {dimensions}",
        f"  Number of objects: {results['num_objects']}",
        f"  Total mask area: {results['total_mask_area']} pixels"
    ]
    
    if 'object_stats' in results and not results['object_stats'].empty:
        df = results['object_stats']
        summary.extend([
            f"  Object statistics:",
            f"    Min area: {df['area'].min()} pixels",
            f"    Max area: {df['area'].max()} pixels",
            f"    Mean area: {df['area'].mean():.2f} pixels",
            f"    Median area: {df['area'].median()} pixels"
        ])
    
    summary_text = '\n'.join(summary)
    
    if print_summary:
        print(summary_text)
    
    return summary_text


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze segmentation mask")
    parser.add_argument("mask_path", type=str, help="Path to the segmentation mask .npy file")
    parser.add_argument("--no-stats", action="store_true", help="Don't compute per-object statistics")
    parser.add_argument("--min-size", type=int, default=0, help="Minimum object size to include (pixels)")
    parser.add_argument("--output", type=str, help="Save results to this JSON file")
    
    args = parser.parse_args()
    
    try:
        results = analyze_segmentation(
            args.mask_path, 
            compute_object_stats=not args.no_stats,
            min_size=args.min_size
        )
        
        summarize_segmentation(results)
        
        if args.output:
            import json
            
            # Convert numpy types to Python native types for JSON serialization
            def convert_for_json(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, pd.DataFrame):
                    return obj.to_dict(orient='records')
                return obj
            
            # Convert results to JSON-serializable format
            json_results = {k: convert_for_json(v) for k, v in results.items()}
            
            with open(args.output, 'w') as f:
                json.dump(json_results, f, indent=2)
                
            print(f"Results saved to {args.output}")
            
    except Exception as e:
        print(f"Error: {e}")