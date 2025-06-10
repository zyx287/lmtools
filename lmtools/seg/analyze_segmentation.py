'''
author: zyx
date: 2025-04-04
last_modified: 2025-04-05
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
    '''
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
    '''
    # Load the mask if it's a file path
    if isinstance(mask, str):
        if not os.path.exists(mask):
            raise FileNotFoundError(f"Mask file not found: {mask}")
        
        try:
            mask_data = np.load(mask, allow_pickle=True)
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
    '''
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
    '''
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


def get_bounding_boxes(
    mask: Union[str, np.ndarray],
    label_id: Optional[int] = None,
    return_format: str = 'slices'
) -> Union[List[Optional[Tuple]], Dict[int, Union[Tuple, List[int]]], Tuple, List[int], None]:
    '''
    Get bounding boxes for labeled objects in a segmentation mask
    
    Parameters
    ----------
    mask : str or numpy.ndarray
        Path to a .npy file containing a segmentation mask or the mask array directly.
        Mask should contain integer labels where 0 is background and positive integers
        represent different objects.
    label_id : int, optional
        Specific label ID to get the bounding box for. If None, returns bounding boxes
        for all objects in the mask.
    return_format : str, optional
        Format for returned bounding boxes, by default 'slices'
        - 'slices': Returns slice objects as returned by ndimage.find_objects
        - 'coords': Returns [min_y, min_x, max_y, max_x] for 2D or 
                   [min_z, min_y, min_x, max_z, max_y, max_x] for 3D
        
    Returns
    -------
    Various types depending on input parameters:
        - If label_id is None and return_format is 'slices':
            List of tuples of slice objects, one per labeled object
        - If label_id is None and return_format is 'coords':
            Dict mapping label IDs to coordinate lists
        - If label_id is specified and return_format is 'slices':
            Tuple of slice objects for the specified label
        - If label_id is specified and return_format is 'coords':
            List of coordinates for the specified label
        - None if label_id is specified but not found in the mask
    
    Raises
    ------
    ValueError
        If mask is not a valid segmentation mask or return_format is invalid
    FileNotFoundError
        If mask file path does not exist
    '''
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
    
    # Validate return_format
    if return_format not in ['slices', 'coords']:
        raise ValueError("return_format must be either 'slices' or 'coords'")
    
    # Find objects in the mask
    objects = ndimage.find_objects(mask_data)
    
    # If a specific label is requested
    if label_id is not None:
        # Validate label_id
        if not isinstance(label_id, int) or label_id <= 0:
            raise ValueError("label_id must be a positive integer")
        
        # Check if label_id exists in the mask
        max_label = len(objects)
        if label_id > max_label:
            return None
        
        # Get the bounding box for the specified label
        # Note: find_objects returns 0-indexed results for 1-indexed labels
        bbox_slices = objects[label_id - 1]
        
        if bbox_slices is None:
            return None
            
        # Return in requested format
        if return_format == 'slices':
            return bbox_slices
        else:  # 'coords'
            # Convert slices to coordinates [min_y, min_x, max_y, max_x] for 2D
            # or [min_z, min_y, min_x, max_z, max_y, max_x] for 3D
            coords = []
            for s in bbox_slices:
                coords.extend([s.start, s.stop - 1])
            
            # Reorder to [min_y, min_x, max_y, max_x] for 2D
            # or [min_z, min_y, min_x, max_z, max_y, max_x] for 3D
            if len(coords) == 4:  # 2D
                return [coords[0], coords[2], coords[1], coords[3]]
            elif len(coords) == 6:  # 3D
                return [coords[0], coords[2], coords[4], coords[1], coords[3], coords[5]]
            else:
                return coords
    
    # Return all bounding boxes
    if return_format == 'slices':
        return objects
    else:  # 'coords'
        result = {}
        for i, bbox_slices in enumerate(objects):
            if bbox_slices is None:
                result[i + 1] = None
                continue
                
            # Convert slices to coordinates
            coords = []
            for s in bbox_slices:
                coords.extend([s.start, s.stop - 1])
            
            # Reorder coordinates
            if len(coords) == 4:  # 2D
                result[i + 1] = [coords[0], coords[2], coords[1], coords[3]]
            elif len(coords) == 6:  # 3D
                result[i + 1] = [coords[0], coords[2], coords[4], coords[1], coords[3], coords[5]]
            else:
                result[i + 1] = coords
                
        return result

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze segmentation mask")
    parser.add_argument("mask_path", type=str, help="Path to the segmentation mask .npy file")
    parser.add_argument("--no-stats", action="store_true", help="Don't compute per-object statistics")
    parser.add_argument("--min-size", type=int, default=0, help="Minimum object size to include (pixels)")
    parser.add_argument("--output", type=str, help="Save results to this JSON file")
    parser.add_argument("--bbox", action="store_true", help="Get bounding boxes for objects")
    parser.add_argument("--label", type=int, help="Specific label to get bounding box for")
    
    args = parser.parse_args()
    
    try:
        if args.bbox:
            # Get bounding boxes
            bboxes = get_bounding_boxes(
                args.mask_path,
                label_id=args.label,
                return_format='coords'
            )
            
            if args.label:
                print(f"Bounding box for label {args.label}: {bboxes}")
            else:
                print(f"Found {len(bboxes)} bounding boxes")
                if len(bboxes) > 0:
                    print("First 5 bounding boxes:")
                    for i, bbox in enumerate(list(bboxes.items())[:5]):
                        print(f"  Label {bbox[0]}: {bbox[1]}")
            
            # Save to JSON if requested
            if args.output:
                import json
                with open(args.output, 'w') as f:
                    json.dump(bboxes, f, indent=2)
                print(f"Results saved to {args.output}")
        else:
            # Run standard segmentation analysis
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