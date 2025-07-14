'''
author: zyx
date: 2025-03-31
last_modified: 2025-05-24
description: 
    Functions for generating masks from vaious formats
'''
import json
import numpy as np
import cv2
from PIL import Image
import os
import argparse
import logging

logger = logging.getLogger(__name__)

def arguments_parser():
    parser = argparse.ArgumentParser(description="Convert QuPath GeoJSON to NPY segmentation mask.")
    
    parser.add_argument("geojson_path", type=str, help="Path to the QuPath .geojson file.")
    parser.add_argument("output_dir", type=str, help="Directory to save output masks.")
    parser.add_argument("image_width", type=int, help="Width of the original image.")
    parser.add_argument("image_height", type=int, help="Height of the original image.")
    parser.add_argument("--inner_holes", action="store_true", help="Include inner holes in the mask.")
    parser.add_argument("--downsample_factor", type=int, default=4, help="Downsampling factor (default: 4).")

    return parser

def generate_segmentation_mask(geojson_path:str,
                               output_dir:str,
                               image_width:int, image_height:int,
                               inner_holes:bool=True,
                               downsample_factor:float=0.1,
                               erosion_radius_before_upscaling:int=None)->bool:
    '''
    Read QuPath GeoJSON->generate segmentation mask->erosion(optional)->upscaling->saving
    Parms:
        geojson_path, output_dir: str
          Path to the QuPath .geojson file and directory to save the masks
        image_width, image_height: int
          Width and height of the original image
        downsample_factor: float
          Downsample factor for initial mask generation
        inner_holes: bool
          Whether to include inner holes in the mask
        erosion_radius_before_upscaling: int, optional
          If provided, erode the mask at downsampled resolution before upscaling
          This is much faster than eroding at full resolution
    
    Returns:
        bool: True if successful, False if error
    '''
    try:
        with open(geojson_path, "r") as f:
            data = json.load(f)

        mask = np.zeros((int(image_width*downsample_factor), int(image_height*downsample_factor)), dtype=np.uint8) #255 for white, 0 for black

        for feature in data['features']:
            geom = feature['geometry']
            # normalize to a list of polygons
            if geom['type'] == 'Polygon' or geom['type'] == 'MultiPolygon':
                coords_list = [geom['coordinates']]
            else:
                logger.warning(f"Unsupported geometry type: {geom['type']}")
                continue

            for polygon in coords_list:
                if not polygon:
                    continue

                # Outer boundary
                outer_boundary = np.array(polygon[0], np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [outer_boundary], color=255)
                # Inner holes
                if inner_holes:
                    for hole in polygon[1:]:
                        hole_boundary = np.array(hole, np.int32).reshape((-1, 1, 2))
                        cv2.fillPoly(mask, [hole_boundary], color=0)

        # Apply erosion at downsampled resolution if requested
        if erosion_radius_before_upscaling is not None and erosion_radius_before_upscaling > 0:
            from lmtools.compute.morphology import erode_binary_mask_2D
            # Scale erosion radius by downsample factor
            scaled_radius = int(erosion_radius_before_upscaling * downsample_factor)
            if scaled_radius > 0:
                print(f"Applying erosion with radius {scaled_radius} at downsampled resolution...")
                mask_binary = (mask > 0)
                eroded_mask = erode_binary_mask_2D(mask_binary, scaled_radius)
                mask = eroded_mask.astype(np.uint8) * 255

        os.makedirs(output_dir, exist_ok=True)

        # Save downsampled mask (with erosion if applied)
        base_name = os.path.splitext(os.path.basename(geojson_path))[0]
        if erosion_radius_before_upscaling is not None and erosion_radius_before_upscaling > 0:
            downsampled_path = os.path.join(output_dir, f"{base_name}_x{downsample_factor}mask_eroded{erosion_radius_before_upscaling}.npy")
        else:
            downsampled_path = os.path.join(output_dir, f"{base_name}_x{downsample_factor}mask.npy")
        np.save(downsampled_path, mask)
        print(f"Saved downsampled mask: {downsampled_path}")

        # Downsample the mask
        # Check if we're downsampling or upsampling
        if downsample_factor > 1:
            # Downsampling case
            new_width = int(image_width)
            new_height = int(image_height)
            full_res_mask = cv2.resize(mask, (new_height, new_width), #resize input is (height, width)
                         interpolation=cv2.INTER_NEAREST)
        else:
            # Upsampling case (downsample_factor < 1)
            new_width = int(image_width)
            new_height = int(image_height) 
            full_res_mask = cv2.resize(mask, (new_height, new_width),
                         interpolation=cv2.INTER_LINEAR)
            # Threshold to ensure binary mask after interpolation
            full_res_mask = (full_res_mask > 127).astype(np.uint8) * 255

        # Save full resolution mask
        if erosion_radius_before_upscaling is not None and erosion_radius_before_upscaling > 0:
            full_res_path = os.path.join(output_dir, f"{base_name}_tissue_mask_eroded{erosion_radius_before_upscaling}.npy")
        else:
            full_res_path = os.path.join(output_dir, f"{base_name}_tissue_mask.npy")
        np.save(full_res_path, full_res_mask)
        print(f"Saved full resolution mask: {full_res_path}")

        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    '''
    Usage:
        1. Creat a python environment with cv2 and PIL (for lilab-wsl1, napari-env is good)
        2. Run the script with the following command:
        ```bash
            python geojson2npy.py <geojson_path> <output_dir> <image_width> <image_height> [--inner_holes] <downsample_factor>
        ```        
    '''
    parser = argparse.ArgumentParser(description="Convert QuPath GeoJSON to NPY segmentation mask.")
    
    parser.add_argument("geojson_path", type=str, help="Path to the QuPath .geojson file.")
    parser.add_argument("output_dir", type=str, help="Directory to save output masks.")
    parser.add_argument("image_width", type=int, help="Width of the original image.")
    parser.add_argument("image_height", type=int, help="Height of the original image.")
    parser.add_argument("--inner_holes", action="store_true", help="Include inner holes in the mask.")
    parser.add_argument("--downsample_factor", type=int, default=4, help="Downsampling factor (default: 4).")

    args = parser.parse_args()

    generate_segmentation_mask(
        geojson_path=args.geojson_path,
        output_dir=args.output_dir,
        image_width=args.image_width,
        image_height=args.image_height,
        inner_holes=args.inner_holes,
        downsample_factor=args.downsample_factor
    )