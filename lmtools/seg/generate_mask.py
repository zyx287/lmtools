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
                               downsample_factor:float=0.1)->bool:
    '''
    Read QuPath GeoJSON->generate segmentation mask->downsampling->saving
    Parms:
        geojson_path, output_dir: str
          Path to the QuPath .geojson file and directory to save the masks
        image_width, image_height: int
          Width and height of the original image
        downsample_factor: int
          Downsample <factor> times
    
    Returns:
        y: type
          description
    '''
    try:
        with open(geojson_path, "r") as f:
            data = json.load(f)

        mask = np.zeros((image_width, image_height), dtype=np.uint8) #255 for white, 0 for black

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

        os.makedirs(output_dir, exist_ok=True)

        # Save full-resolution mask
        base_name = os.path.splitext(os.path.basename(geojson_path))[0]
        full_res_path = os.path.join(output_dir, f"{base_name}_mask.npy")
        np.save(full_res_path, mask)
        print(f"Saved full-resolution mask: {full_res_path}")

        # Downsample the mask
        # Check if we're downsampling or upsampling
        if downsample_factor > 1:
            # Downsampling case
            new_width = int(image_width / downsample_factor)
            new_height = int(image_height / downsample_factor)
            downsampled_mask = cv2.resize(mask, (new_height, new_width), 
                         interpolation=cv2.INTER_NEAREST)
        else:
            # Upsampling case (downsample_factor < 1)
            new_width = int(image_width / downsample_factor)
            new_height = int(image_height / downsample_factor) 
            downsampled_mask = cv2.resize(mask, (new_height, new_width),
                         interpolation=cv2.INTER_LINEAR)
            # Threshold to ensure binary mask after interpolation
            downsampled_mask = (downsampled_mask > 127).astype(np.uint8) * 255

        # Save downsampled mask
        downsampled_path = os.path.join(output_dir, f"{base_name}_mask_{downsample_factor}xdown.npy")
        np.save(downsampled_path, downsampled_mask)
        print(f"Saved downsampled mask: {downsampled_path}")

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