'''
author: zyx
date: 2025-03-31
last_modified: 2025-07-31
description: 
    Functions for generating masks from various formats
'''
import json
import numpy as np
import cv2
from PIL import Image
import os
import argparse
import logging
from pathlib import Path
from typing import Optional, Union, Tuple

logger = logging.getLogger(__name__)

def geojson2npy(
    geojson_path: Union[str, Path],
    width: int,
    height: int,
    inner_holes: bool = True,
    save_npy: bool = False,
    output_path: Optional[Union[str, Path]] = None,
    sample_name: Optional[str] = None
) -> np.ndarray:
    '''
    Convert QuPath GeoJSON annotations to numpy binary mask.
    
    Parameters:
    -----------
    geojson_path : str or Path
        Path to the QuPath .geojson file
    width : int
        Width of the mask to generate
    height : int
        Height of the mask to generate
    inner_holes : bool
        Whether to include inner holes in the mask (default: True)
    save_npy : bool
        Whether to save the mask as npy file (default: False)
    output_path : str or Path, optional
        Path to save the npy file (required if save_npy=True)
    sample_name : str, optional
        Sample name for data organizer naming convention (used if save_npy=True)
        
    Returns:
    --------
    np.ndarray
        Binary mask array (0 and 255)
    '''
    try:
        with open(geojson_path, "r") as f:
            data = json.load(f)
        
        # Create empty mask
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Process each feature
        for feature in data['features']:
            geom = feature['geometry']
            
            # Handle different geometry types
            if geom['type'] == 'Polygon':
                coords_list = [geom['coordinates']]
            elif geom['type'] == 'MultiPolygon':
                coords_list = geom['coordinates']
            else:
                logger.warning(f"Unsupported geometry type: {geom['type']}")
                continue
            
            # Process each polygon
            for polygon in coords_list:
                if not polygon:
                    continue
                
                # Scale coordinates to the specified dimensions
                # GeoJSON coordinates are in original image space
                scaled_polygon = []
                for ring in polygon:
                    scaled_ring = np.array(ring, np.float32)
                    # Note: GeoJSON uses (x,y) order, which matches cv2 expectations
                    scaled_polygon.append(scaled_ring.astype(np.int32))
                
                # Fill outer boundary
                if scaled_polygon:
                    outer_boundary = scaled_polygon[0].reshape((-1, 1, 2))
                    cv2.fillPoly(mask, [outer_boundary], color=255)
                    
                    # Handle inner holes
                    if inner_holes and len(scaled_polygon) > 1:
                        for hole in scaled_polygon[1:]:
                            hole_boundary = hole.reshape((-1, 1, 2))
                            cv2.fillPoly(mask, [hole_boundary], color=0)
        
        # Save if requested
        if save_npy:
            if output_path is None:
                raise ValueError("output_path must be provided when save_npy=True")
            
            output_path = Path(output_path)
            
            # Use data organizer naming convention if sample_name provided
            if sample_name:
                output_file = output_path / f"{sample_name}_tissue_mask.npy"
            else:
                # Fallback to simple naming
                base_name = Path(geojson_path).stem
                output_file = output_path / f"{base_name}_tissue_mask.npy"
            
            output_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_file, mask)
            logger.info(f"Saved tissue mask: {output_file}")
        
        return mask
        
    except Exception as e:
        logger.error(f"Error converting GeoJSON to mask: {e}")
        raise


def arguments_parser():
    parser = argparse.ArgumentParser(description="Convert QuPath GeoJSON to NPY segmentation mask.")
    
    parser.add_argument("geojson_path", type=str, help="Path to the QuPath .geojson file.")
    parser.add_argument("output_dir", type=str, help="Directory to save output masks.")
    parser.add_argument("image_width", type=int, help="Width of the original image.")
    parser.add_argument("image_height", type=int, help="Height of the original image.")
    parser.add_argument("--inner_holes", action="store_true", help="Include inner holes in the mask.")
    parser.add_argument("--downsample_factor", type=int, default=4, help="Downsampling factor (default: 4).")

    return parser

def generate_segmentation_mask(
    geojson_path: str,
    output_dir: str,
    image_width: int, 
    image_height: int,
    inner_holes: bool = True,
    downsample_factor: float = 0.1,
    erosion_strategy: str = "before_upscaling",
    erosion_radius: int = None,
    erosion_downsample_factor: Optional[float] = None,
    erosion_method: str = "cv2",
    sample_name: Optional[str] = None,
    save_intermediate: bool = True
) -> Tuple[bool, Optional[np.ndarray]]:
    '''
    Read QuPath GeoJSON and generate segmentation mask with flexible erosion strategies.
    
    Parameters:
    -----------
    geojson_path : str
        Path to the QuPath .geojson file
    output_dir : str
        Directory to save the masks
    image_width, image_height : int
        Width and height of the original image
    downsample_factor : float
        Downsample factor for initial mask generation (e.g., 0.1 means 10% of original size)
    inner_holes : bool
        Whether to include inner holes in the mask
    erosion_strategy : str
        "before_upscaling": Downsample -> Erosion -> Upscale (fast, less accurate)
        "after_upscaling": Downsample -> Upscale -> Erosion (slow, more accurate)
        "none": No erosion
    erosion_radius : int, optional
        Erosion radius in pixels (at the scale where erosion is applied)
    erosion_downsample_factor : float, optional
        Separate downsample factor specifically for erosion operation.
        If provided, the mask will be downsampled to this factor before erosion,
        then upscaled back. This allows faster erosion on large images.
        E.g., 0.25 means erode at 25% resolution
    erosion_method : str
        Method for erosion: "cv2" (default), "edt" (Euclidean distance transform), "gpu" (GPU-accelerated)
    sample_name : str, optional
        Sample name for data organizer naming convention
    save_intermediate : bool
        Whether to save intermediate masks (default: True)
    
    Returns:
    --------
    Tuple[bool, Optional[np.ndarray]]
        (success, final_mask) - success indicates if operation completed, final_mask is the result
    '''
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Generate initial mask at target resolution
        if downsample_factor == 1.0:
            # No downsampling case - generate at full resolution
            target_width = image_width
            target_height = image_height
        else:
            # Downsampled case
            target_width = int(image_width * downsample_factor)
            target_height = int(image_height * downsample_factor)
        
        logger.info(f"Generating mask at {target_width}x{target_height} (downsample_factor={downsample_factor})")
        
        # Use geojson2npy to generate the initial mask
        mask = geojson2npy(
            geojson_path=geojson_path,
            width=target_width,
            height=target_height,
            inner_holes=inner_holes,
            save_npy=False  # We'll handle saving ourselves
        )
        
        # Determine base name for saving
        if sample_name:
            base_name = sample_name
        else:
            base_name = Path(geojson_path).stem
        
        # Step 2: Apply erosion strategy
        if erosion_strategy == "before_upscaling" and erosion_radius is not None and erosion_radius > 0:
            # Strategy 1: Erode at downsampled resolution
            
            # Check if we need to further downsample for erosion
            if erosion_downsample_factor is not None and erosion_downsample_factor < 1.0:
                # Additional downsampling specifically for erosion
                erosion_width = int(image_width * erosion_downsample_factor)
                erosion_height = int(image_height * erosion_downsample_factor)
                
                logger.info(f"Downsampling mask for erosion to {erosion_width}x{erosion_height} (factor={erosion_downsample_factor})")
                
                # Downsample the mask for erosion
                erosion_mask = cv2.resize(mask, (erosion_width, erosion_height), 
                                         interpolation=cv2.INTER_NEAREST)
                
                # Scale erosion radius by the erosion downsample factor
                scaled_radius = max(1, int(erosion_radius * erosion_downsample_factor))
                logger.info(f"Applying erosion with radius {scaled_radius} at erosion resolution")
                
                # Apply erosion
                if erosion_method == "cv2":
                    from lmtools.compute.morphology import erode_binary_mask_2D
                    mask_binary = (erosion_mask > 0)
                    eroded_mask = erode_binary_mask_2D(mask_binary, scaled_radius)
                    erosion_mask = eroded_mask.astype(np.uint8) * 255
                elif erosion_method == "edt":
                    from lmtools.compute.morphology import erode_mask_2D_with_dt
                    mask_binary = (erosion_mask > 0)
                    eroded_mask = erode_mask_2D_with_dt(mask_binary, scaled_radius)
                    erosion_mask = eroded_mask.astype(np.uint8) * 255
                elif erosion_method == "gpu":
                    from lmtools.compute.morphology import erode_mask_2D_gpu_edt, check_gpu_available
                    if not check_gpu_available():
                        logger.warning("GPU not available, falling back to EDT method")
                        from lmtools.compute.morphology import erode_mask_2D_with_dt
                        mask_binary = (erosion_mask > 0)
                        eroded_mask = erode_mask_2D_with_dt(mask_binary, scaled_radius)
                    else:
                        mask_binary = (erosion_mask > 0)
                        eroded_mask = erode_mask_2D_gpu_edt(mask_binary, scaled_radius)
                    erosion_mask = eroded_mask.astype(np.uint8) * 255
                else:
                    raise ValueError(f"Unsupported erosion method: {erosion_method}")
                
                # Upscale eroded mask back to the original downsampled size
                mask = cv2.resize(erosion_mask, (target_width, target_height),
                                 interpolation=cv2.INTER_NEAREST)
                
                # Save intermediate if requested
                if save_intermediate:
                    intermediate_path = output_path / f"{base_name}_x{erosion_downsample_factor}_eroded{erosion_radius}.npy"
                    np.save(intermediate_path, erosion_mask)
                    logger.info(f"Saved intermediate eroded mask: {intermediate_path}")
                    
            else:
                # Original behavior: erode at current downsample resolution
                if downsample_factor != 1.0:
                    # Scale erosion radius by downsample factor
                    scaled_radius = max(1, int(erosion_radius * downsample_factor))
                    logger.info(f"Applying erosion with radius {scaled_radius} at downsampled resolution")
                else:
                    # No downsampling, use original radius
                    scaled_radius = erosion_radius
                    logger.info(f"Applying erosion with radius {scaled_radius} at full resolution")
                
                # Apply erosion
                if erosion_method == "cv2":
                    from lmtools.compute.morphology import erode_binary_mask_2D
                    mask_binary = (mask > 0)
                    eroded_mask = erode_binary_mask_2D(mask_binary, scaled_radius)
                    mask = eroded_mask.astype(np.uint8) * 255
                elif erosion_method == "edt":
                    from lmtools.compute.morphology import erode_mask_2D_with_dt
                    mask_binary = (mask > 0)
                    eroded_mask = erode_mask_2D_with_dt(mask_binary, scaled_radius)
                    mask = eroded_mask.astype(np.uint8) * 255
                elif erosion_method == "gpu":
                    from lmtools.compute.morphology import erode_mask_2D_gpu_edt, check_gpu_available
                    if not check_gpu_available():
                        logger.warning("GPU not available, falling back to EDT method")
                        from lmtools.compute.morphology import erode_mask_2D_with_dt
                        mask_binary = (mask > 0)
                        eroded_mask = erode_mask_2D_with_dt(mask_binary, scaled_radius)
                    else:
                        mask_binary = (mask > 0)
                        eroded_mask = erode_mask_2D_gpu_edt(mask_binary, scaled_radius)
                    mask = eroded_mask.astype(np.uint8) * 255
                else:
                    raise ValueError(f"Unsupported erosion method: {erosion_method}")
                
                # Save intermediate if requested
                if save_intermediate:
                    intermediate_path = output_path / f"{base_name}_x{downsample_factor}_eroded{erosion_radius}.npy"
                    np.save(intermediate_path, mask)
                    logger.info(f"Saved intermediate eroded mask: {intermediate_path}")
        
        # Step 3: Upscale to full resolution if needed
        if downsample_factor != 1.0:
            logger.info(f"Upscaling mask to {image_width}x{image_height}")
            # Note: cv2.resize takes (width, height) not (height, width)
            full_res_mask = cv2.resize(mask, (image_width, image_height), 
                                      interpolation=cv2.INTER_NEAREST)
            # Ensure binary mask
            full_res_mask = (full_res_mask > 127).astype(np.uint8) * 255
        else:
            full_res_mask = mask
        
        # Step 4: Apply erosion after upscaling if requested
        if erosion_strategy == "after_upscaling" and erosion_radius is not None and erosion_radius > 0:
            
            # Check if we should downsample for erosion
            if erosion_downsample_factor is not None and erosion_downsample_factor < 1.0:
                # Downsample -> Erode -> Upscale approach for performance
                erosion_width = int(image_width * erosion_downsample_factor)
                erosion_height = int(image_height * erosion_downsample_factor)
                
                logger.info(f"Downsampling full-res mask for erosion to {erosion_width}x{erosion_height}")
                
                # Downsample for erosion
                erosion_mask = cv2.resize(full_res_mask, (erosion_width, erosion_height),
                                         interpolation=cv2.INTER_NEAREST)
                
                # Scale erosion radius
                scaled_radius = max(1, int(erosion_radius * erosion_downsample_factor))
                logger.info(f"Applying erosion with radius {scaled_radius} at erosion resolution")
                
                # Apply erosion
                if erosion_method == "cv2":
                    from lmtools.compute.morphology import erode_binary_mask_2D
                    mask_binary = (erosion_mask > 0)
                    eroded_mask = erode_binary_mask_2D(mask_binary, scaled_radius)
                    erosion_mask = eroded_mask.astype(np.uint8) * 255
                elif erosion_method == "edt":
                    from lmtools.compute.morphology import erode_mask_2D_with_dt
                    mask_binary = (erosion_mask > 0)
                    eroded_mask = erode_mask_2D_with_dt(mask_binary, scaled_radius)
                    erosion_mask = eroded_mask.astype(np.uint8) * 255
                elif erosion_method == "gpu":
                    from lmtools.compute.morphology import erode_mask_2D_gpu_edt, check_gpu_available
                    if not check_gpu_available():
                        logger.warning("GPU not available, falling back to EDT method")
                        from lmtools.compute.morphology import erode_mask_2D_with_dt
                        mask_binary = (erosion_mask > 0)
                        eroded_mask = erode_mask_2D_with_dt(mask_binary, scaled_radius)
                    else:
                        mask_binary = (erosion_mask > 0)
                        eroded_mask = erode_mask_2D_gpu_edt(mask_binary, scaled_radius)
                    erosion_mask = eroded_mask.astype(np.uint8) * 255
                else:
                    raise ValueError(f"Unsupported erosion method: {erosion_method}")
                
                # Upscale back to full resolution
                full_res_mask = cv2.resize(erosion_mask, (image_width, image_height),
                                          interpolation=cv2.INTER_NEAREST)
                # Ensure binary
                full_res_mask = (full_res_mask > 127).astype(np.uint8) * 255
                
                if save_intermediate:
                    intermediate_path = output_path / f"{base_name}_erosion_x{erosion_downsample_factor}_r{erosion_radius}.npy"
                    np.save(intermediate_path, erosion_mask)
                    logger.info(f"Saved erosion intermediate: {intermediate_path}")
                    
            else:
                # Original behavior: erode at full resolution
                logger.info(f"Applying erosion with radius {erosion_radius} at full resolution")
                
                # ADD MORE FUNCTIONS FOR EROSION
                if erosion_method == "cv2":
                    from lmtools.compute.morphology import erode_binary_mask_2D
                    mask_binary = (full_res_mask > 0)
                    eroded_mask = erode_binary_mask_2D(mask_binary, erosion_radius)
                    full_res_mask = eroded_mask.astype(np.uint8) * 255
                elif erosion_method == "edt":
                    from lmtools.compute.morphology import erode_mask_2D_with_dt
                    mask_binary = (full_res_mask > 0)
                    eroded_mask = erode_mask_2D_with_dt(mask_binary, erosion_radius)
                    full_res_mask = eroded_mask.astype(np.uint8) * 255
                elif erosion_method == "gpu":
                    from lmtools.compute.morphology import erode_mask_2D_gpu_edt, check_gpu_available
                    if not check_gpu_available():
                        logger.warning("GPU not available, falling back to EDT method")
                        from lmtools.compute.morphology import erode_mask_2D_with_dt
                        mask_binary = (full_res_mask > 0)
                        eroded_mask = erode_mask_2D_with_dt(mask_binary, erosion_radius)
                    else:
                        mask_binary = (full_res_mask > 0)
                        eroded_mask = erode_mask_2D_gpu_edt(mask_binary, erosion_radius)
                    full_res_mask = eroded_mask.astype(np.uint8) * 255
                else:
                    raise ValueError(f"Unsupported erosion method: {erosion_method}")
        
        # Step 5: Save final mask
        if erosion_radius and erosion_radius > 0:
            final_path = output_path / f"{base_name}_tissue_mask_eroded{erosion_radius}.npy"
        else:
            final_path = output_path / f"{base_name}_tissue_mask.npy"
        
        np.save(final_path, full_res_mask)
        logger.info(f"Saved final mask: {final_path}")
        
        return True, full_res_mask
    
    except Exception as e:
        logger.error(f"Error generating segmentation mask: {e}")
        return False, None

if __name__ == "__main__":
    '''
    Usage:
        1. Create a python environment with cv2 and PIL
        2. Run the script with the following command:
        ```bash
            python generate_mask.py <geojson_path> <output_dir> <image_width> <image_height> [options]
        ```        
    '''
    parser = argparse.ArgumentParser(description="Convert QuPath GeoJSON to NPY segmentation mask.")
    
    parser.add_argument("geojson_path", type=str, help="Path to the QuPath .geojson file.")
    parser.add_argument("output_dir", type=str, help="Directory to save output masks.")
    parser.add_argument("image_width", type=int, help="Width of the original image.")
    parser.add_argument("image_height", type=int, help="Height of the original image.")
    parser.add_argument("--inner_holes", action="store_true", help="Include inner holes in the mask.")
    parser.add_argument("--downsample_factor", type=float, default=0.1, help="Downsampling factor (default: 0.1).")
    parser.add_argument("--erosion_strategy", type=str, default="before_upscaling", 
                       choices=["before_upscaling", "after_upscaling", "none"],
                       help="Erosion strategy (default: before_upscaling)")
    parser.add_argument("--erosion_radius", type=int, default=None, help="Erosion radius in pixels.")
    parser.add_argument("--erosion_downsample_factor", type=float, default=None, 
                       help="Downsample factor for erosion operation (e.g., 0.25 for 25% resolution)")
    parser.add_argument("--erosion_method", type=str, default="cv2", 
                       choices=["cv2", "edt", "gpu"],
                       help="Erosion method: cv2, edt, or gpu (default: cv2)")
    parser.add_argument("--sample_name", type=str, default=None, help="Sample name for output naming.")
    parser.add_argument("--no_intermediate", action="store_true", help="Don't save intermediate masks.")

    args = parser.parse_args()

    success, mask = generate_segmentation_mask(
        geojson_path=args.geojson_path,
        output_dir=args.output_dir,
        image_width=args.image_width,
        image_height=args.image_height,
        inner_holes=args.inner_holes,
        downsample_factor=args.downsample_factor,
        erosion_strategy=args.erosion_strategy,
        erosion_radius=args.erosion_radius,
        erosion_downsample_factor=args.erosion_downsample_factor,
        erosion_method=args.erosion_method,
        sample_name=args.sample_name,
        save_intermediate=not args.no_intermediate
    )
    
    if success:
        print("Mask generation completed successfully!")
    else:
        print("Mask generation failed!")