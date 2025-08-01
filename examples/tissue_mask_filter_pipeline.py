from pathlib import Path
from lmtools.io import create_data_paths_from_organized, ProcessingStep
from lmtools.seg import (
    tissue_mask_filter_by_overlap,
    count_cells,
    relabel_sequential_labels
)
from datetime import datetime
import pandas as pd
from tqdm import tqdm


def process_all_tissue_mask_filter(
    organized_output_dir: str,
    mask_to_filter: str = 'cy3_cy5_overlap_filtered',  # or 'cy5_filtered_relabeled'
    experiment_name: str = 'Tissue Mask Filtering',
    erosion_radius: int = 50,
    erosion_downsample_factor: float = 1.0,
    erosion_method: str = 'edt',
    min_overlap_ratio: float = 0.99,
    save_filtered_masks: bool = True
):
    """
    Process all samples in organized output directory with tissue mask filtering.
    This function is designed to work on already filtered masks from other pipelines.
    
    Parameters:
    -----------
    organized_output_dir : str
        Path to organized output directory containing samples folder.
        Expected structure: organized_output_dir/samples/Sample_Name/
        
    mask_to_filter : str
        Name of the processed mask to filter. Common examples:
        - 'cy3_cy5_overlap_filtered': Output from cy3_dapi_cy5_filter_pipeline
        - 'cy5_filtered_relabeled': Output from cy5_dapi_filter_pipeline
        - 'cy3_filtered_relabeled': Output from cy3-only pipeline
        - Any custom mask name saved in the results directory
        
    experiment_name : str
        Name for the experiment (used in metadata tracking)
        
    erosion_radius : int
        Radius for tissue mask erosion in pixels (default: 50). This helps to:
        - Remove cells at the tissue edge that might be artifacts
        - Focus on cells well within the tissue boundary
        - Default of 50 pixels provides good edge artifact removal
        - Adjust based on your image resolution and needs
        
    erosion_downsample_factor : float
        Downsample factor for erosion operation (default: 1.0).
        - 1.0: Full resolution erosion (most accurate, default)
        - 0.25: Perform erosion at 25% resolution (16x faster)
        - 0.1: Perform erosion at 10% resolution (100x faster)
        - Lower values trade accuracy for speed
        
    erosion_method : str
        Method for erosion operation (default: 'edt'):
        - 'cv2': Traditional morphological erosion (accurate but slow for large radii)
        - 'edt': Euclidean distance transform (default, fast and accurate)
        - 'gpu': GPU-accelerated EDT (fastest, requires CuPy)
        
    min_overlap_ratio : float
        Minimum overlap ratio between cell and tissue mask to keep the cell.
        - 0.99: Cell must be 99% within tissue (very strict)
        - 0.95: Cell must be 95% within tissue (slightly relaxed)
        - 0.90: Cell must be 90% within tissue (more permissive)
        - Higher values = more conservative filtering
        
    save_filtered_masks : bool
        Whether to save the tissue-filtered masks to disk.
        Output will be saved as: {sample_name}_{mask_to_filter}_tissue_filtered.npy
    
    Returns:
    --------
    pd.DataFrame
        Summary of processing results with columns:
        - sample: Sample name
        - status: Processing status (success/error/mask_not_found)
        - initial_cells: Number of cells before filtering
        - tissue_filtered_cells: Number of cells after filtering
        - cells_removed_by_tissue: Number of cells removed
        - percent_retained: Percentage of cells retained
        - has_tissue_mask: Whether tissue mask was found
        - erosion_radius: Erosion radius used
        - min_overlap_ratio: Overlap threshold used
        - tissue_area_pixels: Area of the (eroded) tissue mask in pixels
        - average_density_per_pixel: Average cell density (cells/pixel)
        - average_density_per_100x100: Average cell density per 100x100 area
    
    Notes:
    ------
    - Requires tissue masks to be organized using organize_data(step=3)
    - Tissue masks should be GeoJSON files from QuPath
    - The function will skip samples without tissue masks
    - Results are saved as CSV for further analysis
    """
    
    samples_dir = Path(organized_output_dir) / "samples"
    sample_dirs = [d for d in samples_dir.iterdir() if d.is_dir()]
    
    results = []
    
    print(f"Found {len(sample_dirs)} samples to process")
    print(f"Filtering mask: {mask_to_filter}")
    print(f"Tissue erosion radius: {erosion_radius} pixels")
    print(f"Erosion method: {erosion_method}")
    if erosion_downsample_factor:
        print(f"Erosion downsample factor: {erosion_downsample_factor}")
    print(f"Min overlap ratio: {min_overlap_ratio}")
    
    for sample_dir in tqdm(sample_dirs, desc="Processing samples"):
        sample_name = sample_dir.name
        print(f"\n{'='*60}")
        print(f"Processing: {sample_name}")
        print(f"{'='*60}")
        
        try:
            # Create DataPaths instance
            data_paths = create_data_paths_from_organized(
                organized_sample_dir=str(sample_dir),
                experiment_name=experiment_name,
                notes=f'Tissue mask filtering for {mask_to_filter} in sample: {sample_name}'
            )
            
            # Load the processed mask to filter
            processed_mask_path = data_paths.output_dir / f"{sample_name}_{mask_to_filter}.npy"
            
            if not processed_mask_path.exists():
                print(f"WARNING: Processed mask {sample_name}_{mask_to_filter}.npy not found, skipping sample")
                result = {
                    'sample': sample_name,
                    'status': 'mask_not_found',
                    'error': f'{sample_name}_{mask_to_filter}.npy not found',
                    'input_mask': mask_to_filter,
                    'initial_cells': 0,
                    'tissue_filtered_cells': 0,
                    'cells_removed_by_tissue': 0,
                    'percent_retained': 0.0,
                    'tissue_area_pixels': 0,
                    'average_density_per_pixel': 0.0,
                    'average_density_per_100x100': 0.0
                }
                results.append(result)
                continue
            
            # Load the mask
            import numpy as np
            seg_mask = np.load(processed_mask_path)
            
            # Count initial cells
            initial_count = count_cells(seg_mask)
            print(f"Initial cell count: {initial_count}")
            
            # Check for tissue mask
            tissue_mask_path = data_paths.get_tissue_mask_path()
            
            if tissue_mask_path and tissue_mask_path.exists():
                print(f"Found tissue mask: {tissue_mask_path.name}")
                has_tissue_mask = True
            else:
                print(f"WARNING: No tissue mask found for {sample_name}")
                has_tissue_mask = False
                tissue_mask_path = None
            
            # Apply tissue mask filtering if available
            if has_tissue_mask and initial_count > 0:
                print(f"Applying tissue mask filter...")
                
                # Generate tissue mask with erosion using the same logic as tissue_mask_filter_by_overlap
                from lmtools.seg.generate_mask import generate_segmentation_mask
                
                # Save the eroded tissue mask to the output directory
                success, tissue_mask_eroded = generate_segmentation_mask(
                    geojson_path=str(tissue_mask_path),
                    output_dir=str(data_paths.output_dir),  # Save to results folder
                    image_width=seg_mask.shape[1],
                    image_height=seg_mask.shape[0],
                    inner_holes=True,
                    downsample_factor=1.0,  # Full resolution
                    erosion_strategy="after_upscaling" if erosion_downsample_factor is None or erosion_downsample_factor == 1.0 else "before_upscaling",
                    erosion_radius=erosion_radius if erosion_radius > 0 else None,
                    erosion_downsample_factor=erosion_downsample_factor,
                    erosion_method=erosion_method,
                    sample_name=sample_name,  # Use sample name for consistent naming
                    save_intermediate=False
                )
                
                if not success:
                    raise ValueError("Failed to generate tissue mask")
                
                # The mask is already saved by generate_segmentation_mask
                if erosion_radius and erosion_radius > 0:
                    print(f"Saved eroded tissue mask: {sample_name}_tissue_mask_eroded{erosion_radius}.npy")
                else:
                    print(f"Saved tissue mask: {sample_name}_tissue_mask.npy")
                
                # Calculate tissue area (number of pixels)
                tissue_area_pixels = np.sum(tissue_mask_eroded > 0)
                
                filtered_mask = tissue_mask_filter_by_overlap(
                    seg_mask=seg_mask,
                    tissue_geojson_path=tissue_mask_path,
                    img_shape=seg_mask.shape,
                    downsample_factor=1.0,
                    erosion_radius=erosion_radius,
                    erosion_downsample_factor=erosion_downsample_factor,
                    erosion_method=erosion_method,
                    min_overlap_ratio=min_overlap_ratio,
                    data_paths=data_paths,
                    step_name=f"tissue_mask_filter_{mask_to_filter}"
                )
                
                # Relabel sequentially
                filtered_mask = relabel_sequential_labels(filtered_mask)
                
                # Count filtered cells
                filtered_count = count_cells(filtered_mask)
                cells_removed = initial_count - filtered_count
                percent_retained = (filtered_count / initial_count * 100) if initial_count > 0 else 0
                
                # Calculate average density (cells per pixel)
                average_density = filtered_count / tissue_area_pixels if tissue_area_pixels > 0 else 0
                
                print(f"After tissue filtering: {filtered_count} cells ({percent_retained:.1f}% retained)")
                print(f"Tissue area: {tissue_area_pixels:,} pixels")
                print(f"Average density: {average_density:.6f} cells/pixel ({average_density * 10000:.2f} cells per 100x100 area)")
                
                # Save filtered mask
                if save_filtered_masks:
                    output_name = f"{mask_to_filter}_tissue_filtered"
                    data_paths.save_processed_mask(
                        mask=filtered_mask,
                        name=output_name,
                        processing_info=ProcessingStep(
                            step_name=f"save_{output_name}",
                            timestamp=datetime.now().isoformat(),
                            parameters={
                                "input_mask": mask_to_filter,
                                "tissue_mask": str(tissue_mask_path),
                                "erosion_radius": erosion_radius,
                                "erosion_method": erosion_method,
                                "erosion_downsample_factor": erosion_downsample_factor,
                                "min_overlap_ratio": min_overlap_ratio,
                                "initial_cells": initial_count,
                                "filtered_cells": filtered_count,
                                "cells_removed": cells_removed,
                                "percent_retained": percent_retained,
                                "tissue_area_pixels": tissue_area_pixels,
                                "average_density": average_density
                            },
                            input_data=[mask_to_filter, "tissue_mask"],
                            notes=f"Applied tissue mask filter to {mask_to_filter}"
                        )
                    )
                    print(f"Saved: {output_name}.npy")
            else:
                # No tissue mask or no cells, keep original counts
                filtered_count = initial_count
                cells_removed = 0
                percent_retained = 100.0 if initial_count > 0 else 0
                tissue_area_pixels = 0
                average_density = 0.0
                
                if not has_tissue_mask:
                    print("No tissue filtering applied (no tissue mask)")
                if initial_count == 0:
                    print("No cells to filter")
            
            # Save metadata
            metadata_path = data_paths.save_metadata()
            
            # Collect results
            result = {
                'sample': sample_name,
                'status': 'success',
                'error': None,
                'input_mask': mask_to_filter,
                'has_tissue_mask': has_tissue_mask,
                'tissue_mask_path': str(tissue_mask_path) if tissue_mask_path else None,
                'erosion_radius': erosion_radius,
                'erosion_method': erosion_method,
                'erosion_downsample_factor': erosion_downsample_factor,
                'min_overlap_ratio': min_overlap_ratio,
                'initial_cells': initial_count,
                'tissue_filtered_cells': filtered_count,
                'cells_removed_by_tissue': cells_removed,
                'percent_retained': percent_retained,
                'tissue_area_pixels': tissue_area_pixels,
                'average_density_per_pixel': average_density,
                'average_density_per_100x100': average_density * 10000 if tissue_area_pixels > 0 else 0,
                'metadata_path': str(metadata_path)
            }
            
        except Exception as e:
            print(f"ERROR: {str(e)}")
            result = {
                'sample': sample_name,
                'status': 'error',
                'error': str(e),
                'input_mask': mask_to_filter,
                'initial_cells': 0,
                'tissue_filtered_cells': 0,
                'cells_removed_by_tissue': 0,
                'percent_retained': 0.0,
                'tissue_area_pixels': 0,
                'average_density_per_pixel': 0.0,
                'average_density_per_100x100': 0.0
            }
        
        results.append(result)
    
    # Create results dataframe
    df = pd.DataFrame(results)
    
    # Save results CSV
    results_path = Path(organized_output_dir) / f"tissue_filtering_results_{mask_to_filter}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")
    
    successful = df[df['status'] == 'success']
    print(f"\nTotal samples: {len(df)}")
    print(f"Successful: {len(successful)}")
    print(f"Samples with tissue masks: {df['has_tissue_mask'].sum()}")
    
    # Statistics for samples with tissue masks
    tissue_filtered = df[df['has_tissue_mask'] == True]
    if len(tissue_filtered) > 0:
        total_initial = tissue_filtered['initial_cells'].sum()
        total_filtered = tissue_filtered['tissue_filtered_cells'].sum()
        total_removed = tissue_filtered['cells_removed_by_tissue'].sum()
        total_tissue_area = tissue_filtered['tissue_area_pixels'].sum()
        
        print(f"\nFor samples with tissue masks:")
        print(f"  Total initial cells: {total_initial:,}")
        print(f"  Total after tissue filtering: {total_filtered:,}")
        print(f"  Total cells removed: {total_removed:,}")
        print(f"  Total tissue area: {total_tissue_area:,} pixels")
        if total_initial > 0:
            print(f"  Overall retention rate: {(total_filtered/total_initial*100):.1f}%")
        
        # Average density statistics
        avg_density_per_sample = tissue_filtered['average_density_per_100x100'].mean()
        std_density_per_sample = tissue_filtered['average_density_per_100x100'].std()
        print(f"\nAverage density across samples:")
        print(f"  Mean: {avg_density_per_sample:.2f} ± {std_density_per_sample:.2f} cells per 100x100 area")
        print(f"  Range: {tissue_filtered['average_density_per_100x100'].min():.2f} - {tissue_filtered['average_density_per_100x100'].max():.2f}")
    
    return df


# Usage examples
if __name__ == "__main__":
    # Example 1: Default settings (recommended for most cases)
    df = process_all_tissue_mask_filter(
        organized_output_dir='/path/to/organized/output',
        mask_to_filter='cy3_cy5_overlap_filtered',
        experiment_name='Tissue Mask Filtering',
        erosion_radius=50,                    # Default: removes cells within 50 pixels of tissue edge
        erosion_downsample_factor=1.0,        # Default: full resolution (most accurate)
        erosion_method='edt',                 # Default: EDT method (fast and accurate)
        min_overlap_ratio=0.99,               # Default: cells must be 99% within tissue
        save_filtered_masks=True              # Default: save the filtered masks
    )
    
    # Example 2: Faster processing for large images with downsampling
    # df = process_all_tissue_mask_filter(
    #     organized_output_dir='/path/to/organized/output',
    #     mask_to_filter='cy5_filtered_relabeled',
    #     experiment_name='Tissue Mask Filtering - Fast',
    #     erosion_radius=100,                   # Larger erosion radius
    #     erosion_downsample_factor=0.25,       # 16x faster erosion at 25% resolution
    #     erosion_method='edt',                 # EDT method
    #     min_overlap_ratio=0.99,               # Keep strict overlap requirement
    #     save_filtered_masks=True              # Save results
    # )
    
    # Example 3: Maximum speed with GPU acceleration
    # df = process_all_tissue_mask_filter(
    #     organized_output_dir='/path/to/organized/output',
    #     mask_to_filter='cy3_filtered_relabeled',
    #     experiment_name='Tissue Mask Filtering - GPU',
    #     erosion_radius=200,                   # Very large erosion radius
    #     erosion_downsample_factor=0.1,        # 100x faster at 10% resolution
    #     erosion_method='gpu',                 # GPU acceleration (requires CUDA)
    #     min_overlap_ratio=0.95,               # Slightly relaxed overlap requirement
    #     save_filtered_masks=True              # Save results
    # )


'''
Example workflow:
  from lmtools.io import organize_data
  from tissue_mask_filter_pipeline import process_all_tissue_mask_filter

  # Step 1: Run existing pipeline (e.g., cy3_dapi_cy5_filter_pipeline)

  # Step 2: Organize tissue masks (step 3)
  mask_df = organize_data(
      source_dir='/path/to/original/data',
      output_dir='/path/to/organized/output',
      step=3,
      qupath_dir='/path/to/tissue/masks'
  )

  # Step 3: Apply tissue mask filtering to the results
  df = process_all_tissue_mask_filter(
      organized_output_dir='/path/to/organized/output',
      mask_to_filter='cy3_cy5_overlap_filtered',  # from cy3_dapi_cy5 pipeline
      experiment_name='Tissue Mask Filtering',
      erosion_radius=50,                        # Remove cells within 50 pixels of edge
      erosion_downsample_factor=1.0,            # Full resolution erosion
      erosion_method='edt',                     # EDT method (fast)
      min_overlap_ratio=0.99,                   # 99% overlap required
      save_filtered_masks=True                  # Save the results
  )

Erosion Method Performance Guide:
---------------------------------
1. 'edt' (default): Euclidean Distance Transform
   - 10-100x faster than cv2 for large radii
   - Excellent balance of speed and accuracy
   - Recommended for most use cases
   - Default erosion radius of 50 pixels works very well with EDT

2. 'cv2': Traditional morphological erosion
   - Most accurate for very small radii (<20 pixels)
   - Can be slow for large radii
   - Use when pixel-perfect accuracy is critical

3. 'gpu': GPU-accelerated EDT
   - Fastest option (requires CUDA GPU + CuPy)
   - 10-30x faster than CPU EDT
   - Ideal for very large images or batch processing
   - Automatically falls back to EDT if GPU unavailable

Erosion Downsample Factor:
-------------------------
- 1.0 (default): Erosion at full tissue mask resolution (most accurate)
- 0.25: Erosion at 25% resolution (16x faster)
- 0.1: Erosion at 10% resolution (100x faster)
- Trade-off: Lower values = faster but less accurate boundaries
- Recommendation: Use 1.0 for accuracy, 0.25 for balanced speed/accuracy
'''