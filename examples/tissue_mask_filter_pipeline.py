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
    erosion_radius: int = 10,
    erosion_downsample_factor: float = None,
    erosion_method: str = 'cv2',
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
        Radius for tissue mask erosion in pixels. This helps to:
        - Remove cells at the tissue edge that might be artifacts
        - Focus on cells well within the tissue boundary
        - Typical values: 5-20 pixels depending on resolution
        - Higher values = more conservative (remove more edge cells)
        
    erosion_downsample_factor : float, optional
        Downsample factor for erosion operation to speed up processing.
        - None: Use same resolution as mask generation
        - 0.25: Perform erosion at 25% resolution (4x faster)
        - Useful for large images with large erosion radii
        
    erosion_method : str
        Method for erosion operation:
        - 'cv2': Traditional morphological erosion (most accurate)
        - 'edt': Euclidean distance transform (faster for large radii)
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
                    'percent_retained': 0.0
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
                
                print(f"After tissue filtering: {filtered_count} cells ({percent_retained:.1f}% retained)")
                
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
                                "percent_retained": percent_retained
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
                'percent_retained': 0.0
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
        
        print(f"\nFor samples with tissue masks:")
        print(f"  Total initial cells: {total_initial:,}")
        print(f"  Total after tissue filtering: {total_filtered:,}")
        print(f"  Total cells removed: {total_removed:,}")
        if total_initial > 0:
            print(f"  Overall retention rate: {(total_filtered/total_initial*100):.1f}%")
    
    return df


# Usage examples
if __name__ == "__main__":
    # Example 1: Filter cy3_cy5_overlap_filtered masks (from cy3_dapi_cy5_filter_pipeline)
    df = process_all_tissue_mask_filter(
        organized_output_dir='/path/to/organized/output',
        mask_to_filter='cy3_cy5_overlap_filtered',
        erosion_radius=10,
        min_overlap_ratio=0.99
    )
    
    # Example 2: Filter with GPU acceleration for large erosion radius
    # df = process_all_tissue_mask_filter(
    #     organized_output_dir='/path/to/organized/output',
    #     mask_to_filter='cy5_filtered_relabeled',
    #     erosion_radius=100,
    #     erosion_method='gpu',  # Use GPU acceleration
    #     min_overlap_ratio=0.99
    # )
    
    # Example 3: Filter with erosion downsampling for performance
    # df = process_all_tissue_mask_filter(
    #     organized_output_dir='/path/to/organized/output',
    #     mask_to_filter='my_custom_filtered_mask',
    #     erosion_radius=200,
    #     erosion_downsample_factor=0.25,  # Erode at 25% resolution
    #     erosion_method='edt',  # Use EDT method
    #     min_overlap_ratio=0.95
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
      erosion_radius=10,
      min_overlap_ratio=0.99
  )

Erosion Method Performance Guide:
---------------------------------
1. 'cv2' (default): Traditional morphological erosion
   - Most accurate for small radii (<50 pixels)
   - Can be slow for large radii
   - Good for small images or when accuracy is critical

2. 'edt': Euclidean Distance Transform
   - 10-100x faster than cv2 for large radii
   - Good balance of speed and accuracy
   - Recommended for radii > 50 pixels

3. 'gpu': GPU-accelerated EDT
   - Fastest option (requires CUDA GPU + CuPy)
   - 10-30x faster than CPU EDT
   - Ideal for large images and radii
   - Falls back to EDT if GPU unavailable

Erosion Downsample Factor:
-------------------------
- None: Erosion at full tissue mask resolution
- 0.25: Erosion at 25% resolution (16x faster)
- 0.1: Erosion at 10% resolution (100x faster)
- Trade-off: Lower values = faster but less accurate boundaries
'''