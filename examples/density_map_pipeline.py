from pathlib import Path
from lmtools.io import create_data_paths_from_organized, ProcessingStep
from lmtools.seg import (
    generate_cell_density_heatmap,
    generate_cell_distribution_plot,
    count_cells
)
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import numpy as np


def process_all_density_maps(
    organized_output_dir: str,
    masks_to_process: list = ['cy3_cy5_overlap_filtered_tissue_filtered', 'cy5_filtered_relabeled'],
    experiment_name: str = 'Cell Density Analysis',
    bin_size: int = 100,
    smooth: bool = False,
    sigma: float = 2.0,
    generate_distribution_plots: bool = False,
    save_density_arrays: bool = False
):
    """
    Process all samples in organized output directory to generate density heatmaps.
    
    Parameters:
    -----------
    organized_output_dir : str
        Path to organized output directory containing samples folder
    masks_to_process : list
        List of mask names to process (e.g., filtered masks from other pipelines)
    experiment_name : str
        Name for the experiment
    bin_size : int
        Size of square bin in pixels for density calculation
    smooth : bool
        Whether to apply Gaussian smoothing to heatmaps
    sigma : float
        Standard deviation for Gaussian filter
    generate_distribution_plots : bool
        Whether to also generate cell distribution scatter plots
    save_density_arrays : bool
        Whether to save the density arrays as .npy files
    
    Returns:
    --------
    pd.DataFrame
        Summary of processing results for all samples
    """
    
    samples_dir = Path(organized_output_dir) / "samples"
    sample_dirs = [d for d in samples_dir.iterdir() if d.is_dir()]
    
    results = []
    
    print(f"Found {len(sample_dirs)} samples to process")
    print(f"Masks to process: {masks_to_process}")
    print(f"Bin size: {bin_size} pixels")
    print(f"Smoothing: {smooth} (sigma={sigma})" if smooth else "Smoothing: False")
    
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
                notes=f'Density map generation for sample: {sample_name}'
            )
            
            # Get tissue mask path
            tissue_mask_path = data_paths.get_tissue_mask_path()
            
            # Check if we have a tissue mask (either GeoJSON-generated or from directory)
            tissue_mask_npy = None
            tissue_mask_found = False
            
            # First check for generated tissue mask
            tissue_masks_dir = data_paths.output_dir / "tissue_masks"
            if tissue_masks_dir.exists():
                tissue_mask_files = list(tissue_masks_dir.glob("*tissue_mask*.npy"))
                if tissue_mask_files:
                    tissue_mask_npy = tissue_mask_files[0]
                    tissue_mask_found = True
                    print(f"Found tissue mask: {tissue_mask_npy.name}")
            
            # If not found, check for GeoJSON
            if not tissue_mask_found and tissue_mask_path and tissue_mask_path.exists():
                print(f"Found tissue GeoJSON: {tissue_mask_path.name}")
                tissue_mask_found = True
            
            if not tissue_mask_found:
                print(f"WARNING: No tissue mask found for {sample_name}")
            
            # Create output directories
            density_output_dir = data_paths.output_dir / "density_maps"
            density_output_dir.mkdir(exist_ok=True)
            
            if generate_distribution_plots:
                distribution_output_dir = data_paths.output_dir / "distribution_plots"
                distribution_output_dir.mkdir(exist_ok=True)
            
            # Process each mask
            sample_result = {
                'sample': sample_name,
                'has_tissue_mask': tissue_mask_found,
                'bin_size': bin_size,
                'smooth': smooth,
                'sigma': sigma,
                'status': 'pending'
            }
            
            for mask_name in masks_to_process:
                mask_path = data_paths.output_dir / f"{sample_name}_{mask_name}.npy"
                
                if not mask_path.exists():
                    print(f"  {mask_name}: Not found, skipping")
                    sample_result[f'{mask_name}_status'] = 'not_found'
                    sample_result[f'{mask_name}_cell_count'] = 0
                    continue
                
                # Load mask and count cells
                mask = np.load(mask_path)
                cell_count = count_cells(mask)
                sample_result[f'{mask_name}_cell_count'] = cell_count
                
                print(f"  {mask_name}: {cell_count} cells")
                
                if cell_count == 0:
                    print(f"    No cells to visualize")
                    sample_result[f'{mask_name}_status'] = 'no_cells'
                    continue
                
                # Generate density heatmap if tissue mask available
                if tissue_mask_npy and tissue_mask_npy.exists():
                    density_output_path = density_output_dir / f"{sample_name}_{mask_name}_density_heatmap.png"
                    
                    try:
                        density_array = generate_cell_density_heatmap(
                            tissue_mask_path=tissue_mask_npy,
                            relabeled_mask_path=mask_path,
                            bin_size=bin_size,
                            output_path=density_output_path,
                            smooth=smooth,
                            sigma=sigma,
                            figsize=(12, 10),
                            dpi=300
                        )
                        
                        print(f"    Saved density heatmap: {density_output_path.name}")
                        sample_result[f'{mask_name}_density_map'] = str(density_output_path)
                        
                        # Save density array if requested
                        if save_density_arrays:
                            density_array_path = density_output_dir / f"{sample_name}_{mask_name}_density_array.npy"
                            np.save(density_array_path, density_array)
                            sample_result[f'{mask_name}_density_array'] = str(density_array_path)
                        
                        # Calculate density statistics
                        sample_result[f'{mask_name}_max_density'] = float(density_array.max())
                        sample_result[f'{mask_name}_mean_density'] = float(density_array.mean())
                        sample_result[f'{mask_name}_density_std'] = float(density_array.std())
                        
                    except Exception as e:
                        print(f"    ERROR generating density map: {str(e)}")
                        sample_result[f'{mask_name}_density_error'] = str(e)
                
                # Generate distribution plot
                if generate_distribution_plots:
                    distribution_output_path = distribution_output_dir / f"{sample_name}_{mask_name}_distribution.png"
                    
                    try:
                        # Use tissue mask for overlay if available
                        tissue_for_plot = tissue_mask_npy if tissue_mask_npy and tissue_mask_npy.exists() else None
                        
                        stats = generate_cell_distribution_plot(
                            relabeled_mask=mask_path,
                            tissue_mask=tissue_for_plot,
                            output_path=distribution_output_path,
                            figsize=(12, 10),
                            dpi=300
                        )
                        
                        print(f"    Saved distribution plot: {distribution_output_path.name}")
                        sample_result[f'{mask_name}_distribution_plot'] = str(distribution_output_path)
                        
                        # Add distribution statistics
                        sample_result[f'{mask_name}_centroid_mean_x'] = stats['mean_x']
                        sample_result[f'{mask_name}_centroid_mean_y'] = stats['mean_y']
                        sample_result[f'{mask_name}_centroid_std_x'] = stats['std_x']
                        sample_result[f'{mask_name}_centroid_std_y'] = stats['std_y']
                        
                    except Exception as e:
                        print(f"    ERROR generating distribution plot: {str(e)}")
                        sample_result[f'{mask_name}_distribution_error'] = str(e)
                
                sample_result[f'{mask_name}_status'] = 'success'
            
            # Save metadata
            metadata_path = data_paths.save_metadata()
            sample_result['metadata_path'] = str(metadata_path)
            sample_result['status'] = 'success'
            
        except Exception as e:
            print(f"ERROR processing sample: {str(e)}")
            sample_result = {
                'sample': sample_name,
                'status': 'error',
                'error': str(e)
            }
        
        results.append(sample_result)
    
    # Create results dataframe
    df = pd.DataFrame(results)
    
    # Save results CSV
    results_path = Path(organized_output_dir) / f"density_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")
    
    successful = df[df['status'] == 'success']
    print(f"\nTotal samples: {len(df)}")
    print(f"Successful: {len(successful)}")
    print(f"Samples with tissue masks: {df['has_tissue_mask'].sum() if 'has_tissue_mask' in df.columns else 0}")
    
    # Summarize cell counts per mask type
    for mask_name in masks_to_process:
        col_name = f'{mask_name}_cell_count'
        if col_name in df.columns:
            total_cells = df[col_name].sum()
            avg_cells = df[col_name].mean()
            print(f"\n{mask_name}:")
            print(f"  Total cells across samples: {total_cells:,}")
            print(f"  Average cells per sample: {avg_cells:.1f}")
    
    return df


# Usage examples
if __name__ == "__main__":
    # Example 1: Generate density maps for tissue-filtered masks
    df = process_all_density_maps(
        organized_output_dir='/path/to/organized/output',
        masks_to_process=['cy3_cy5_overlap_filtered_tissue_filtered'],
        bin_size=100,
        smooth=True,
        sigma=2
    )
    
    # Example 2: Generate density maps for multiple mask types
    # df = process_all_density_maps(
    #     organized_output_dir='/path/to/organized/output',
    #     masks_to_process=[
    #         'cy3_cy5_overlap_filtered',
    #         'cy3_cy5_overlap_filtered_tissue_filtered',
    #         'cy5_filtered_relabeled'
    #     ],
    #     bin_size=50,  # Smaller bins for higher resolution
    #     smooth=True,
    #     sigma=1.5
    # )
    
    # Example 3: Generate without smoothing and with distribution plots
    # df = process_all_density_maps(
    #     organized_output_dir='/path/to/organized/output',
    #     masks_to_process=['cy3_filtered_relabeled'],
    #     bin_size=100,
    #     smooth=False,  # No smoothing
    #     generate_distribution_plots=True,
    #     save_density_arrays=True  # Save raw density arrays
    # )

'''
from lmtools.seg import generate_cell_density_heatmap
from density_map_pipeline import process_all_density_maps

# After running tissue filtering pipeline
# Generate density maps for tissue-filtered results
df = process_all_density_maps(
    organized_output_dir='/path/to/organized/output',
    masks_to_process=['cy3_cy5_overlap_filtered_tissue_filtered'],
    bin_size=100,
    smooth=True
  )
'''