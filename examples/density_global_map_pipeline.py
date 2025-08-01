'''
Density Map Pipeline with Global Color Scale

This pipeline generates density heatmaps for all samples using a unified color scale,
enabling direct comparison of cell density across different samples.

Prerequisites:
1. Run tissue_mask_filter_pipeline.py first to generate filtered masks
2. The results CSV from tissue filtering contains sample information

The pipeline:
1. Loads the tissue filtering results CSV
2. Calculates global min/max density across all samples
3. Generates individual density maps with the same color scale
4. Creates a comparison figure showing all samples
'''

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from datetime import datetime
import logging

from lmtools.io import create_data_paths_from_organized
from lmtools.seg.visualize import generate_cell_density_heatmap

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_global_density_range(
    organized_output_dir: str,
    results_df: pd.DataFrame,
    mask_name: str,
    bin_size: int = 100,
    smooth: bool = True,
    sigma: float = 1.5
) -> tuple:
    """
    Calculate the global min/max density values across all samples.
    
    Parameters:
    -----------
    organized_output_dir : str
        Path to organized output directory
    results_df : pd.DataFrame
        DataFrame from tissue filtering results
    mask_name : str
        Name of the mask to process (e.g., 'cy3_cy5_overlap_filtered_tissue_filtered')
    bin_size : int
        Bin size for density calculation
    smooth : bool
        Whether to apply Gaussian smoothing
    sigma : float
        Sigma for Gaussian smoothing
        
    Returns:
    --------
    tuple : (global_min, global_max, all_densities)
        Global min/max density values and list of all density arrays
    """
    all_densities = []
    density_stats = []
    
    # Filter successful samples
    successful_samples = results_df[results_df['status'] == 'success']
    
    print(f"Calculating density range for {len(successful_samples)} samples...")
    
    for _, row in tqdm(successful_samples.iterrows(), total=len(successful_samples), 
                       desc="Calculating densities"):
        sample_name = row['sample']
        sample_dir = Path(organized_output_dir) / "samples" / sample_name
        
        try:
            # Create data paths
            data_paths = create_data_paths_from_organized(
                organized_sample_dir=str(sample_dir),
                experiment_name='Density Map Analysis'
            )
            
            # Get mask paths
            tissue_mask_path = data_paths.output_dir / f"{sample_name}_tissue_mask.npy"
            cell_mask_path = data_paths.output_dir / f"{sample_name}_{mask_name}.npy"
            
            if not tissue_mask_path.exists() or not cell_mask_path.exists():
                logger.warning(f"Missing masks for {sample_name}, skipping")
                continue
            
            # Generate density heatmap
            density = generate_cell_density_heatmap(
                tissue_mask_path=str(tissue_mask_path),
                relabeled_mask_path=str(cell_mask_path),
                save_path=None,  # Don't save yet
                bin_size=bin_size,
                smooth=smooth,
                sigma=sigma,
                show_plot=False  # Don't show individual plots
            )
            
            all_densities.append(density)
            
            # Track statistics
            density_stats.append({
                'sample': sample_name,
                'min_density': np.min(density[density > 0]) if np.any(density > 0) else 0,
                'max_density': np.max(density),
                'mean_density': np.mean(density[density > 0]) if np.any(density > 0) else 0
            })
            
        except Exception as e:
            logger.error(f"Error processing {sample_name}: {e}")
            continue
    
    # Calculate global min/max
    if all_densities:
        # Get non-zero values for min calculation
        all_nonzero = []
        for density in all_densities:
            nonzero_vals = density[density > 0]
            if len(nonzero_vals) > 0:
                all_nonzero.extend(nonzero_vals)
        
        global_min = np.min(all_nonzero) if all_nonzero else 0
        global_max = max(np.max(density) for density in all_densities)
        
        # Create stats dataframe
        stats_df = pd.DataFrame(density_stats)
        
        print(f"\nGlobal density range: {global_min:.2f} - {global_max:.2f}")
        print(f"Mean density across samples: {stats_df['mean_density'].mean():.2f}")
        
        return global_min, global_max, all_densities, stats_df
    else:
        raise ValueError("No valid density maps generated")


def generate_density_maps_with_global_scale(
    organized_output_dir: str,
    results_df: pd.DataFrame,
    mask_name: str,
    global_min: float,
    global_max: float,
    bin_size: int = 100,
    smooth: bool = True,
    sigma: float = 1.5,
    cmap: str = 'hot',
    save_individual: bool = True
) -> list:
    """
    Generate density maps for all samples using the global color scale.
    
    Returns:
    --------
    list : List of dictionaries containing sample info and density data
    """
    density_results = []
    
    # Filter successful samples
    successful_samples = results_df[results_df['status'] == 'success']
    
    for _, row in tqdm(successful_samples.iterrows(), total=len(successful_samples),
                       desc="Generating density maps"):
        sample_name = row['sample']
        sample_dir = Path(organized_output_dir) / "samples" / sample_name
        
        try:
            # Create data paths
            data_paths = create_data_paths_from_organized(
                organized_sample_dir=str(sample_dir),
                experiment_name='Density Map Analysis'
            )
            
            # Get mask paths
            tissue_mask_path = data_paths.output_dir / f"{sample_name}_tissue_mask.npy"
            cell_mask_path = data_paths.output_dir / f"{sample_name}_{mask_name}.npy"
            
            if not tissue_mask_path.exists() or not cell_mask_path.exists():
                continue
            
            # Set save path if requested
            if save_individual:
                save_path = data_paths.output_dir / f"{sample_name}_density_heatmap_global_scale.png"
            else:
                save_path = None
            
            # Generate density heatmap with custom normalization
            plt.figure(figsize=(10, 8))
            
            # Load masks
            tissue_mask = np.load(tissue_mask_path)
            cell_mask = np.load(cell_mask_path)
            
            # Generate density
            density = generate_cell_density_heatmap(
                tissue_mask_path=str(tissue_mask_path),
                relabeled_mask_path=str(cell_mask_path),
                save_path=None,
                bin_size=bin_size,
                smooth=smooth,
                sigma=sigma,
                show_plot=False,
                cmap=cmap
            )
            
            # Create plot with global normalization
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot with global scale
            norm = Normalize(vmin=global_min, vmax=global_max)
            im = ax.imshow(density, cmap=cmap, norm=norm, interpolation='bilinear')
            
            # Add tissue boundary
            from skimage import measure
            contours = measure.find_contours(tissue_mask, 0.5)
            for contour in contours:
                ax.plot(contour[:, 1], contour[:, 0], 'cyan', linewidth=2, alpha=0.7)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Cell Density (cells per bin)', fontsize=12)
            
            # Add title and labels
            total_cells = len(np.unique(cell_mask)) - 1  # Exclude background
            ax.set_title(f'{sample_name}\nTotal Cells: {total_cells:,} | Bin Size: {bin_size}px', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('X Position (pixels)', fontsize=12)
            ax.set_ylabel('Y Position (pixels)', fontsize=12)
            
            # Remove axes for cleaner look
            ax.set_xticks([])
            ax.set_yticks([])
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved density map: {save_path}")
            
            plt.close()
            
            # Store results
            density_results.append({
                'sample': sample_name,
                'density': density,
                'total_cells': total_cells,
                'tissue_mask': tissue_mask,
                'cell_mask': cell_mask
            })
            
        except Exception as e:
            logger.error(f"Error generating map for {sample_name}: {e}")
            continue
    
    return density_results


def create_comparison_figure(
    density_results: list,
    global_min: float,
    global_max: float,
    output_path: str,
    cmap: str = 'hot',
    ncols: int = 4,
    fig_size_per_sample: tuple = (4, 3.5)
) -> None:
    """
    Create a comparison figure showing all density maps with the same color scale.
    
    Parameters:
    -----------
    density_results : list
        List of density result dictionaries
    global_min, global_max : float
        Global density range
    output_path : str
        Path to save the comparison figure
    cmap : str
        Colormap to use
    ncols : int
        Number of columns in the grid
    fig_size_per_sample : tuple
        Figure size per sample (width, height)
    """
    n_samples = len(density_results)
    nrows = (n_samples + ncols - 1) // ncols
    
    # Create figure
    fig_width = ncols * fig_size_per_sample[0]
    fig_height = nrows * fig_size_per_sample[1]
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Create grid with space for colorbar
    gs = gridspec.GridSpec(nrows, ncols + 1, width_ratios=[1]*ncols + [0.05],
                          hspace=0.3, wspace=0.2)
    
    # Common normalization
    norm = Normalize(vmin=global_min, vmax=global_max)
    
    # Plot each sample
    for idx, result in enumerate(density_results):
        row = idx // ncols
        col = idx % ncols
        ax = fig.add_subplot(gs[row, col])
        
        # Plot density
        im = ax.imshow(result['density'], cmap=cmap, norm=norm, interpolation='bilinear')
        
        # Add tissue boundary
        from skimage import measure
        contours = measure.find_contours(result['tissue_mask'], 0.5)
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], 'cyan', linewidth=1, alpha=0.7)
        
        # Title
        ax.set_title(f"{result['sample']}\n{result['total_cells']:,} cells", 
                    fontsize=10, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide empty subplots
    for idx in range(n_samples, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        ax = fig.add_subplot(gs[row, col])
        ax.axis('off')
    
    # Add shared colorbar
    cbar_ax = fig.add_subplot(gs[:, -1])
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Cell Density\n(cells per bin)', fontsize=12)
    
    # Main title
    fig.suptitle('Cell Density Comparison - Global Scale', fontsize=16, fontweight='bold', y=0.98)
    
    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved comparison figure: {output_path}")
    plt.close()


def process_density_maps_global_scale(
    organized_output_dir: str,
    tissue_filter_results_csv: str,
    mask_name: str = 'cy3_cy5_overlap_filtered_tissue_filtered',
    bin_size: int = 100,
    smooth: bool = True,
    sigma: float = 1.5,
    cmap: str = 'hot',
    save_individual: bool = True,
    create_comparison: bool = True,
    comparison_ncols: int = 4
) -> pd.DataFrame:
    """
    Main function to process all density maps with global color scale.
    
    Parameters:
    -----------
    organized_output_dir : str
        Path to organized output directory
    tissue_filter_results_csv : str
        Path to the CSV file from tissue_mask_filter_pipeline
    mask_name : str
        Name of the filtered mask to use (default: tissue filtered result)
    bin_size : int
        Bin size for density calculation (default: 100 pixels)
    smooth : bool
        Whether to apply Gaussian smoothing (default: True)
    sigma : float
        Sigma for Gaussian smoothing (default: 1.5)
    cmap : str
        Colormap to use (default: 'hot')
    save_individual : bool
        Whether to save individual density maps (default: True)
    create_comparison : bool
        Whether to create comparison figure (default: True)
    comparison_ncols : int
        Number of columns in comparison figure (default: 4)
        
    Returns:
    --------
    pd.DataFrame
        Statistics about the density analysis
    """
    
    # Load tissue filtering results
    print(f"Loading results from: {tissue_filter_results_csv}")
    results_df = pd.read_csv(tissue_filter_results_csv)
    
    # Filter for successful samples with tissue masks
    valid_samples = results_df[
        (results_df['status'] == 'success') & 
        (results_df['has_tissue_mask'] == True) &
        (results_df['tissue_filtered_cells'] > 0)
    ]
    
    print(f"Found {len(valid_samples)} valid samples for density analysis")
    
    if len(valid_samples) == 0:
        raise ValueError("No valid samples found for density analysis")
    
    # Step 1: Calculate global density range
    print("\nStep 1: Calculating global density range...")
    global_min, global_max, all_densities, density_stats_df = calculate_global_density_range(
        organized_output_dir=organized_output_dir,
        results_df=valid_samples,
        mask_name=mask_name,
        bin_size=bin_size,
        smooth=smooth,
        sigma=sigma
    )
    
    # Step 2: Generate density maps with global scale
    print("\nStep 2: Generating density maps with global scale...")
    density_results = generate_density_maps_with_global_scale(
        organized_output_dir=organized_output_dir,
        results_df=valid_samples,
        mask_name=mask_name,
        global_min=global_min,
        global_max=global_max,
        bin_size=bin_size,
        smooth=smooth,
        sigma=sigma,
        cmap=cmap,
        save_individual=save_individual
    )
    
    # Step 3: Create comparison figure
    if create_comparison and len(density_results) > 1:
        print("\nStep 3: Creating comparison figure...")
        output_dir = Path(organized_output_dir)
        comparison_path = output_dir / f"density_comparison_global_scale_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        create_comparison_figure(
            density_results=density_results,
            global_min=global_min,
            global_max=global_max,
            output_path=str(comparison_path),
            cmap=cmap,
            ncols=comparison_ncols
        )
    
    # Save density statistics
    stats_path = Path(organized_output_dir) / f"density_stats_global_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    density_stats_df['global_min'] = global_min
    density_stats_df['global_max'] = global_max
    density_stats_df.to_csv(stats_path, index=False)
    print(f"\nDensity statistics saved to: {stats_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("DENSITY ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total samples analyzed: {len(density_results)}")
    print(f"Global density range: {global_min:.2f} - {global_max:.2f}")
    print(f"Mean density: {density_stats_df['mean_density'].mean():.2f} ± {density_stats_df['mean_density'].std():.2f}")
    print(f"Bin size: {bin_size} pixels")
    print(f"Smoothing: {'Yes' if smooth else 'No'} (sigma={sigma})")
    
    return density_stats_df


# Usage examples
if __name__ == "__main__":
    # Example 1: Basic usage with default parameters
    df = process_density_maps_global_scale(
        organized_output_dir='/path/to/organized/output',
        tissue_filter_results_csv='/path/to/tissue_filtering_results_cy3_cy5_overlap_filtered_20240101_120000.csv',
        mask_name='cy3_cy5_overlap_filtered_tissue_filtered',  # Tissue filtered mask
        bin_size=100,                      # 100x100 pixel bins
        smooth=False,                       # Apply smoothing
        sigma=1.5,                         # Smoothing strength
        cmap='hot',                        # Color scheme
        save_individual=True,              # Save individual maps
        create_comparison=False,            # Create comparison figure
        comparison_ncols=4                 # 4 columns in comparison
    )
    
    # Example 2: Higher resolution analysis
    # df = process_density_maps_global_scale(
    #     organized_output_dir='/path/to/organized/output',
    #     tissue_filter_results_csv='/path/to/tissue_filtering_results.csv',
    #     mask_name='cy5_filtered_relabeled_tissue_filtered',
    #     bin_size=50,                       # Smaller bins for higher resolution
    #     smooth=True,
    #     sigma=2.0,                         # Stronger smoothing
    #     cmap='viridis',                    # Different colormap
    #     save_individual=True,
    #     create_comparison=True,
    #     comparison_ncols=5                 # 5 columns for more samples
    # )
    
    # Example 3: No smoothing for raw density
    # df = process_density_maps_global_scale(
    #     organized_output_dir='/path/to/organized/output',
    #     tissue_filter_results_csv='/path/to/tissue_filtering_results.csv',
    #     mask_name='cy3_filtered_relabeled_tissue_filtered',
    #     bin_size=100,
    #     smooth=False,                      # No smoothing
    #     cmap='plasma',                     # Different colormap
    #     save_individual=True,
    #     create_comparison=True,
    #     comparison_ncols=3                 # 3 columns
    # )


'''
Workflow Example:
-----------------
1. First run tissue mask filtering:
   df = process_all_tissue_mask_filter(
       organized_output_dir='/data/organized',
       mask_to_filter='cy3_cy5_overlap_filtered'
   )

2. Then run global density analysis:
   density_df = process_density_maps_global_scale(
       organized_output_dir='/data/organized',
       tissue_filter_results_csv='/data/organized/tissue_filtering_results_cy3_cy5_overlap_filtered_20240101.csv',
       mask_name='cy3_cy5_overlap_filtered_tissue_filtered'
   )

Output Files:
-------------
- Individual density maps: {sample_name}_density_heatmap_global_scale.png
- Comparison figure: density_comparison_global_scale_{timestamp}.png
- Statistics CSV: density_stats_global_{timestamp}.csv

The global color scale ensures that colors represent the same density values
across all samples, enabling direct visual comparison.
'''