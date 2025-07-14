'''
author: zyx
date: 2024-01-20
description: 
    Visualization functions for segmentation results
'''

import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops, find_contours
from scipy.ndimage import gaussian_filter
from pathlib import Path
from typing import Optional, Union, Tuple
import os


def generate_cell_density_heatmap(
    tissue_mask_path: Union[str, Path, np.ndarray],
    relabeled_mask_path: Union[str, Path, np.ndarray],
    bin_size: int = 100,
    output_path: Optional[Union[str, Path]] = None,
    smooth: bool = True,
    sigma: float = 2,
    figsize: Tuple[int, int] = (10, 10),
    dpi: int = 300,
    cmap: str = 'hot',
    overlay_color: str = 'cyan',
    overlay_linewidth: float = 1.5,
    show_plot: bool = False
) -> np.ndarray:
    """
    Generate a binned heatmap of cell density with tissue boundary overlay.

    Parameters:
    -----------
    tissue_mask_path : str, Path, or np.ndarray
        Path to binary tissue mask (.npy) or the mask array itself. 
        Values should be 0 or non-zero.
    relabeled_mask_path : str, Path, or np.ndarray
        Path to labeled cell mask (.npy) or the mask array itself.
        Each cell should have a unique label.
    bin_size : int, default=100
        Size of square bin in pixels.
    output_path : str or Path, optional
        Path to save the output heatmap image (.png).
        If None, the plot is not saved.
    smooth : bool, default=True
        Whether to apply Gaussian smoothing to the heatmap.
    sigma : float, default=2
        Standard deviation for Gaussian filter.
    figsize : tuple, default=(10, 10)
        Figure size in inches.
    dpi : int, default=300
        Dots per inch for saved figure.
    cmap : str, default='hot'
        Colormap for the heatmap.
    overlay_color : str, default='cyan'
        Color for tissue boundary overlay.
    overlay_linewidth : float, default=1.5
        Line width for tissue boundary.
    show_plot : bool, default=False
        Whether to display the plot.
        
    Returns:
    --------
    np.ndarray
        The density heatmap array (normalized).
    """
    # Load masks if paths are provided
    if isinstance(tissue_mask_path, (str, Path)):
        tissue_mask = np.load(tissue_mask_path)
    else:
        tissue_mask = tissue_mask_path
        
    if isinstance(relabeled_mask_path, (str, Path)):
        relabeled_mask = np.load(relabeled_mask_path)
    else:
        relabeled_mask = relabeled_mask_path

    # Ensure binary mask
    tissue_mask_bin = (tissue_mask > 0).astype(np.uint8)

    h, w = relabeled_mask.shape
    
    # Calculate heatmap dimensions
    heatmap_h = (h + bin_size - 1) // bin_size  # Ceiling division
    heatmap_w = (w + bin_size - 1) // bin_size
    heatmap = np.zeros((heatmap_h, heatmap_w), dtype=np.float32)

    # Count cell centroids per bin
    regions = regionprops(relabeled_mask)
    cell_count = len(regions)
    
    for region in regions:
        cy, cx = map(int, region.centroid)
        bin_y = cy // bin_size
        bin_x = cx // bin_size
        if 0 <= bin_y < heatmap.shape[0] and 0 <= bin_x < heatmap.shape[1]:
            heatmap[bin_y, bin_x] += 1

    # Calculate cells per bin for reporting
    max_cells_per_bin = int(heatmap.max())
    
    # Normalize heatmap
    if heatmap.max() > 0:
        heatmap_normalized = heatmap / heatmap.max()
    else:
        heatmap_normalized = heatmap

    # Optional smoothing
    if smooth and sigma > 0:
        heatmap_smoothed = gaussian_filter(heatmap_normalized, sigma=sigma)
    else:
        heatmap_smoothed = heatmap_normalized

    # Resize heatmap back to image size
    heatmap_resized = np.kron(heatmap_smoothed, np.ones((bin_size, bin_size)))
    heatmap_resized = heatmap_resized[:h, :w]  # Clip to exact size

    # Find contours in tissue mask
    contours = find_contours(tissue_mask_bin, level=0.5)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    im = ax.imshow(heatmap_resized, cmap=cmap, interpolation='nearest', aspect='auto')
    
    # Plot tissue boundary
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], color=overlay_color, linewidth=overlay_linewidth)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Normalized Cell Density')
    
    # Add title with information
    title = f"Cell Density Heatmap\n"
    title += f"Total cells: {cell_count}, Bin size: {bin_size}px, "
    title += f"Max cells/bin: {max_cells_per_bin}"
    if smooth:
        title += f", Smoothed (σ={sigma})"
    ax.set_title(title)
    ax.axis('off')

    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=dpi)
        print(f"Saved heatmap to: {output_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return heatmap_normalized


def generate_cell_distribution_plot(
    relabeled_mask: Union[str, Path, np.ndarray],
    tissue_mask: Optional[Union[str, Path, np.ndarray]] = None,
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 10),
    dpi: int = 300,
    cell_color: str = 'red',
    tissue_color: str = 'lightgray',
    tissue_alpha: float = 0.3,
    cell_size: int = 10,
    show_plot: bool = False
) -> dict:
    """
    Generate a scatter plot showing cell distributions within tissue boundaries.
    
    Parameters:
    -----------
    relabeled_mask : str, Path, or np.ndarray
        Path to labeled cell mask or the mask array itself.
    tissue_mask : str, Path, or np.ndarray, optional
        Path to tissue mask or the mask array itself.
    output_path : str or Path, optional
        Path to save the output plot.
    figsize : tuple, default=(12, 10)
        Figure size in inches.
    dpi : int, default=300
        Dots per inch for saved figure.
    cell_color : str, default='red'
        Color for cell centroids.
    tissue_color : str, default='lightgray'
        Color for tissue area.
    tissue_alpha : float, default=0.3
        Transparency for tissue overlay.
    cell_size : int, default=10
        Size of cell markers.
    show_plot : bool, default=False
        Whether to display the plot.
        
    Returns:
    --------
    dict
        Dictionary containing cell statistics.
    """
    # Load masks
    if isinstance(relabeled_mask, (str, Path)):
        relabeled_mask = np.load(relabeled_mask)
    else:
        relabeled_mask = relabeled_mask
        
    if tissue_mask is not None:
        if isinstance(tissue_mask, (str, Path)):
            tissue_mask = np.load(tissue_mask)
    
    # Get cell centroids
    regions = regionprops(relabeled_mask)
    centroids = np.array([region.centroid for region in regions])
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot tissue mask if provided
    if tissue_mask is not None:
        tissue_bin = (tissue_mask > 0)
        ax.imshow(tissue_bin, cmap='gray', alpha=tissue_alpha, aspect='auto')
    
    # Plot cell centroids
    if len(centroids) > 0:
        ax.scatter(centroids[:, 1], centroids[:, 0], 
                   c=cell_color, s=cell_size, alpha=0.6)
    
    # Set limits and labels
    ax.set_xlim(0, relabeled_mask.shape[1])
    ax.set_ylim(relabeled_mask.shape[0], 0)
    ax.set_aspect('equal')
    ax.set_title(f'Cell Distribution (n={len(centroids)} cells)')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=dpi)
        print(f"Saved distribution plot to: {output_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    # Calculate statistics
    stats = {
        'total_cells': len(centroids),
        'mean_x': np.mean(centroids[:, 1]) if len(centroids) > 0 else 0,
        'mean_y': np.mean(centroids[:, 0]) if len(centroids) > 0 else 0,
        'std_x': np.std(centroids[:, 1]) if len(centroids) > 0 else 0,
        'std_y': np.std(centroids[:, 0]) if len(centroids) > 0 else 0,
    }
    
    return stats