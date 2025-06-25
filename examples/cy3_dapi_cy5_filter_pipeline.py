from pathlib import Path
from lmtools.io import create_data_paths_from_organized, ProcessingStep
from lmtools.seg import (
    compute_average_intensity, compute_gmm_threshold,
    size_and_dapi_filter, count_cells, relabel_sequential_labels,
    immune_filter_by_overlap
)
from datetime import datetime
import pandas as pd
from tqdm import tqdm

def process_all_immune_samples(
    organized_output_dir: str,
    experiment_name: str = 'Immune Cell Analysis',
    min_cell_size: int = 100,
    min_overlap_ratio: float = 0.8,
    gmm_n_components: int = 1,
    gmm_n_delta: float = -1.0,
    save_plots: bool = True
):
    """
    Process all samples in organized output directory with immune cell filtering pipeline.
    
    Parameters:
    -----------
    organized_output_dir : str
        Path to organized output directory containing samples folder
    experiment_name : str
        Name for the experiment
    min_cell_size : int
        Minimum cell size in pixels for filtering
    min_overlap_ratio : float
        Minimum overlap ratio between CY3 and CY5 cells
    gmm_n_components : int
        Number of components for GMM threshold calculation
    gmm_n_delta : float
        Number of standard deviations for GMM threshold
    save_plots : bool
        Whether to save diagnostic plots
    
    Returns:
    --------
    pd.DataFrame
        Summary of processing results for all samples with counts at each step
    """

    samples_dir = Path(organized_output_dir) / "samples"
    sample_dirs = [d for d in samples_dir.iterdir() if d.is_dir()]

    results = []

    print(f"Found {len(sample_dirs)} samples to process")

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
                notes=f'Processing sample: {sample_name}'
            )

            # Load CY3 segmentation and DAPI image
            seg_cy3, = data_paths.load_segs(['cy3'])
            img_dapi, = data_paths.load_imgs(['dapi'])
            initial_cy3_count = count_cells(seg_cy3)

            # Load CY5 for reference
            seg_cy5, = data_paths.load_segs(['cy5'])
            cy5_count = count_cells(seg_cy5)

            # Calculate DAPI intensities
            dapi_intensities = compute_average_intensity(
                seg_mask=seg_cy3,
                intensity_img=img_dapi,
                use_donut=False,
                erode_radius=4
            )

            # Calculate GMM threshold
            threshold, gmm_info = compute_gmm_threshold(
                intensity_dict=dapi_intensities,
                n_components=gmm_n_components,
                exclude_components=1,
                n_delta=gmm_n_delta,
                data_paths=data_paths,
                intensity_channel="dapi",
                step_name="dapi_gmm_threshold"
            )

            # Count cells below threshold (will be removed)
            cells_below_dapi = sum(1 for intensity in dapi_intensities.values() if intensity < threshold)

            # Apply size and DAPI filter
            filtered_mask = size_and_dapi_filter(
                seg_mask=seg_cy3,
                dapi_mask=None,
                dapi_img=img_dapi,
                min_size=min_cell_size,
                min_overlap_ratio=None,
                min_dapi_intensity=threshold,
                data_paths=data_paths,
                step_name="size_dapi_filter"
            )

            # Relabel and count after size/DAPI filter
            filtered_mask = relabel_sequential_labels(filtered_mask)
            cy3_after_size_dapi = count_cells(filtered_mask)

            # Save filtered CY3 mask
            data_paths.save_processed_mask(
                mask=filtered_mask,
                name="cy3_filtered_relabeled",
                processing_info=ProcessingStep(
                    step_name="save_filtered_relabeled_mask",
                    timestamp=datetime.now().isoformat(),
                    parameters={
                        "filter_type": "size_and_dapi",
                        "min_size": min_cell_size,
                        "dapi_threshold": threshold,
                        "relabeled": True,
                        "final_cell_count": cy3_after_size_dapi
                    },
                    input_data=["cy3_segmentation", "dapi_image"],
                    notes=f"Filtered CY3 cells by size and DAPI intensity, then relabeled sequentially"
                )
            )

            # Filter by CY5 overlap
            filtered_cy3_cy5 = immune_filter_by_overlap(
                seg_mask=seg_cy3,
                ref_mask=seg_cy5,
                min_overlap_ratio=min_overlap_ratio,
                data_paths=data_paths,
                step_name="Cy5_mask_filter"
            )

            # Final relabel and count
            filtered_cy3_cy5 = relabel_sequential_labels(filtered_cy3_cy5)
            final_count = count_cells(filtered_cy3_cy5)

            # Save final filtered mask
            data_paths.save_processed_mask(
                mask=filtered_cy3_cy5,
                name="cy3_cy5_overlap_filtered",
                processing_info=ProcessingStep(
                    step_name="Cy5_mask_filter",
                    timestamp=datetime.now().isoformat(),
                    parameters={
                        "base_mask": "cy3",
                        "ref_mask": "cy5",
                        "min_overlap_ratio": min_overlap_ratio,
                        "original_cell_count": initial_cy3_count,
                        "filtered_cell_count": final_count,
                        "cells_removed": initial_cy3_count - final_count,
                        "removal_percentage": (initial_cy3_count - final_count) / initial_cy3_count * 100
                    },
                    input_data=["cy3_segmentation", "cy5_segmentation"],
                    notes="CY3 cells filtered by requiring >80% overlap with CY5 segmentation"
                )
            )

            # Save metadata
            metadata_path = data_paths.save_metadata()

            # Collect detailed results
            result = {
                'sample': sample_name,
                # Initial counts
                'initial_cy3_cells': initial_cy3_count,
                'initial_cy5_cells': cy5_count,

                # After each filtering step
                'cy3_cells_below_dapi_threshold': cells_below_dapi,
                'cy3_after_size_dapi_filter': cy3_after_size_dapi,
                'cy3_removed_by_size_dapi': initial_cy3_count - cy3_after_size_dapi,

                # Final counts
                'final_cy3_cy5_positive': final_count,
                'cy3_removed_by_cy5_overlap': initial_cy3_count - final_count,

                # Percentages
                'percent_cy3_passed_size_dapi': (cy3_after_size_dapi / initial_cy3_count * 100) if initial_cy3_count > 0 else 0,
                'percent_cy3_cy5_positive': (final_count / initial_cy3_count * 100) if initial_cy3_count > 0 else 0,
                'percent_of_cy5_that_are_cy3': (final_count / cy5_count * 100) if cy5_count > 0 else 0,

                # Filter parameters
                'dapi_threshold': threshold,
                'min_cell_size': min_cell_size,
                'min_overlap_ratio': min_overlap_ratio,

                # Status
                'status': 'success',
                'error': None
            }

            results.append(result)

            print(f"✓ Successfully processed {sample_name}")
            print(f"  Initial CY3 cells: {initial_cy3_count}")
            print(f"  After size/DAPI filter: {cy3_after_size_dapi} ({result['percent_cy3_passed_size_dapi']:.1f}%)")
            print(f"  Final CY3+CY5+ cells: {final_count} ({result['percent_cy3_cy5_positive']:.1f}%)")

        except Exception as e:
            print(f"✗ Error processing {sample_name}: {str(e)}")
            results.append({
                'sample': sample_name,
                'initial_cy3_cells': None,
                'initial_cy5_cells': None,
                'cy3_cells_below_dapi_threshold': None,
                'cy3_after_size_dapi_filter': None,
                'cy3_removed_by_size_dapi': None,
                'final_cy3_cy5_positive': None,
                'cy3_removed_by_cy5_overlap': None,
                'percent_cy3_passed_size_dapi': None,
                'percent_cy3_cy5_positive': None,
                'percent_of_cy5_that_are_cy3': None,
                'dapi_threshold': None,
                'min_cell_size': min_cell_size,
                'min_overlap_ratio': min_overlap_ratio,
                'status': 'failed',
                'error': str(e)
            })

    # Create summary DataFrame
    results_df = pd.DataFrame(results)

    # Save summary
    summary_path = Path(organized_output_dir) / "processing_summary_detailed.csv"
    results_df.to_csv(summary_path, index=False)
    print(f"\n{'='*60}")
    print(f"Processing complete! Summary saved to: {summary_path}")

    # Print summary statistics
    successful = results_df[results_df['status'] == 'success']
    if len(successful) > 0:
        print(f"\nSuccessfully processed: {len(successful)}/{len(results_df)} samples")
        print(f"\nAverage cell counts across samples:")
        print(f"  Initial CY3: {successful['initial_cy3_cells'].mean():.0f} ± {successful['initial_cy3_cells'].std():.0f}")
        print(f"  After size/DAPI: {successful['cy3_after_size_dapi_filter'].mean():.0f} ± {successful['cy3_after_size_dapi_filter'].std():.0f}")
        print(f"  Final CY3+CY5+: {successful['final_cy3_cy5_positive'].mean():.0f} ± {successful['final_cy3_cy5_positive'].std():.0f}")
        print(f"\nAverage percentages:")
        print(f"  CY3 cells passing size/DAPI: {successful['percent_cy3_passed_size_dapi'].mean():.1f}% ± {successful['percent_cy3_passed_size_dapi'].std():.1f}%")
        print(f"  CY3+CY5+ of all CY3: {successful['percent_cy3_cy5_positive'].mean():.1f}% ± {successful['percent_cy3_cy5_positive'].std():.1f}%")

    return results_df