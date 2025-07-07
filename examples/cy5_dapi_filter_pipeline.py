from pathlib import Path
from lmtools.io import create_data_paths_from_organized, ProcessingStep
from lmtools.seg import (
    compute_average_intensity, compute_gmm_threshold,
    size_and_dapi_filter, count_cells, relabel_sequential_labels
)
from datetime import datetime
import pandas as pd
from tqdm import tqdm

def process_all_cy5_samples(
    organized_output_dir: str,
    experiment_name: str = 'CY5 DAPI Analysis',
    min_cell_size: int = 100,
    gmm_n_components: int = 1,
    gmm_n_delta: float = -1.0,
    save_plots: bool = True
):
    """
    Process all samples in organized output directory with CY5-DAPI filtering pipeline.
    
    Parameters:
    -----------
    organized_output_dir : str
        Path to organized output directory containing samples folder
    experiment_name : str
        Name for the experiment
    min_cell_size : int
        Minimum cell size in pixels for filtering
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

            # Load CY5 segmentation and DAPI image
            seg_cy5, = data_paths.load_segs(['cy5'])
            img_dapi, = data_paths.load_imgs(['dapi'])
            initial_cy5_count = count_cells(seg_cy5)

            # Calculate DAPI intensities for CY5 cells
            dapi_intensities = compute_average_intensity(
                seg_mask=seg_cy5,
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
                step_name="cy5_dapi_gmm_threshold"
            )

            # Count cells below threshold (will be removed)
            cells_below_dapi = sum(1 for intensity in dapi_intensities.values() if intensity < threshold)

            # Apply size and DAPI filter to CY5
            filtered_mask = size_and_dapi_filter(
                seg_mask=seg_cy5,
                dapi_mask=None,
                dapi_img=img_dapi,
                min_size=min_cell_size,
                min_overlap_ratio=None,
                min_dapi_intensity=threshold,
                data_paths=data_paths,
                step_name="cy5_size_dapi_filter"
            )

            # Relabel and count after size/DAPI filter
            filtered_mask = relabel_sequential_labels(filtered_mask)
            cy5_after_size_dapi = count_cells(filtered_mask)

            # Save filtered CY5 mask
            data_paths.save_processed_mask(
                mask=filtered_mask,
                name="cy5_filtered_relabeled",
                processing_info=ProcessingStep(
                    step_name="save_cy5_filtered_relabeled_mask",
                    timestamp=datetime.now().isoformat(),
                    parameters={
                        "filter_type": "size_and_dapi",
                        "min_size": min_cell_size,
                        "dapi_threshold": threshold,
                        "relabeled": True,
                        "final_cell_count": cy5_after_size_dapi
                    },
                    input_data=["cy5_segmentation", "dapi_image"],
                    notes=f"Filtered CY5 cells by size and DAPI intensity, then relabeled sequentially"
                )
            )

            # Save metadata
            metadata_path = data_paths.save_metadata()

            # Collect detailed results
            result = {
                'sample': sample_name,
                # Initial counts
                'initial_cy5_cells': initial_cy5_count,

                # After filtering
                'cy5_cells_below_dapi_threshold': cells_below_dapi,
                'cy5_after_size_dapi_filter': cy5_after_size_dapi,
                'cy5_removed_by_size_dapi': initial_cy5_count - cy5_after_size_dapi,

                # Percentages
                'percent_cy5_passed_size_dapi': (cy5_after_size_dapi / initial_cy5_count * 100) if initial_cy5_count > 0 else 0,
                'percent_cy5_removed': ((initial_cy5_count - cy5_after_size_dapi) / initial_cy5_count * 100) if initial_cy5_count > 0 else 0,

                # Filter parameters
                'dapi_threshold': threshold,
                'min_cell_size': min_cell_size,

                # Status
                'status': 'success',
                'error': None
            }

            results.append(result)

            print(f"✓ Successfully processed {sample_name}")
            print(f"  Initial CY5 cells: {initial_cy5_count}")
            print(f"  CY5 cells below DAPI threshold: {cells_below_dapi}")
            print(f"  Final CY5 cells after size/DAPI filter: {cy5_after_size_dapi} ({result['percent_cy5_passed_size_dapi']:.1f}%)")
            print(f"  Total CY5 cells removed: {initial_cy5_count - cy5_after_size_dapi} ({result['percent_cy5_removed']:.1f}%)")

        except Exception as e:
            print(f"✗ Error processing {sample_name}: {str(e)}")
            results.append({
                'sample': sample_name,
                'initial_cy5_cells': None,
                'cy5_cells_below_dapi_threshold': None,
                'cy5_after_size_dapi_filter': None,
                'cy5_removed_by_size_dapi': None,
                'percent_cy5_passed_size_dapi': None,
                'percent_cy5_removed': None,
                'dapi_threshold': None,
                'min_cell_size': min_cell_size,
                'status': 'failed',
                'error': str(e)
            })

    # Create summary DataFrame
    results_df = pd.DataFrame(results)

    # Save summary
    summary_path = Path(organized_output_dir) / "cy5_dapi_processing_summary.csv"
    results_df.to_csv(summary_path, index=False)
    print(f"\n{'='*60}")
    print(f"Processing complete! Summary saved to: {summary_path}")

    # Print summary statistics
    successful = results_df[results_df['status'] == 'success']
    if len(successful) > 0:
        print(f"\nSuccessfully processed: {len(successful)}/{len(results_df)} samples")
        print(f"\nAverage cell counts across samples:")
        print(f"  Initial CY5: {successful['initial_cy5_cells'].mean():.0f} ± {successful['initial_cy5_cells'].std():.0f}")
        print(f"  After size/DAPI: {successful['cy5_after_size_dapi_filter'].mean():.0f} ± {successful['cy5_after_size_dapi_filter'].std():.0f}")
        print(f"  Cells removed: {successful['cy5_removed_by_size_dapi'].mean():.0f} ± {successful['cy5_removed_by_size_dapi'].std():.0f}")
        print(f"\nAverage percentages:")
        print(f"  CY5 cells passing size/DAPI: {successful['percent_cy5_passed_size_dapi'].mean():.1f}% ± {successful['percent_cy5_passed_size_dapi'].std():.1f}%")
        print(f"  CY5 cells removed: {successful['percent_cy5_removed'].mean():.1f}% ± {successful['percent_cy5_removed'].std():.1f}%")
        print(f"\nAverage DAPI threshold: {successful['dapi_threshold'].mean():.2f} ± {successful['dapi_threshold'].std():.2f}")

    return results_df