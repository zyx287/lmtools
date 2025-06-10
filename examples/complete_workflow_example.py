#!/usr/bin/env python3
"""
Complete Data Organization and Processing Workflow Example
=========================================================

This example demonstrates the complete workflow:
1. Organize raw images by channel for cellpose
2. Run cellpose segmentation
3. Reorganize by sample ID
4. Process immune cells with filtering

Author: LMTools
"""

import os
import subprocess
from pathlib import Path
import pandas as pd
from lmtools.seg import create_data_paths
from lmtools.seg.immune_cell import process_immune_cells_batch


def run_workflow(raw_images_dir: str, output_dir: str):
    """Run the complete data organization and processing workflow."""
    
    print("=" * 70)
    print("LMTools Complete Workflow Example")
    print("=" * 70)
    
    # Convert to Path objects
    raw_dir = Path(raw_images_dir)
    out_dir = Path(output_dir)
    
    # Step 1: Organize images by channel
    print("\n[Step 1] Organizing images by channel for cellpose...")
    cmd1 = [
        "lmtools", "organize_data",
        "-s", str(raw_dir),
        "-o", str(out_dir),
        "--step", "1"
    ]
    subprocess.run(cmd1, check=True)
    
    # Step 2: Run cellpose (example - adjust parameters as needed)
    print("\n[Step 2] Running cellpose segmentation...")
    print("Note: This is an example. Adjust cellpose parameters for your data.")
    
    channels_dir = out_dir / "channels_for_segmentation"
    for channel in ["CY5", "CY3", "DAPI"]:
        channel_path = channels_dir / channel
        if channel_path.exists() and any(channel_path.glob("*.tif*")):
            print(f"\nProcessing {channel} channel...")
            # Example cellpose command
            cmd_cellpose = [
                "cellpose",
                "--dir", str(channel_path),
                "--pretrained_model", "cyto",
                "--diameter", "30",
                "--save_npy",
                "--no_npy"  # Remove if you want numpy output
            ]
            print(f"Command: {' '.join(cmd_cellpose)}")
            # Uncomment to actually run cellpose:
            # subprocess.run(cmd_cellpose, check=True)
    
    # Step 3: Reorganize by sample ID
    print("\n[Step 3] Reorganizing data by sample ID...")
    cmd3 = [
        "lmtools", "organize_data",
        "-s", str(raw_dir),
        "-o", str(out_dir),
        "--step", "2"
    ]
    subprocess.run(cmd3, check=True)
    
    # Step 4: Process immune cells for each sample
    print("\n[Step 4] Processing immune cells for each sample...")
    
    # Load the master sample list
    samples_dir = out_dir / "samples"
    master_list_path = samples_dir / "master_sample_list.csv"
    
    if master_list_path.exists():
        samples_df = pd.read_csv(master_list_path)
        
        # Process each sample
        for idx, row in samples_df.iterrows():
            sample_id = row['sample_id']
            sample_dir = Path(row['sample_dir'])
            
            print(f"\nProcessing sample: {sample_id}")
            print(f"  Channels available: {row['channels']}")
            
            # Create DataPaths for this sample
            data_paths = create_data_paths(
                base_dir=sample_dir,
                base_name=sample_id,
                experiment_name="Immune Cell Analysis",
                sample_id=sample_id,
                channel_dirs={
                    'cy5': 'raw_images',
                    'cy3': 'raw_images',
                    'dapi': 'raw_images'
                },
                seg_dirs={
                    'cy5_seg': 'segmentations',
                    'cy3_seg': 'segmentations',
                    'dapi_seg': 'segmentations'
                }
            )
            
            # Process immune cells if CY5 is available
            if row['has_cy5']:
                try:
                    # Run immune cell processing using the batch function
                    # Note: You'll need to implement your own processing logic here
                    # This is an example of what you might do:
                    
                    # Load data
                    img_cy5 = data_paths.load_channel('cy5')
                    img_dapi = data_paths.load_channel('dapi')
                    seg_cy5 = data_paths.load_segmentation('cy5_seg')
                    seg_dapi = data_paths.load_segmentation('dapi_seg')
                    
                    if img_cy5 is not None and seg_cy5 is not None:
                        from lmtools.seg import count_cells
                        cy5_count = count_cells(seg_cy5)
                        dapi_count = count_cells(seg_dapi) if seg_dapi is not None else 0
                        
                        results = {
                            'cy5_cell_count': cy5_count,
                            'dapi_cell_count': dapi_count
                        }
                        
                        # Save results
                        data_paths.add_processing_step(
                            'cell_counting',
                            {'cy5_cells': cy5_count, 'dapi_cells': dapi_count}
                        )
                        data_paths.save_metadata()
                    else:
                        raise ValueError("Could not load required data")
                    
                    print(f"  Processing complete!")
                    print(f"  CY5+ cells: {results.get('cy5_cell_count', 0)}")
                    print(f"  DAPI+ cells: {results.get('dapi_cell_count', 0)}")
                    if 'cd11b_cell_count' in results:
                        print(f"  CD11b+ cells: {results['cd11b_cell_count']}")
                    
                except Exception as e:
                    print(f"  Error processing sample: {e}")
            else:
                print("  Skipping - no CY5 channel available")
    
    print("\n" + "=" * 70)
    print("Workflow complete!")
    print(f"All results saved in: {out_dir}")
    print("=" * 70)


def demonstrate_data_loading(sample_dir: str):
    """Demonstrate how to load and work with organized data."""
    
    print("\n" + "-" * 50)
    print("Example: Loading organized data")
    print("-" * 50)
    
    sample_path = Path(sample_dir)
    
    # Load sample metadata
    metadata_path = sample_path / "sample_metadata.json"
    if metadata_path.exists():
        import json
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        print(f"\nSample ID: {metadata['sample_id']}")
        print(f"Available channels: {', '.join(metadata['channels'])}")
        print("\nRaw images:")
        for channel, path in metadata['raw_images'].items():
            print(f"  {channel}: {path}")
        
        print("\nSegmentations:")
        for channel, path in metadata['segmentations'].items():
            print(f"  {channel}: {path}")
    
    # Example: Load specific data
    from lmtools.seg import create_data_paths
    import numpy as np
    
    data_paths = create_data_paths(
        base_dir=sample_path,
        base_name=metadata['sample_id'],
        experiment_name="Example",
        sample_id=metadata['sample_id']
    )
    
    # Load images and masks
    if 'CY5' in metadata['channels']:
        cy5_img = data_paths.load_channel('cy5')
        print(f"\nCY5 image shape: {cy5_img.shape if cy5_img is not None else 'Not found'}")
        
        cy5_mask = data_paths.load_segmentation('cy5_seg')
        if cy5_mask is not None:
            print(f"CY5 mask shape: {cy5_mask.shape}")
            print(f"Number of cells: {len(np.unique(cy5_mask)) - 1}")  # -1 for background


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Complete workflow example for LMTools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Run complete workflow
  python complete_workflow_example.py -r /path/to/raw/images -o /path/to/output
  
  # Just demonstrate data loading
  python complete_workflow_example.py --demo-load /path/to/output/samples/Sample01
        """
    )
    
    parser.add_argument('-r', '--raw-dir', help='Directory with raw images')
    parser.add_argument('-o', '--output-dir', help='Output directory')
    parser.add_argument('--demo-load', help='Demonstrate loading from a sample directory')
    
    args = parser.parse_args()
    
    if args.demo_load:
        demonstrate_data_loading(args.demo_load)
    elif args.raw_dir and args.output_dir:
        run_workflow(args.raw_dir, args.output_dir)
    else:
        parser.print_help()