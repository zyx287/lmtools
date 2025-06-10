# Data Processing Workflow Guide

This guide walks you through the complete workflow of organizing and processing microscopy data using the LM Tools package.

## Overview

The workflow consists of four main steps:
1. **Initial Organization**: Organize raw images by channel for cellpose batch processing
2. **Segmentation**: Run cellpose on organized channels
3. **Sample Organization**: Reorganize data by sample ID after segmentation
4. **Analysis**: Process immune cells with filtering pipelines

## Step 1: Initial Data Organization by Channel

### Prerequisites
- Raw microscopy images named like: `id_CY5.tiff`, `id_CY3.tiff`, `id_DAPI.tiff`
- Images should follow the pattern: `{sample_id}_{CHANNEL}.tiff`
- Note: There are no images with suffixes like `_cd11b` - only CY5, CY3, and DAPI

### Using the New Two-Step Organization Workflow

1. **Run Step 1 - Organize by Channel**:
```bash
python organize_data_workflow.py -s /path/to/raw/images -o /path/to/organized --step 1
```

This creates:
```
organized/
├── channels_for_segmentation/
│   ├── CY5/          # All CY5 images for batch cellpose
│   ├── CY3/          # All CY3 images for batch cellpose
│   └── DAPI/         # All DAPI images for batch cellpose
├── sample_list.csv   # Tracking all files and samples
└── run_cellpose_batch.sh  # Script for running cellpose
```

## Step 2: Run Cellpose Segmentation

### Batch Processing with Cellpose

After organizing by channel, run cellpose on each channel directory:

```bash
# For each channel directory
cellpose --dir organized/channels_for_segmentation/CY5 \
        --pretrained_model cyto \
        --diameter 30 \
        --save_npy

cellpose --dir organized/channels_for_segmentation/CY3 \
        --pretrained_model cyto \
        --diameter 30 \
        --save_npy

cellpose --dir organized/channels_for_segmentation/DAPI \
        --pretrained_model nuclei \
        --diameter 20 \
        --save_npy
```

Or use the generated script:
```bash
# Edit parameters as needed, then run:
bash organized/run_cellpose_batch.sh
```

## Step 3: Reorganize by Sample ID

After cellpose segmentation completes, reorganize data by sample:

```bash
python organize_data_workflow.py -s /path/to/raw/images -o /path/to/organized --step 2
```

This creates:
```
organized/
├── samples/
│   ├── Sample01/
│   │   ├── raw_images/
│   │   │   ├── Sample01_CY5.tiff
│   │   │   ├── Sample01_CY3.tiff
│   │   │   └── Sample01_DAPI.tiff
│   │   ├── segmentations/
│   │   │   ├── Sample01_CY5_masks.npy
│   │   │   ├── Sample01_CY3_masks.npy
│   │   │   └── Sample01_DAPI_masks.npy
│   │   ├── results/          # For filtered data
│   │   └── sample_metadata.json
│   ├── Sample02/
│   │   └── ...
│   ├── master_sample_list.csv   # All samples with channel info
│   ├── master_sample_list.xlsx  # Excel version
│   └── sample_ids.txt          # Simple list of IDs
```

### Sample Tracking

The workflow automatically generates:
- **master_sample_list.csv**: Complete sample information including available channels
- **sample_metadata.json**: Per-sample metadata with file paths
- Handles samples with missing channels (e.g., some samples may only have CY5 and DAPI)

## Step 4: Process Data with DataPaths

### Basic Usage for Sample-Organized Data

```python
from pathlib import Path
from lmtools.seg import create_data_paths
import pandas as pd

# Load master sample list
samples_dir = Path("/path/to/organized/samples")
master_list = pd.read_csv(samples_dir / "master_sample_list.csv")

# Process each sample
for idx, row in master_list.iterrows():
    sample_id = row['sample_id']
    sample_dir = Path(row['sample_dir'])
    
    # Check available channels
    if not row['has_cy5']:
        print(f"Skipping {sample_id} - no CY5 channel")
        continue
    
    # Create DataPaths for this sample
    data_paths = create_data_paths(
        base_dir=sample_dir,
        base_name=sample_id,
        experiment_name="Immune Cell Analysis",
        sample_id=sample_id,
        # Simple paths since everything is in sample folder
        channel_dirs={
            'cy5': 'raw_images',
            'cy3': 'raw_images' if row['has_cy3'] else None,
            'dapi': 'raw_images'
        },
        seg_dirs={
            'cy5_seg': 'segmentations',
            'cy3_seg': 'segmentations' if row['has_cy3'] else None,
            'dapi_seg': 'segmentations'
        }
    )
    
    # Process the sample
    # ... your processing code ...
```

### Complete Processing Pipeline Example

```python
from lmtools.seg import process_immune_cells
import json

# Process a specific sample
sample_dir = Path("/path/to/organized/samples/Sample01")

# Load sample metadata
with open(sample_dir / "sample_metadata.json") as f:
    metadata = json.load(f)

# Create DataPaths
data_paths = create_data_paths(
    base_dir=sample_dir,
    base_name=metadata['sample_id'],
    experiment_name="Immune Cell Processing",
    sample_id=metadata['sample_id']
)

# Run immune cell processing
results = process_immune_cells(
    data_paths,
    cy5_threshold=1000,
    dapi_threshold=500,
    cd11b_threshold=800,  # Will be skipped if no CD11b
    min_area=50,
    max_area=5000
)

# Results are automatically saved in the results/ folder
print(f"CY5+ cells: {results['cy5_cell_count']}")
print(f"DAPI+ cells: {results['dapi_cell_count']}")
```

## Complete Workflow Script

Use the provided example script for the complete workflow:

```bash
# Run the complete workflow
python examples/complete_workflow_example.py \
    -r /path/to/raw/images \
    -o /path/to/output

# Or just demonstrate loading organized data
python examples/complete_workflow_example.py \
    --demo-load /path/to/output/samples/Sample01
```

## Advanced Usage

### Processing Samples with Missing Channels

The workflow automatically handles missing channels:

```python
# The master_sample_list.csv includes boolean flags
# has_cy5, has_cy3, has_dapi for each sample

# Filter samples with specific channels
df = pd.read_csv("master_sample_list.csv")

# Only samples with all three channels
complete_samples = df[(df['has_cy5']) & (df['has_cy3']) & (df['has_dapi'])]

# Samples missing CY3 (common case)
no_cy3_samples = df[~df['has_cy3']]

print(f"Samples with all channels: {len(complete_samples)}")
print(f"Samples missing CY3: {len(no_cy3_samples)}")
```

### Custom Processing Based on Available Channels

```python
def process_sample_adaptive(sample_dir, metadata):
    """Process a sample based on available channels"""
    
    channels = metadata['channels']
    
    # Always require CY5 and DAPI
    if 'CY5' not in channels or 'DAPI' not in channels:
        print(f"Skipping {metadata['sample_id']} - missing required channels")
        return
    
    # Create DataPaths
    data_paths = create_data_paths(
        base_dir=sample_dir,
        base_name=metadata['sample_id'],
        experiment_name="Adaptive Processing",
        sample_id=metadata['sample_id']
    )
    
    # Load available data
    img_cy5 = data_paths.load_channel('cy5')
    img_dapi = data_paths.load_channel('dapi')
    seg_cy5 = data_paths.load_segmentation('cy5_seg')
    seg_dapi = data_paths.load_segmentation('dapi_seg')
    
    # Check for optional CY3
    if 'CY3' in channels:
        img_cy3 = data_paths.load_channel('cy3')
        # Process with CY3
    else:
        # Process without CY3
        print(f"Processing {metadata['sample_id']} without CY3")
```

### Batch Processing All Samples

```python
#!/usr/bin/env python3
"""
Batch process all samples after organization
"""
import pandas as pd
from pathlib import Path
from lmtools.seg import create_data_paths, process_immune_cells
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

def batch_process(organized_dir):
    """Process all samples in organized directory"""
    
    samples_dir = Path(organized_dir) / "samples"
    master_list = pd.read_csv(samples_dir / "master_sample_list.csv")
    
    # Track results
    results_summary = []
    
    for idx, row in master_list.iterrows():
        sample_id = row['sample_id']
        logging.info(f"Processing {sample_id} ({idx+1}/{len(master_list)})")
        
        try:
            sample_dir = Path(row['sample_dir'])
            
            # Create DataPaths
            data_paths = create_data_paths(
                base_dir=sample_dir,
                base_name=sample_id,
                experiment_name="Batch Processing",
                sample_id=sample_id
            )
            
            # Process
            results = process_immune_cells(
                data_paths,
                cy5_threshold=1000,
                dapi_threshold=500
            )
            
            # Record results
            results_summary.append({
                'sample_id': sample_id,
                'status': 'success',
                'cy5_cells': results['cy5_cell_count'],
                'dapi_cells': results['dapi_cell_count'],
                **row.to_dict()
            })
            
        except Exception as e:
            logging.error(f"Failed to process {sample_id}: {e}")
            results_summary.append({
                'sample_id': sample_id,
                'status': 'failed',
                'error': str(e),
                **row.to_dict()
            })
    
    # Save results summary
    results_df = pd.DataFrame(results_summary)
    results_df.to_csv(samples_dir / "processing_results.csv", index=False)
    
    # Print summary
    success_count = len(results_df[results_df['status'] == 'success'])
    print(f"\nProcessing complete: {success_count}/{len(results_df)} samples processed successfully")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python batch_process.py <organized_directory>")
        sys.exit(1)
    
    batch_process(sys.argv[1])
```

## Troubleshooting

### Common Issues

1. **Filename Pattern Mismatch**
   - Ensure files follow pattern: `{sample_id}_{CHANNEL}.tiff`
   - Channel must be CY5, CY3, or DAPI (case insensitive)
   - No spaces in channel name (use underscore)

2. **Missing Segmentation Masks**
   - Check cellpose output naming convention
   - Default cellpose output: `{filename}_cp_masks.npy`
   - The workflow looks for various patterns: `*_masks.npy`, `*_cp_masks.npy`

3. **Sample Organization Issues**
   - Check `sample_list.csv` was created in Step 1
   - Verify all expected samples are listed
   - Review `sample_organization_summary.csv` for issues

### Debug Helpers

```python
# Check what files were found
import json

sample_dir = Path("/path/to/samples/Sample01")
with open(sample_dir / "sample_metadata.json") as f:
    metadata = json.load(f)

print(f"Sample ID: {metadata['sample_id']}")
print(f"Channels found: {metadata['channels']}")
print("\nRaw images:")
for ch, path in metadata['raw_images'].items():
    print(f"  {ch}: {path}")
print("\nSegmentations:")
for ch, path in metadata['segmentations'].items():
    print(f"  {ch}: {path}")
```

## Best Practices

1. **Always run both organization steps**: Channel organization → Cellpose → Sample organization
2. **Check intermediate outputs**: Verify cellpose masks before Step 2
3. **Use the master sample list**: It tracks all samples and their available channels
4. **Save processing parameters**: Record all thresholds and settings used
5. **Handle missing channels gracefully**: Not all samples have all channels
6. **Use the results/ folder**: Keep processed data separate from raw data

## Next Steps

- Review the generated `master_sample_list.csv` to understand your dataset
- Check `sample_metadata.json` files for detailed sample information
- Use the example scripts as templates for your analysis
- See the [Data Organization Guide](data_organization_guide.md) for more details