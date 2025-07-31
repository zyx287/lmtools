# Data Organization Guide

This guide describes the required directory structure and naming conventions for the LM Tools data organization workflow.

## Overview

The data organization process consists of 3 steps:
1. **Step 1**: Organize raw images by channel for cellpose batch processing
2. **Step 2**: Reorganize data by sample after segmentation 
3. **Step 3**: Add tissue masks from QuPath to organized samples

## Step 1: Organize by Channel

### Purpose
Prepare images for batch cellpose segmentation by grouping them by channel type.

### Input Requirements
- Raw microscopy images in a single directory
- File naming pattern: `{sample_name}_{channel}.tif` or `{sample_name}_{channel}.tiff`
- Supported channels: CY5, CY3, DAPI, CD11b (case-insensitive in filename)

### Example Input Files
```
raw_images/
├── Slide 8 of 1_Region 001_CY5.tiff
├── Slide 8 of 1_Region 001_CY3.tiff  
├── Slide 8 of 1_Region 001_DAPI.tiff
├── Slide 8 of 1_Region 002_CY5.tiff
├── Slide 8 of 1_Region 002_CY3.tiff
├── Slide 8 of 1_Region 002_DAPI.tiff
└── ...
```

### Output Structure After Step 1
```
organized_output/
├── channels_for_segmentation/
│   ├── CY5/
│   │   ├── Slide 8 of 1_Region 001_CY5.tiff
│   │   └── Slide 8 of 1_Region 002_CY5.tiff
│   ├── CY3/
│   │   ├── Slide 8 of 1_Region 001_CY3.tiff
│   │   └── Slide 8 of 1_Region 002_CY3.tiff
│   └── DAPI/
│       ├── Slide 8 of 1_Region 001_DAPI.tiff
│       └── Slide 8 of 1_Region 002_DAPI.tiff
├── sample_list.csv
└── run_cellpose_batch.sh
```

### Usage
```python
from lmtools.io import organize_data

# Run step 1
channel_df = organize_data(
    source_dir='/path/to/raw/images',
    output_dir='/path/to/organized/output', 
    step=1
)
```

## Step 2: Organize by Sample

### Purpose
After cellpose segmentation, reorganize data by sample for processing pipelines.

### Requirements Before Step 2
1. Completed Step 1
2. Cellpose segmentation masks in channel directories
3. Segmentation mask naming: `{sample_name}_{channel}_cellpose_masks.npy`

### Expected Segmentation Files
```
organized_output/
└── channels_for_segmentation/
    ├── CY5/
    │   ├── Slide 8 of 1_Region 001_CY5_cellpose_masks.npy
    │   └── Slide 8 of 1_Region 002_CY5_cellpose_masks.npy
    ├── CY3/
    │   ├── Slide 8 of 1_Region 001_CY3_cellpose_masks.npy
    │   └── Slide 8 of 1_Region 002_CY3_cellpose_masks.npy
    └── DAPI/
        ├── Slide 8 of 1_Region 001_DAPI_cellpose_masks.npy
        └── Slide 8 of 1_Region 002_DAPI_cellpose_masks.npy
```

### Output Structure After Step 2
```
organized_output/
└── samples/
    ├── Slide 8 of 1_Region 001/
    │   ├── raw_images/
    │   │   ├── Slide 8 of 1_Region 001_CY5.tiff
    │   │   ├── Slide 8 of 1_Region 001_CY3.tiff
    │   │   └── Slide 8 of 1_Region 001_DAPI.tiff
    │   ├── segmentations/
    │   │   ├── Slide 8 of 1_Region 001_CY5_cellpose_masks.npy
    │   │   ├── Slide 8 of 1_Region 001_CY3_cellpose_masks.npy
    │   │   └── Slide 8 of 1_Region 001_DAPI_cellpose_masks.npy
    │   ├── results/
    │   └── sample_metadata.json
    ├── Slide 8 of 1_Region 002/
    │   └── ... (same structure)
    └── sample_ids.txt
```

### Usage
```python
# Run step 2 after cellpose segmentation
sample_df = organize_data(
    source_dir='/path/to/raw/images',
    output_dir='/path/to/organized/output',
    step=2
)
```

## Step 3: Add Tissue Masks

### Purpose
Copy QuPath tissue masks to appropriate sample folders.

### Requirements
- QuPath GeoJSON files with tissue annotations
- File naming must match sample names exactly
- Expected format: `{sample_name}_DAPI.geojson` or `{sample_name}.geojson`

### Example QuPath Files
```
qupath_masks/
├── Slide 8 of 1_Region 001_DAPI.geojson
├── Slide 8 of 1_Region 002_DAPI.geojson
└── ...
```

### Output After Step 3
```
organized_output/samples/
└── Slide 8 of 1_Region 001/
    ├── raw_images/
    ├── segmentations/
    ├── results/
    ├── tissue_masks/
    │   └── Slide 8 of 1_Region 001_DAPI.geojson
    └── sample_metadata.json
```

### Usage
```python
# Run step 3 to add tissue masks
mask_df = organize_data(
    source_dir='/path/to/raw/images',
    output_dir='/path/to/organized/output',
    step=3,
    qupath_dir='/path/to/qupath/masks'
)
```

## File Naming Rules

### Critical Naming Requirements

1. **Sample Name Extraction**
   - The sample name is extracted by removing the channel suffix
   - Example: `Slide 8 of 1_Region 001_CY5.tiff` → Sample name: `Slide 8 of 1_Region 001`

2. **Channel Identification**
   - Channels are identified by suffix: `_CY5`, `_CY3`, `_DAPI`, `_CD11b`
   - Case-insensitive matching (e.g., `_cy5`, `_Cy5`, `_CY5` all work)
   - Must appear at the end of filename before extension

3. **Segmentation Mask Naming**
   - Must follow pattern: `{sample_name}_{channel}_cellpose_masks.npy`
   - The sample name must match exactly between raw image and mask

4. **Tissue Mask Naming**
   - QuPath files must match sample name exactly
   - Acceptable patterns:
     - `{sample_name}_DAPI.geojson`
     - `{sample_name}.geojson`

### Valid File Name Examples
```
✓ Slide 8 of 1_Region 001_CY5.tiff
✓ Mouse1_Brain_Section2_DAPI.tif
✓ Exp20240115_S1_cy3.tiff
✓ Sample_001_Cd11b.tif
```

### Invalid File Name Examples
```
✗ CY5_Slide 8 of 1_Region 001.tiff  (channel at beginning)
✗ Slide8Region001CY5.tiff           (no underscore separator)
✗ Slide 8 of 1_Region 001.CY5.tiff  (channel after extension)
```

## Data Structure Reference

### DataOrganizer Attributes
```python
class DataOrganizer:
    source_dir: Path           # Input directory with raw images
    output_dir: Path          # Output directory for organized data
    sample_info: List[Dict]   # Information about processed samples
```

### Sample Metadata Structure (sample_metadata.json)
```json
{
    "sample_id": "Slide 8 of 1_Region 001",
    "raw_images": {
        "CY5": "Slide 8 of 1_Region 001_CY5.tiff",
        "CY3": "Slide 8 of 1_Region 001_CY3.tiff",
        "DAPI": "Slide 8 of 1_Region 001_DAPI.tiff"
    },
    "segmentations": {
        "CY5": "Slide 8 of 1_Region 001_CY5_cellpose_masks.npy",
        "CY3": "Slide 8 of 1_Region 001_CY3_cellpose_masks.npy",
        "DAPI": "Slide 8 of 1_Region 001_DAPI_cellpose_masks.npy"
    },
    "tissue_masks": {
        "QuPath": "Slide 8 of 1_Region 001_DAPI.geojson"
    },
    "organization_date": "2024-01-20T14:30:00",
    "source_directory": "/path/to/raw/images"
}
```

### Output DataFrames

**Step 1 DataFrame (channel_df)**
| Column | Description |
|--------|-------------|
| filename | Original filename |
| sample_id | Extracted sample name |
| channel | Detected channel (CY5, CY3, DAPI) |
| source_path | Original file path |
| dest_path | Organized file path |
| status | Copy status |

**Step 2 DataFrame (sample_df)**  
| Column | Description |
|--------|-------------|
| sample_id | Sample name |
| num_channels | Number of channels found |
| channels | List of channels |
| num_masks | Number of segmentation masks |
| masks | List of mask files |
| sample_dir | Path to sample directory |
| status | Organization status |

**Step 3 DataFrame (mask_df)**
| Column | Description |
|--------|-------------|
| sample_id | Sample name |
| tissue_mask_file | Source mask filename |
| source_path | Original mask path |
| dest_path | Organized mask path |
| status | Copy status |

## Complete Workflow Example

```python
from lmtools.io import organize_data

# Step 1: Organize by channel
print("Step 1: Organizing by channel...")
channel_df = organize_data(
    source_dir='/data/microscopy/raw_images',
    output_dir='/data/microscopy/organized',
    step=1
)

# Run cellpose segmentation here using the generated script
# ./organized/run_cellpose_batch.sh

# Step 2: Organize by sample after segmentation
print("Step 2: Organizing by sample...")
sample_df = organize_data(
    source_dir='/data/microscopy/raw_images',
    output_dir='/data/microscopy/organized',
    step=2
)

# Step 3: Add tissue masks from QuPath
print("Step 3: Adding tissue masks...")
mask_df = organize_data(
    source_dir='/data/microscopy/raw_images',
    output_dir='/data/microscopy/organized',
    step=3,
    qupath_dir='/data/microscopy/qupath_masks'
)

print("Data organization complete!")
```

## Using Organized Data in Pipelines

After organization, use the data with processing pipelines:

```python
from lmtools.io import create_data_paths_from_organized

# For each organized sample
sample_dir = '/data/microscopy/organized/samples/Slide 8 of 1_Region 001'

data_paths = create_data_paths_from_organized(
    organized_sample_dir=sample_dir,
    experiment_name='My Experiment'
)

# Access files
print(f"CY5 image: {data_paths.cy5_img}")
print(f"CY5 mask: {data_paths.cy5_seg}")
print(f"Tissue mask: {data_paths.get_tissue_mask_path()}")
```

## Important Notes

1. **File Extensions**: Both `.tif` and `.tiff` are supported
2. **Case Sensitivity**: Channel names in filenames are case-insensitive
3. **Special Characters**: Sample names can contain spaces and special characters
4. **Overwriting**: Step 2 will overwrite existing sample directories
5. **Missing Files**: The organizer will continue processing even if some files are missing
6. **Mask Copying**: Segmentation masks are only copied if they exist (no error if missing)