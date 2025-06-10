# Data Organization Guide for LM Tools

This guide describes the recommended file structure and naming conventions for using the DataPaths system in the immune cell processing pipeline.

## Directory Structure

### Recommended Channel-Separated Structure
```
experiment_folder/
│
├── raw_images/              # Original microscopy images organized by channel
│   ├── CY5/
│   │   ├── Slide 8 of 1_Region 001_CY5.tiff
│   │   ├── Slide 8 of 1_Region 002_CY5.tiff
│   │   └── ...
│   ├── CY3/
│   │   ├── Slide 8 of 1_Region 001_CY3.tiff
│   │   ├── Slide 8 of 1_Region 002_CY3.tiff
│   │   └── ...
│   ├── DAPI/
│   │   ├── Slide 8 of 1_Region 001_DAPI.tiff
│   │   ├── Slide 8 of 1_Region 002_DAPI.tiff
│   │   └── ...
│   └── CD11b/
│       ├── Slide 8 of 1_Region 001_CD11b.tiff
│       ├── Slide 8 of 1_Region 002_CD11b.tiff
│       └── ...
│
├── segmentations/           # Cellpose or other segmentation outputs by channel
│   ├── CY5/
│   │   ├── Slide 8 of 1_Region 001_CY5_cellpose_masks.npy
│   │   └── ...
│   ├── CY3/
│   │   ├── Slide 8 of 1_Region 001_CY3_cellpose_masks.npy
│   │   └── ...
│   ├── DAPI/
│   │   ├── Slide 8 of 1_Region 001_DAPI_cellpose_masks.npy
│   │   └── ...
│   └── QuPath/              # Optional QuPath masks
│       ├── Slide 8 of 1_Region 001_qupath_mask.npy
│       └── ...
│
└── processed/               # Auto-created by DataPaths
    ├── Slide 8 of 1_Region 001_cy5_dapi_filtered.npy
    ├── Slide 8 of 1_Region 001_final_immune_cells.npy
    ├── Slide 8 of 1_Region 001_metadata.json
    └── ...
```

### Alternative Flat Structure (also supported)
```
experiment_folder/
│
├── raw_images/              # All images in one directory
│   ├── Sample01_CY5.tif
│   ├── Sample01_CY3.tif
│   ├── Sample01_DAPI.tif
│   ├── Sample01_CD11b.tif
│   └── ...
│
├── segmentations/           # All segmentations in one directory
│   ├── Sample01_CY5_cellpose_masks.npy
│   ├── Sample01_CY3_cellpose_masks.npy
│   ├── Sample01_DAPI_cellpose_masks.npy
│   └── ...
│
└── processed/
    └── ...
```

## Naming Conventions

### Base Name Pattern
Each sample should have a consistent base name. Common patterns include:
- **Slide-based**: `Slide 8 of 1_Region 001` (for slide scanners)
- **Simple**: `Sample01`, `Mouse1_Section2`
- **Date-based**: `Exp20240115_S1`

### Default File Suffixes

#### Raw Image Files
- **CY5 channel**: `{base_name}_CY5.tiff` or `{base_name}_CY5.tif`
- **CY3 channel**: `{base_name}_CY3.tiff` or `{base_name}_CY3.tif`
- **DAPI channel**: `{base_name}_DAPI.tiff` or `{base_name}_DAPI.tif`
- **CD11b channel**: `{base_name}_CD11b.tiff` or `{base_name}_CD11b.tif`

#### Segmentation Files
- **CY5 segmentation**: `{base_name}_CY5_cellpose_masks.npy`
- **CY3 segmentation**: `{base_name}_CY3_cellpose_masks.npy`
- **DAPI segmentation**: `{base_name}_DAPI_cellpose_masks.npy`
- **QuPath segmentation**: `{base_name}_qupath_mask.npy` or `qupath_mask.npy`

### Alternative Naming Patterns (Auto-detected)
The system will also try these patterns if the default isn't found:
- `{base_name}_{CHANNEL}.tiff` / `{base_name}_{CHANNEL}.tif`
- `{base_name}-{CHANNEL}.tiff` / `{base_name}-{CHANNEL}.tif`
- `{CHANNEL}_{base_name}.tiff` / `{CHANNEL}_{base_name}.tif`

Where `{CHANNEL}` is the uppercase channel name (CY5, CY3, DAPI, CD11B).

## File Formats

### Images
- **Format**: TIFF (.tif or .tiff)
- **Bit depth**: 8-bit, 16-bit, or 32-bit
- **Dimensions**: 2D (XY) or 3D (XYZ) supported
- **Multi-channel**: Split into separate files

### Segmentations
- **Format**: NumPy array (.npy)
- **Type**: Label mask (0 = background, 1+ = object IDs)
- **Dimensions**: Must match corresponding image

### Metadata
- **Format**: JSON (.json)
- **Auto-generated**: Created by DataPaths.save_metadata()

## Example Setup

### Option 1: Channel-Separated Directories (Recommended)
Organize files by channel:
```
my_experiment/
├── raw_images/
│   ├── CY5/
│   │   ├── Slide 8 of 1_Region 001_CY5.tiff
│   │   └── Slide 8 of 1_Region 002_CY5.tiff
│   ├── CY3/
│   │   ├── Slide 8 of 1_Region 001_CY3.tiff
│   │   └── Slide 8 of 1_Region 002_CY3.tiff
│   ├── DAPI/
│   │   ├── Slide 8 of 1_Region 001_DAPI.tiff
│   │   └── Slide 8 of 1_Region 002_DAPI.tiff
│   └── CD11b/
│       ├── Slide 8 of 1_Region 001_CD11b.tiff
│       └── Slide 8 of 1_Region 002_CD11b.tiff
├── segmentations/
│   ├── CY5/
│   │   └── Slide 8 of 1_Region 001_CY5_cellpose_masks.npy
│   ├── DAPI/
│   │   └── Slide 8 of 1_Region 001_DAPI_cellpose_masks.npy
│   └── QuPath/
│       └── Slide 8 of 1_Region 001_qupath_mask.npy
└── processed/  # Auto-created
```

### Option 2: Single Directory
All files in one directory:
```
my_experiment/
├── Sample01_CY5.tif
├── Sample01_CY3.tif
├── Sample01_DAPI.tif
├── Sample01_CD11b.tif
├── Sample01_CY5_cellpose_masks.npy
├── Sample01_DAPI_cellpose_masks.npy
├── Sample01_qupath_mask.npy
└── processed/  # Auto-created
```

## Organizing Raw Data

Use the provided bash script to organize your data:
```bash
./organize_data.sh -i /path/to/raw/images -o /path/to/organized/data
```

This will create the recommended channel-separated directory structure automatically.

## Usage Examples

### For Channel-Separated Directories (Recommended)

```python
from lmtools.seg import create_data_paths

# For organized data with channel subdirectories
data_paths = create_data_paths(
    base_dir="/path/to/my_experiment",
    base_name="Slide 8 of 1_Region 001",
    experiment_name="Immune Cell Study",
    sample_id="Slide8_Region001",
    # Specify channel subdirectories
    channel_dirs={
        'cy5': 'raw_images/CY5',
        'cy3': 'raw_images/CY3',
        'dapi': 'raw_images/DAPI',
        'cd11b': 'raw_images/CD11b'
    },
    # Specify segmentation subdirectories
    seg_dirs={
        'cy5_seg': 'segmentations/CY5',
        'cy3_seg': 'segmentations/CY3',
        'dapi_seg': 'segmentations/DAPI',
        'qupath_seg': 'segmentations/QuPath'
    }
)

# Access your data
print(f"CY5 image: {data_paths.cy5_img}")
print(f"DAPI segmentation: {data_paths.dapi_seg}")
```

### For Single Directory

```python
# For flat directory structure (all files in one place)
data_paths = create_data_paths(
    base_dir="/path/to/my_experiment",
    base_name="Sample01",
    experiment_name="Immune Cell Study",
    sample_id="Mouse1_Brain_S1"
)
```

## Custom Channel Names

If your channels have different names, customize the suffixes:

```python
data_paths = create_data_paths(
    base_dir="/path/to/data",
    base_name="Sample01",
    experiment_name="My Experiment",
    sample_id="S1",
    channel_suffixes={
        'cy5': '_CD45.tif',      # Instead of _CY5.tif
        'cy3': '_CD68.tif',      # Instead of _CY3.tif
        'dapi': '_Hoechst.tif',  # Instead of _DAPI.tif
        'cd11b': '_CD11b.tif'    # Keep default
    },
    seg_suffixes={
        'cy5_seg': '_CD45_segmentation.npy',
        'dapi_seg': '_nuclei_mask.npy',
        # ... etc
    }
)
```

## Best Practices

1. **Consistent Naming**: Use the same base name for all files from one sample
2. **No Spaces**: Avoid spaces in file names (use underscores or hyphens)
3. **Descriptive Base Names**: Include relevant info (e.g., `Mouse01_Ctx_Section03`)
4. **Channel Order**: Keep channel order consistent across experiments
5. **Backup Raw Data**: Keep original images separate from processed data

## Batch Processing Structure

For experiments with multiple samples:

```
experiment_2024/
├── batch_metadata.json      # Overall experiment metadata
├── Sample01/
│   ├── Sample01_CY5.tif
│   ├── Sample01_DAPI.tif
│   ├── Sample01_CY5_cellpose_masks.npy
│   └── processed/
├── Sample02/
│   ├── Sample02_CY5.tif
│   ├── Sample02_DAPI.tif
│   ├── Sample02_CY5_cellpose_masks.npy
│   └── processed/
└── ...
```

## Output Files (Auto-generated)

The DataPaths system will create:

1. **Processed masks**: `{base_name}_{description}.npy`
   - Example: `Sample01_cy5_dapi_filtered.npy`
   - Example: `Sample01_final_immune_cells.npy`

2. **Metadata**: `{base_name}_metadata.json`
   - Complete processing history
   - Parameter values
   - Input/output file paths
   - Timestamps

3. **Directory**: `processed/`
   - Created automatically
   - Contains all outputs

## Metadata File Structure

The auto-generated metadata includes:
```json
{
  "experiment_name": "Immune Cell Study",
  "sample_id": "Mouse1_Brain_S1",
  "acquisition_date": "2024-01-15",
  "processing_date": "2024-01-20T14:30:00",
  "channel_mappings": {
    "cy5": "Sample01_CY5.tif",
    "dapi": "Sample01_DAPI.tif"
  },
  "segmentation_sources": {
    "cy5_seg": "/path/to/Sample01_CY5_cellpose_masks.npy"
  },
  "intensity_filter_sources": {
    "cd11b_filter": "cd11b"
  },
  "processing_steps": [
    {
      "step_name": "cy5_dapi_overlap_filter",
      "timestamp": "2024-01-20T14:30:15",
      "parameters": {
        "min_overlap_ratio": 0.5,
        "removed_objects": 45,
        "remaining_objects": 123
      }
    }
  ]
}
```

## Troubleshooting

### File Not Found Errors
1. Check file naming matches the pattern
2. Verify files are in the correct directory
3. Use `data_paths.get_all_paths_dict()` to see what paths are being searched

### Custom Patterns
If your files don't match standard patterns, explicitly set paths:
```python
# Override specific suffixes
data_paths.channel_suffixes['cy5'] = '_mymarker.tif'
data_paths.seg_suffixes['cy5_seg'] = '_mysegmentation.npy'
```

### Missing Files
The system will raise clear errors indicating which file pattern it couldn't find, making it easy to correct naming issues.

## Next Steps

For a complete walkthrough of organizing and processing your data, see the [Data Processing Workflow Guide](data_processing_workflow_guide.md).

### Quick Reference

1. **Organize data**: `./organize_data.sh -i input_dir -o output_dir`
2. **Create DataPaths**: Use `channel_dirs` and `seg_dirs` for organized data
3. **Process data**: Load images/segmentations and run your analysis
4. **Save results**: Use `data_paths.save_processed_mask()` and `data_paths.save_metadata()`