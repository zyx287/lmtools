# LMTools

A Python package for light microscopy image analysis.

## Installation

```bash
git clone https://github.com/zyx287/lmtools.git

cd lmtools

pip install .

# With cellpose support
pip install .[cellpose]

# Development installation
pip install -e .[dev]
```

## Quick Start

```bash
# Organize microscopy data from TGXS Slide Scanner
lmtools organize-data -s /raw/images -o /organized --all

# Run cellpose segmentation
lmtools cellpose-segment config.yaml

# Load and visualize ND2 files using Napari
lmtools load-nd2 image.nd2
```

## Features

- **Data Organization**: Automated workflow for organizing microscopy images
- **Image I/O**: Load various microscopy formats (ND2, TIFF)
- **Segmentation**: Multiple segmentation methods including cellpose integration
- **Image Processing**: Channel splitting, downsampling, transformations
- **Analysis Tools**: Intensity filtering, segmentation analysis, cell counting
- **Napari Plugin**: Interactive GUI for all features

---

## Data Organization Workflow

Organize raw TGXS Slide Scanner images for batch processing and analysis.

### Expected Format
Files should be named: `{sample_id}_{CHANNEL}.tiff` (e.g., `Sample01_CY5.tiff`)

### Two-Step Organization

```bash
# Step 1: Organize by channel for batch segmentation
lmtools organize-data -s /raw/images -o /organized --step 1

# Step 2: Reorganize by sample after segmentation  
lmtools organize-data -s /raw/images -o /organized --step 2

# Or run both steps
lmtools organize-data -s /raw/images -o /organized --all
```

### Python API

```python
from lmtools.io import organize_data

# Organize data
channel_df, sample_df = organize_data('/raw/images', '/organized')
```

### Output Structure

```
organized/
├── channels_for_segmentation/    # Step 1
│   ├── CY5/
│   ├── CY3/
│   └── DAPI/
└── samples/                      # Step 2
    ├── Sample01/
    │   ├── raw_images/
    │   ├── segmentations/
    │   ├── results/
    │   └── sample_metadata.json
    └── master_sample_list.csv
```

---

## Image I/O Operations

### Loading ND2 Files

```python
from lmtools.io import load_nd2

# Load and visualize
viewer = load_nd2("microscopy_image.nd2")
```

```bash
lmtools load-nd2 microscopy_image.nd2
```

### Image Downsampling

```python
from lmtools.io import downsample_image, batch_downsample

# Single image
downsampled = downsample_image(
    "input.tif", 
    "output.tif",
    scale_factor=0.5,
    method="bicubic"
)

# Batch processing
batch_downsample(
    "input_directory",
    "output_directory", 
    scale_factor=0.25,
    method="lanczos",
    recursive=True
)
```

```bash
# Command line
lmtools downsample input.tif output.tif --scale 0.5 --method bicubic
lmtools downsample input_dir/ output_dir/ --recursive --method lanczos
# Significant downsampling using area method (better for small output sizes)
lmtools downsample input.tif output.tif --scale 0.1 --method area --library opencv
# Batch process a directory using lanczos algorithm
lmtools downsample input_directory/ output_directory/ --method lanczos --recursive

```

### Channel Splitting

```python
from lmtools.io import split_channels, batch_split_channels

# Split multi-channel image
output_files = split_channels(
    "multi_channel.tif",
    channel_names=["DAPI", "GFP", "mCherry"]
)

# Batch processing
batch_split_channels(
    "input_directory",
    output_dir="output_directory",
    channel_names=["DAPI", "GFP", "mCherry"],
    recursive=True
)
```

```bash
lmtools split-channels multi_channel.tif --sequence DAPI GFP mCherry
lmtools split-channels input_dir/ --recursive --sequence DAPI GFP mCherry
```

### Transform and Split

For images requiring dimension transformation:

```python
from lmtools.io import transform_and_split

# Transform from (C, Z, Y, X) to (C, X, Y, Z)
output_files = transform_and_split(
    "image.tif",
    channel_axis=0,
    transpose_axes=[0, 3, 2, 1],
    channel_names=["Ch1", "Ch2"]
)
```

```bash
lmtools transform-and-split image.tif --channel-axis 0 --transpose 0 3 2 1
```

---

## Segmentation Tools

### Basic Segmentation

```python
from lmtools.seg import basic_segmentation

# Threshold segmentation
mask = basic_segmentation(
    "input.tif",
    method="otsu",
    min_size=50,
    fill_holes=True
)

# Watershed segmentation
mask = basic_segmentation(
    "input.tif", 
    method="watershed",
    min_distance=10
)
```

```bash
lmtools basic-segment threshold input.tif output.tif --method otsu --min-size 50
lmtools basic-segment watershed input.tif output.tif --min-distance 15
```

### Cellpose Segmentation

```python
from lmtools.seg import cellpose_segmentation

# Using config file
masks = cellpose_segmentation("config.yaml")

# Direct usage
mask = cellpose_segmentation(
    image_path="cells.tif",
    model_type="cyto",
    diameter=30,
    channels=[1, 0]
)
```

Configuration file example:
```yaml
model:
  pretrained_model: "cyto"
  
input:
  directories:
    - "/path/to/images"

segmentation_params:
  channels: [1, 0]
  diameter: 30.0
  flow_threshold: 0.4
  cellprob_threshold: 0.0

output:
  suffix: "_masks"
```

### QuPath Integration

Generate masks from QuPath annotations:

```python
from lmtools.seg import generate_mask

success = generate_mask(
    geojson_path="annotations.geojson",
    output_dir="masks",
    image_width=1024,
    image_height=768,
    downsample_factor=2
)
```

```bash
lmtools generate-mask annotations.geojson masks 1024 768 --downsample 2
```

---

## Analysis Tools

### Segmentation Analysis

```python
from lmtools.seg import analyze_segmentation

# Analyze mask
results = analyze_segmentation(
    "segmentation.npy",
    compute_object_stats=True,
    min_size=10
)

print(f"Number of objects: {results['num_objects']}")
print(f"Average area: {results['avg_area']}")
```

```bash
lmtools analyze-segmentation mask.npy --min-size 10 --output results.json
```

### Intensity Filtering

```python
from lmtools.seg import intensity_filter

# Filter by intensity
filtered_mask = intensity_filter(
    segmentation_mask,
    intensity_image,
    threshold_method='otsu',
    region_type='whole'  # or 'membrane'
)
```

```bash
lmtools intensity-filter mask.tif intensity.tif filtered.tif --method otsu
```

---

## Tissue Scanner Image Analysis

Specialized workflows for tissue scanner images with multiple fluorescence channels.

### DataPaths System

Automatic file discovery and metadata tracking:

```python
from lmtools.seg import create_data_paths

# Create data paths with automatic discovery
data_paths = create_data_paths(
    base_dir="/path/to/sample",
    base_name="Sample01",
    experiment_name="Tissue Analysis",
    sample_id="S01"
)

# Load all data
img_cy5, img_dapi, img_cd11b = data_paths.load_imgs()
seg_cy5, seg_dapi, seg_cd11b = data_paths.load_segs()

# Save processed results
data_paths.save_processed_mask(filtered_mask, "filtered_cells")
data_paths.save_metadata()
```

### Cell Filtering Pipeline

```python
from lmtools.seg import filter_by_overlap, intensity_filter, compute_average_intensity

# 1. Filter cells by nuclear overlap
nuclear_positive = filter_by_overlap(
    cell_mask,
    nuclei_mask, 
    min_overlap_ratio=0.5,
    data_paths=data_paths,
    step_name="nuclear_filter"
)

# 2. Filter by marker intensity
intensities = compute_average_intensity(
    nuclear_positive,
    marker_image,
    use_donut=True,  # Membrane measurement
    erode_radius=2
)

marker_positive = intensity_filter(
    nuclear_positive,
    intensities,
    threshold_method="otsu",
    data_paths=data_paths,
    intensity_channel="cd11b"
)
```

### Batch Processing Tissue Samples

```python
import pandas as pd

# Load organized samples
samples = pd.read_csv('/organized/samples/master_sample_list.csv')

for _, row in samples.iterrows():
    if not row['has_cy5']:
        continue
        
    # Process each tissue section
    data_paths = create_data_paths(
        base_dir=row['sample_dir'],
        base_name=row['sample_id'],
        experiment_name="Tissue Scanner Batch"
    )
    
    # Run your analysis pipeline
    results = process_tissue_section(data_paths)
    
    # Results saved automatically with metadata
```

### Custom Channel Names

For tissue-specific markers:

```python
data_paths = create_data_paths(
    base_dir="/data",
    base_name="TissueSection_001",
    channel_suffixes={
        'cy5': '_CD45.tif',
        'cy3': '_CD68.tif', 
        'dapi': '_Hoechst.tif',
        'cd11b': '_CD11b.tif'
    }
)
```

---

## Napari Plugin

Interactive GUI access to all features:

1. Start napari: `napari`
2. Go to `Plugins` → `lmtools`
3. Available widgets:
   - Load ND2 files
   - Cellpose segmentation
   - Basic segmentation
   - Channel splitting
   - Intensity filtering
   - Segmentation analysis

---

## Examples

Complete examples in `examples/` directory:
- `complete_workflow_example.py` - Full pipeline from organization to analysis
- `immune_cell_processing_example.py` - Tissue scanner image analysis
- `create_example_file_structure.py` - Data organization examples

---

## Requirements

- Python ≥ 3.7
- numpy, scipy, pandas
- scikit-image, scikit-learn
- opencv-python, matplotlib
- napari (for GUI)
- cellpose (optional)

---
<!-- 
## License

MIT License
-->