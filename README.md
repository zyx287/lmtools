# lmtools

Tools and scripts for processing and visualizing light microscopy data.

## Installation

```bash
# Basic installation
pip install .

# With cellpose support
pip install .[cellpose]

# With development tools
pip install .[dev]
```

## Features

- **I/O Operations**: Load and visualize microscopy data formats (ND2)
- **Segmentation Tools**: 
  - Generate masks from QuPath GeoJSON annotations
  - Extract masks from Cellpose outputs
  - Run Cellpose segmentation pipelines with configuration files
- **Analysis Tools**:
  - Analyze segmentation masks for statistics
  - Extract bounding boxes for objects
  - Summarize segmentation results

## Usage Examples

### Loading ND2 Files

#### As a Python package

```python
from lmtools import load_nd2

# Load and visualize an ND2 file
viewer = load_nd2("path/to/microscopy_image.nd2")

# Access the viewer object for further customization
viewer.add_shapes(...)
viewer.screenshot("screenshot.png")
```

#### From the command line

```bash
# Load and visualize an ND2 file
lmtools load_nd2 path/to/microscopy_image.nd2
```

### Generating Segmentation Masks from QuPath

#### As a Python package

```python
from lmtools import generate_segmentation_mask

# Generate masks from QuPath GeoJSON annotations
success = generate_segmentation_mask(
    geojson_path="annotations.geojson",
    output_dir="output_masks",
    image_width=1024,
    image_height=768,
    inner_holes=True,
    downsample_factor=2
)

if success:
    print("Masks generated successfully!")
```

#### From the command line

```bash
# Generate masks with inner holes and 2x downsampling
lmtools generate_mask annotations.geojson output_masks 1024 768 --inner_holes --downsample_factor 2
```

### Extracting Masks from Cellpose Output

#### As a Python package

```python
from lmtools import maskExtract

# Extract the mask from a Cellpose output file
mask = maskExtract(
    file_path="cellpose_output.npy",
    output_path="extracted_mask.npy"
)
```

#### From the command line

```bash
# Extract mask from Cellpose output
lmtools extract_mask --file_path cellpose_output.npy --output_path extracted_mask.npy
```

### Analyzing Segmentation Results

#### As a Python package

```python
from lmtools import analyze_segmentation, summarize_segmentation

# Analyze a segmentation mask
results = analyze_segmentation(
    mask="segmentation_mask.npy",
    compute_object_stats=True,
    min_size=10  # Ignore objects smaller than 10 pixels
)

# Print a human-readable summary
summary = summarize_segmentation(results, print_summary=True)

# Access specific information
num_objects = results['num_objects']
total_area = results['total_mask_area']
object_areas = results['object_stats']['area'].tolist()  # List of all object areas
```

#### From the command line

```bash
# Basic analysis
lmtools analyze_segmentation segmentation_mask.npy

# Skip small objects and save results to JSON
lmtools analyze_segmentation segmentation_mask.npy --min-size 10 --output results.json

# Don't compute per-object statistics (faster for large masks)
lmtools analyze_segmentation segmentation_mask.npy --no-stats
```

### Getting Bounding Boxes

#### As a Python package

```python
from lmtools import get_bounding_boxes

# Get all bounding boxes in a mask
all_boxes = get_bounding_boxes('segmentation_mask.npy', return_format='coords')
# Returns: {1: [y_min, x_min, y_max, x_max], 2: [...], ...}

# Get bounding box for a specific object (label ID 5)
box = get_bounding_boxes('segmentation_mask.npy', label_id=5, return_format='coords')
# Returns: [y_min, x_min, y_max, x_max]

# Get bounding boxes as slice objects (useful for cropping)
slices = get_bounding_boxes('segmentation_mask.npy', return_format='slices')
# Use with NumPy: mask[slices[0]] gives you the first labeled object
```

#### From the command line

```bash
# Get all bounding boxes and save to JSON
lmtools analyze_segmentation segmentation_mask.npy --bbox --output boxes.json

# Get bounding box for a specific label
lmtools analyze_segmentation segmentation_mask.npy --bbox --label 5
```

### Running Cellpose Segmentation Pipeline

#### As a Python package

```python
from lmtools import run_pipeline

# Run segmentation with a config file
output_files = run_pipeline("cellpose_config.yaml")

# Process the results
print(f"Generated {len(output_files)} mask files")
```

#### From the command line

```bash
# Basic usage with config file
lmtools cellpose_segment cellpose_config.yaml

# Enable verbose logging
lmtools cellpose_segment cellpose_config.yaml --verbose
```

## Configuration Files

### Cellpose Configuration Example

```yaml
# Model Configuration
model:
  # Path to custom model
  path: "/path/to/your/custom/model"
  # Alternative: use a built-in pretrained model
  # pretrained_model: "cyto"
  
# Input Configuration
input:
  # List of directories to process
  directories:
    - "/path/to/first/directory"
    - "/path/to/second/directory"

# Segmentation Parameters
segmentation_params:
  # Channels to use: [channel_to_segment, optional_nuclear_channel]
  channels: [1, 0]  # For RGB, use green channel for segmentation
  
  # Optional: Override model's default diameter
  # diameter: 30.0
  
  # Parameters for sensitivity adjustment
  flow_threshold: 0.4
  cellprob_threshold: 0.0

# Output Configuration
output:
  # Skip files containing this pattern
  exclude_pattern: "_masks"
  
  # Suffix to add to output files
  suffix: "_masks"
  
  # Clear GPU memory after processing each image
  clear_cache: true

# Force GPU usage
force_gpu: false
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.