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

### Image down-sampling

#### As a package
```python
from lmtools import downsample_image, batch_downsample

# Downsample a single image to 50% size using bicubic interpolation
downsampled = downsample_image(
    "input.tif", 
    "output.tif",
    scale_factor=0.5,
    method="bicubic"
)

# Downsample all TIFF images in a directory
# Using Lanczos algorithm for highest quality
batch_downsample(
    "input_directory",
    "output_directory",
    scale_factor=0.25,  # Reduce to 25% of original size
    method="lanczos",
    recursive=True      # Process subdirectories too
)

# Advanced use: Different scale factors for each dimension
# This creates a non-uniform scaling
downsampled = downsample_image(
    "input.tif",
    "output.tif",
    scale_factor=(0.5, 0.75),  # Scale height by 0.5, width by 0.75
    method="gaussian",         # Use Gaussian pre-filtering for smoother results
    library="skimage"          # Force use of scikit-image
)
```

#### Command line
```bash
# Basic usage - downsample a single image with default settings (50% size, bicubic)
lmtools downsample input.tif output.tif

# Batch process a directory using lanczos algorithm
lmtools downsample input_directory/ output_directory/ --method lanczos --recursive

# Significant downsampling using area method (better for small output sizes)
lmtools downsample input.tif output.tif --scale 0.1 --method area --library opencv

# Enable verbose logging for troubleshooting
lmtools downsample input.tif output.tif --verbose
```
### Channel splitting
#### Package
```python
from lmtools import split_channels

# Split a multi-channel image using custom names
output_files = split_channels(
    "microscopy_image.tif",
    channel_names=["R", "G", "CY5"]
)

# Process all images in a directory
from lmtools import batch_split_channels

batch_split_channels(
    "input_directory",
    output_dir="output_directory",
    channel_names=["DAPI", "GFP", "mCherry"],
    recursive=True
)
```
#### CLI
```bash
# Basic usage - split a single image with default channel names
lmtools split_channels multi_channel_image.tif

# Specify output directory and custom channel names
lmtools split_channels multi_channel_image.tif --output ./channels/ --sequence R G CY5

# Batch process with recursive search
lmtools split_channels ./data/ --recursive --sequence R G CY5 --verbose
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

### Basic segmentation

#### As a package
```python
from lmtools import threshold_segment, watershed_segment, region_growing_segment

# Simple thresholding with Otsu's method
labels = threshold_segment(
    "input.tif",
    "output.tif",
    method="otsu",
    fill_holes=True,
    remove_small_objects=True,
    min_size=50
)

# Watershed segmentation for touching objects
labels = watershed_segment(
    "input.tif",
    "output.tif",
    threshold_method="otsu",
    distance_transform=True,
    min_distance=10
)

# Region growing for complex textures
labels = region_growing_segment(
    "input.tif",
    "output.tif",
    seed_method="intensity",
    num_seeds=100,
    threshold_method="otsu"
)
```

#### Command line
```bash
# Threshold segmentation using Otsu's method
lmtools basic_segment threshold input.tif output.tif --method otsu --min-size 50

# Adaptive thresholding for uneven illumination
lmtools basic_segment threshold input.tif output.tif --method adaptive --block-size 51

# Watershed segmentation for separating touching objects
lmtools basic_segment watershed input.tif output.tif --min-distance 15

# Region growing with SLIC superpixels
lmtools basic_segment region input.tif output.tif --num-seeds 200 --compactness 0.05
```
### Intensity filter
#### Package usage
```python
from lmtools import intensity_filter, visualize_intensity_regions

# Basic filtering - remove low-intensity objects
filtered_mask = intensity_filter(
    "segmentation.tif",                # Segmentation mask with labeled objects
    "intensity_image.tif",             # Corresponding intensity image
    "filtered_segmentation.tif",       # Output path
    threshold_method='otsu',           # Automatic threshold using Otsu's method
    plot_histogram=True                # Show histogram of intensities
)

# Membrane analysis - focus on cell borders
filtered_mask = intensity_filter(
    "segmentation.tif",
    "intensity_image.tif",
    "membrane_filtered.tif",
    region_type='membrane',            # Only consider membrane/border regions
    membrane_width=3,                  # 3-pixel membrane width
    threshold=0.25                     # Manual threshold (0-1 scale)
)

# Visualize the different regions for quality control
visualization = visualize_intensity_regions(
    "segmentation.tif",
    "intensity_image.tif",
    "region_visualization.png",
    label_id=5                         # Visualize regions for object #5
)
```

#### Command line
```bash
# Basic filtering with automatic threshold
lmtools intensity_filter segmentation.tif intensity_image.tif filtered.tif

# Analyze only membrane regions (3px wide) with manual threshold
lmtools intensity_filter segmentation.tif intensity_image.tif membrane_filtered.tif \
    --region membrane --membrane-width 3 --threshold 0.25

# Use percentile-based thresholding (remove bottom 30%)
lmtools intensity_filter segmentation.tif intensity_image.tif filtered.tif \
    --threshold-method percentile --percentile 30

# Visualize the analyzed regions
lmtools intensity_filter segmentation.tif intensity_image.tif filtered.tif \
    --visualize-regions --vis-output regions.png
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

Many thanks to **Claude**