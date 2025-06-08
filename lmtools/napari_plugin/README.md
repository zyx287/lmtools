# LM Tools Napari Plugin

The LM Tools package is now available as a napari plugin, providing an interactive GUI for all the main functionality.

## Installation

```bash
# Install lmtools with napari plugin support
pip install -e .

# For Cellpose functionality, also install:
pip install -e ".[cellpose]"
```

## Usage

1. **Start napari**:
   ```bash
   napari
   ```

2. **Access LM Tools widgets**:
   - Go to `Plugins` → `lmtools` in the napari menu
   - Select the desired tool from the submenu

## Available Widgets

### 1. Load ND2 File
Load and visualize ND2 microscopy files directly in napari.

### 2. Cellpose Segmentation
Deep learning-based cell segmentation with customizable parameters:
- Model type (cyto, cyto2, nuclei, cyto3)
- Cell diameter
- Flow and cell probability thresholds
- GPU acceleration support

### 3. Basic Segmentation
Traditional segmentation methods:
- Threshold-based segmentation
- Watershed (coming soon)
- Region growing (coming soon)

### 4. Split Channels
Automatically split multi-channel images into separate layers with appropriate colormaps.

### 5. Filter by Intensity
Filter segmented objects based on intensity criteria:
- Min/max intensity thresholds
- Percentile-based filtering

### 6. Analyze Segmentation
Analyze segmented objects and display statistics:
- Area, perimeter, eccentricity
- Mean intensity (when image provided)
- Summary statistics

### 7. Downsample Image
Reduce image size by a specified factor for faster processing.

### 8. Generate Mask from QuPath
Import QuPath GeoJSON annotations as segmentation masks.

## Example Workflow

1. **Load an image**: Use `File → Open` or the Load ND2 widget
2. **Segment cells**: Select the image layer and use Cellpose Segmentation
3. **Filter results**: Use Filter by Intensity to remove dim or bright objects
4. **Analyze**: Use Analyze Segmentation to get object statistics

## Tips

- Most widgets operate on the currently selected layer
- For multi-channel images, split channels first for better segmentation
- GPU acceleration significantly speeds up Cellpose segmentation
- Use downsampling for quick previews before running on full resolution images

## Programmatic Usage

You can also use the widgets programmatically:

```python
import napari
from lmtools.napari_plugin import cellpose_segmentation_widget

viewer = napari.Viewer()
# Add your image
viewer.add_image(your_image_data)

# Create and show widget
widget = cellpose_segmentation_widget()
viewer.window.add_dock_widget(widget)
```