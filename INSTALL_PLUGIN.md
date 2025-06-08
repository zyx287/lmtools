# Installing LM Tools as a Napari Plugin

Follow these steps to install and use lmtools as a napari plugin:

## Step 1: Clean Installation

First, make sure to uninstall any previous versions:

```bash
pip uninstall lmtools -y
```

## Step 2: Install the Package

From the lmtools directory, install in development mode:

```bash
# Basic installation
pip install -e .

# Or with Cellpose support
pip install -e ".[cellpose]"
```

## Step 3: Verify Plugin Installation

Check that napari recognizes the plugin:

```bash
# List all installed napari plugins
python -c "from napari.plugins import plugin_manager; print([p for p in plugin_manager.iter_available()])"
```

You should see 'lmtools' in the list.

## Step 4: Start Napari

```bash
napari
```

## Step 5: Access the Plugin

In napari:
1. Go to `Plugins` menu
2. Look for `lmtools` submenu
3. Select any of the available widgets

## Troubleshooting

### Plugin Not Showing Up

1. **Check installation location**:
   ```bash
   pip show lmtools
   ```

2. **Verify plugin files exist**:
   ```bash
   ls -la lmtools/napari_plugin/
   ```
   You should see:
   - `__init__.py`
   - `_widgets.py`
   - `napari.yaml`

3. **Check for import errors**:
   ```bash
   python -c "from lmtools.napari_plugin import *"
   ```

4. **Reinstall with verbose output**:
   ```bash
   pip install -e . -v
   ```

5. **Clear napari plugin cache**:
   ```bash
   rm -rf ~/.napari/
   ```

### Alternative: Manual Plugin Discovery

If automatic discovery fails, you can manually register the plugin:

```python
import napari
from lmtools.napari_plugin import (
    cellpose_segmentation_widget,
    basic_segmentation_widget,
    # ... other widgets
)

viewer = napari.Viewer()

# Add widgets manually
viewer.window.add_dock_widget(
    cellpose_segmentation_widget(), 
    name="Cellpose Segmentation"
)
```

### Common Issues

1. **Missing dependencies**: Make sure all dependencies are installed:
   ```bash
   pip install napari napari-plugin-engine magicgui scikit-image
   ```

2. **Python version**: Napari requires Python 3.8+

3. **Qt backend issues**: On some systems, you may need to set:
   ```bash
   export QT_API=pyqt5
   ```

## Testing the Plugin

Once installed, test with a simple example:

```python
import napari
import numpy as np

# Create viewer
viewer = napari.Viewer()

# Add test image
test_image = np.random.rand(512, 512)
viewer.add_image(test_image, name='test')

# Plugin widgets should now be available in Plugins menu
napari.run()
```