'''
Widget implementations for lmtools napari plugin.
'''

from pathlib import Path
from typing import Optional, List, Tuple, Union
import numpy as np
import nd2
from magicgui import magic_factory
from napari.types import ImageData, LabelsData, LayerDataTuple
import napari
import tempfile
import yaml

from lmtools.io.channel_splitting import split_channels
from lmtools.io.image_downsampling import downsample_image
from lmtools.seg import (
    run_pipeline,
    threshold_segment,
    watershed_segment,
    region_growing_segment,
    intensity_filter,
    analyze_segmentation,
    summarize_segmentation,
    generate_segmentation_mask,
)


@magic_factory(
    call_button="Load ND2",
    nd2_path={"label": "ND2 File", "mode": "r", "filter": "*.nd2"},
)
def load_nd2_widget(
    nd2_path: Path,
    viewer: napari.Viewer,
) -> None:
    '''Load ND2 file into napari viewer.'''
    if nd2_path and nd2_path.exists():
        try:
            # Load the ND2 file
            with nd2.ND2File(str(nd2_path)) as nd2_file:
                image_data = nd2_file.asarray()
            
            # Add to the current viewer instead of creating a new one
            viewer.add_image(image_data, name=f"ND2: {nd2_path.name}")
        except Exception as e:
            import warnings
            warnings.warn(f"Error loading ND2 file: {e}")


@magic_factory(
    call_button="Run Cellpose",
    config_path={"label": "Config File", "mode": "r", "filter": "*.yaml;*.yml", "nullable": True},
    save_masks={"label": "Save Masks", "value": False},
    output_dir={"label": "Output Directory", "mode": "d", "visible": False},
)
def cellpose_segmentation_widget(
    image: ImageData,
    config_path: Optional[Path] = None,
    save_masks: bool = False,
    output_dir: Optional[Path] = None,
    viewer: napari.Viewer = None,
) -> LayerDataTuple:
    '''Run Cellpose segmentation on the selected image layer.'''
    try:
        from cellpose import models
    except ImportError:
        raise ImportError("cellpose not installed. Install with: pip install cellpose torch")
    
    # Create temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Save image to temporary file
        import tifffile
        temp_image_path = temp_path / "temp_image.tif"
        tifffile.imwrite(str(temp_image_path), image)
        
        # Create or use config
        if config_path and config_path.exists():
            config_file = config_path
        else:
            # Create a default config
            default_config = {
                "directories": [str(temp_path)],
                "model": {
                    "model_type": "cyto2",
                    "channels": [0, 0],  # grayscale
                },
                "parameters": {
                    "diameter": 30,
                    "flow_threshold": 0.4,
                    "cellprob_threshold": 0.0,
                },
                "output": {
                    "suffix": "_cellpose_masks"
                },
                "force_gpu": True,
                "clear_cache": True,
            }
            config_file = temp_path / "temp_config.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(default_config, f)
        
        # Run cellpose pipeline
        output_files = run_pipeline(str(config_file))
        
        # Load the generated mask
        if output_files:
            masks = np.load(output_files[0])
            
            # Save if requested
            if save_masks and output_dir:
                output_path = output_dir / f"{Path(viewer.layers[image].name).stem}_cellpose_masks.npy"
                np.save(output_path, masks)
            
            return (masks, {"name": "cellpose_segmentation", "opacity": 0.5}, "labels")
        else:
            raise RuntimeError("Cellpose segmentation failed to produce output")


@magic_factory(
    call_button="Create Config",
    output_path={"label": "Save Config As", "mode": "w", "filter": "*.yaml;*.yml"},
    directories={"label": "Image Directories", "mode": "d"},
    model_type={"choices": ["cyto", "cyto2", "nuclei", "cyto3"], "label": "Model Type", "value": "cyto2"},
    diameter={"label": "Cell Diameter", "min": 0, "max": 1000, "value": 30},
    flow_threshold={"label": "Flow Threshold", "min": 0.0, "max": 10.0, "value": 0.4, "step": 0.1},
    cellprob_threshold={"label": "Cell Probability", "min": -10.0, "max": 10.0, "value": 0.0, "step": 0.5},
)
def create_cellpose_config_widget(
    output_path: Path,
    directories: Path,
    model_type: str = "cyto2",
    diameter: int = 30,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
) -> None:
    '''Create a Cellpose configuration file.'''
    config = {
        "directories": [str(directories)],
        "model": {
            "model_type": model_type,
            "channels": [0, 0],  # grayscale
        },
        "parameters": {
            "diameter": diameter,
            "flow_threshold": flow_threshold,
            "cellprob_threshold": cellprob_threshold,
        },
        "output": {
            "suffix": "_cellpose_masks"
        },
        "force_gpu": True,
        "clear_cache": True,
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f)
    
    print(f"Config saved to: {output_path}")


@magic_factory(
    call_button="Run Segmentation",
    method={"choices": ["threshold", "watershed", "region_growing"], "label": "Method"},
    threshold_method={"choices": ["otsu", "simple", "adaptive", "yen", "li", "triangle"], "label": "Threshold Method", "visible": True},
    threshold_value={"label": "Threshold", "min": 0.0, "max": 1.0, "value": 0.5, "visible": False},
    min_size={"label": "Min Object Size", "min": 0, "max": 10000, "value": 50},
    save_result={"label": "Save Result", "value": False},
    output_path={"label": "Output Path", "mode": "w", "filter": "*.tif;*.tiff", "visible": False},
)
def basic_segmentation_widget(
    image: ImageData,
    method: str = "threshold",
    threshold_method: str = "otsu",
    threshold_value: float = 0.5,
    min_size: int = 50,
    save_result: bool = False,
    output_path: Optional[Path] = None,
) -> LayerDataTuple:
    '''Run basic segmentation on the selected image layer.'''
    # Update visibility based on method
    if method == "threshold" and threshold_method == "simple":
        # Would need dynamic visibility update here
        pass
    
    if method == "threshold":
        labels = threshold_segment(
            image=image,
            output_path=output_path if save_result else None,
            method=threshold_method,
            threshold_value=threshold_value if threshold_method == "simple" else None,
            min_size=min_size,
            return_labels=True
        )
    elif method == "watershed":
        labels = watershed_segment(
            image=image,
            output_path=output_path if save_result else None,
            markers_method="peaks",
            min_distance=10,
            threshold_abs=threshold_value,
            compactness=0.0,
            min_size=min_size
        )
    elif method == "region_growing":
        # For region growing, we need seed points - use peaks
        labels = region_growing_segment(
            image=image,
            output_path=output_path if save_result else None,
            seed_method="peaks",
            tolerance=0.1,
            min_size=min_size
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return (labels, {"name": f"{method}_segmentation", "opacity": 0.5}, "labels")


@magic_factory(
    call_button="Split Channels",
    save_channels={"label": "Save Split Channels", "value": False},
    output_dir={"label": "Output Directory", "mode": "d", "visible": False},
)
def split_channels_widget(
    image: ImageData,
    viewer: napari.Viewer,
    save_channels: bool = False,
    output_dir: Optional[Path] = None,
) -> None:
    '''Split multi-channel image into separate layers.'''
    if image.ndim < 3:
        raise ValueError("Image must have at least 3 dimensions")
    
    if save_channels and output_dir:
        # Save to disk using the actual function
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            import tifffile
            tifffile.imwrite(tmp.name, image)
            
            # Call the actual split_channels function
            channel_files = split_channels(
                input_path=tmp.name,
                output_dir=str(output_dir),
                channel_names=None
            )
            
            # Load and display the split channels
            for ch_name, ch_path in channel_files.items():
                ch_data = tifffile.imread(ch_path)
                viewer.add_image(ch_data, name=ch_name)
    else:
        # Just display in viewer without saving
        # Assume last dimension is channels if it's small (<=4)
        if image.shape[-1] <= 4:
            for i in range(image.shape[-1]):
                viewer.add_image(
                    image[..., i],
                    name=f"Channel {i}",
                    colormap=["red", "green", "blue", "gray"][i] if i < 4 else "gray"
                )
        # Otherwise assume first dimension is channels
        elif image.shape[0] <= 10:
            for i in range(image.shape[0]):
                viewer.add_image(
                    image[i],
                    name=f"Channel {i}",
                    colormap=["red", "green", "blue", "gray"][i] if i < 4 else "gray"
                )
        else:
            raise ValueError("Cannot determine channel dimension")


@magic_factory(
    call_button="Filter by Intensity",
    threshold_method={"choices": ["manual", "otsu", "percentile", "adaptive"], "label": "Threshold Method"},
    min_intensity={"label": "Min Intensity", "min": 0, "max": 65535, "value": 0},
    max_intensity={"label": "Max Intensity", "min": 0, "max": 65535, "value": 65535},
    percentile_low={"label": "Low Percentile", "min": 0, "max": 100, "value": 10},
    percentile_high={"label": "High Percentile", "min": 0, "max": 100, "value": 90},
    region_type={"choices": ["whole", "membrane", "inner", "outer"], "label": "Region Type"},
    save_result={"label": "Save Result", "value": False},
    output_path={"label": "Output Path", "mode": "w", "filter": "*.tif;*.tiff", "visible": False},
)
def intensity_filter_widget(
    labels: LabelsData,
    image: ImageData,
    threshold_method: str = "manual",
    min_intensity: int = 0,
    max_intensity: int = 65535,
    percentile_low: int = 10,
    percentile_high: int = 90,
    region_type: str = "whole",
    save_result: bool = False,
    output_path: Optional[Path] = None,
) -> LayerDataTuple:
    '''Filter segmented objects based on intensity criteria.'''
    # Prepare threshold values based on method
    if threshold_method == "manual":
        threshold_values = (min_intensity, max_intensity)
    elif threshold_method == "percentile":
        threshold_values = (percentile_low, percentile_high)
    else:
        threshold_values = None
    
    # Call the actual intensity_filter function
    filtered, stats = intensity_filter(
        segmentation=labels,
        intensity_image=image,
        output_path=str(output_path) if save_result and output_path else None,
        threshold_method=threshold_method,
        threshold_values=threshold_values,
        region_type=region_type,
        membrane_width=5 if region_type == "membrane" else None,
        shrink_distance=3 if region_type in ["inner", "outer"] else None,
    )
    
    # Print stats
    print(f"\nIntensity Filtering Results:")
    print(f"Original objects: {stats['original_count']}")
    print(f"Filtered objects: {stats['filtered_count']}")
    print(f"Removed objects: {stats['removed_count']}")
    
    return (filtered, {"name": f"filtered_{region_type}", "opacity": 0.5}, "labels")


@magic_factory(
    call_button="Analyze",
    save_results={"label": "Save Results", "value": False},
    output_path={"label": "Output Path", "mode": "w", "filter": "*.json", "visible": False},
)
def analyze_segmentation_widget(
    labels: LabelsData,
    image: Optional[ImageData] = None,
    save_results: bool = False,
    output_path: Optional[Path] = None,
) -> None:
    '''Analyze segmentation and display results.'''
    # Call the actual analyze_segmentation function
    results = analyze_segmentation(
        segmentation=labels,
        intensity_image=image,
        output_path=str(output_path) if save_results and output_path else None,
    )
    
    # Display summary
    summary = summarize_segmentation(results)
    print("\n" + "="*60)
    print("SEGMENTATION ANALYSIS RESULTS")
    print("="*60)
    print(summary)


@magic_factory(
    call_button="Downsample",
    factor={"label": "Downsample Factor", "min": 1, "max": 32, "value": 2},
    method={"choices": ["nearest", "linear", "cubic"], "label": "Interpolation Method"},
    library={"choices": ["scipy", "opencv", "pillow"], "label": "Library"},
    save_result={"label": "Save Result", "value": False},
    output_path={"label": "Output Path", "mode": "w", "filter": "*.tif;*.tiff", "visible": False},
)
def downsample_widget(
    image: ImageData,
    factor: int = 2,
    method: str = "linear",
    library: str = "scipy",
    save_result: bool = False,
    output_path: Optional[Path] = None,
) -> LayerDataTuple:
    '''Downsample image by the specified factor.'''
    # Call the actual downsample_image function
    downsampled = downsample_image(
        image=image,
        downsampling_factor=factor,
        output_path=str(output_path) if save_result and output_path else None,
        method=method,
        preserve_range=True,
        library=library,
    )
    
    return (downsampled, {"name": f"downsampled_{factor}x"}, "image")


@magic_factory(
    call_button="Generate Mask",
    geojson_path={"label": "GeoJSON File", "mode": "r", "filter": "*.geojson"},
    output_dir={"label": "Output Directory", "mode": "d"},
    image_width={"label": "Image Width", "min": 1, "max": 50000, "value": 1024},
    image_height={"label": "Image Height", "min": 1, "max": 50000, "value": 1024},
    downsample_factor={"label": "Downsample Factor", "min": 0.1, "max": 10.0, "value": 1.0, "step": 0.1},
    fill_holes={"label": "Fill Inner Holes", "value": True},
)
def generate_mask_widget(
    geojson_path: Path,
    output_dir: Path,
    image_width: int = 1024,
    image_height: int = 1024,
    downsample_factor: float = 1.0,
    fill_holes: bool = True,
    viewer: napari.Viewer = None,
) -> None:
    '''Generate mask from QuPath GeoJSON annotations.'''
    if geojson_path and geojson_path.exists() and output_dir:
        # Call the actual function which saves to disk
        generate_segmentation_mask(
            geojson_path=str(geojson_path),
            output_dir=str(output_dir),
            image_width=image_width,
            image_height=image_height,
            downsample_factor=downsample_factor,
            inner_holes=not fill_holes,  # Function parameter is opposite
        )
        
        # Load the generated mask to display
        import tifffile
        from pathlib import Path
        
        # Find the generated mask file
        mask_files = list(output_dir.glob("*_mask.tif"))
        if mask_files:
            mask = tifffile.imread(mask_files[0])
            viewer.add_labels(mask, name=f"QuPath mask: {geojson_path.stem}")
            print(f"Mask saved to: {mask_files[0]}")
        else:
            import warnings
            warnings.warn("Mask file was not found after generation")
    else:
        raise ValueError("Please select valid GeoJSON file and output directory")