'''
author: zyx
date: 2025-05-25
last modified: 2025-05-25
description: 
    Integrated functions for immune cell segmentation filtering
    Yupu pipeline
'''
import argparse
from pathlib import Path
import numpy as np
import tifffile
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field, asdict

from scipy.ndimage import binary_erosion, find_objects
from skimage.morphology import ball
from skimage.filters import threshold_otsu
from skimage.measure import label as sk_label
from skimage.segmentation import relabel_sequential

from ..compute.morphology import erode_mask_2D_with_ball, generate_2D_donut

import matplotlib.pyplot as plt

def load_segmentation(path: Path) -> np.ndarray:
    """Load a .npy segmentation mask."""
    return np.load(path, allow_pickle=True)

def load_image(path: Path) -> np.ndarray:
    """Load an image volume or slice (TIFF)."""
    return tifffile.imread(path)


@dataclass
class ProcessingStep:
    """Record a single processing step in the pipeline."""
    step_name: str
    timestamp: str
    parameters: Dict[str, Any]
    input_data: List[str]
    output_data: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class ImageMetadata:
    """Metadata for tracking image processing pipeline."""
    experiment_name: str
    sample_id: str
    acquisition_date: Optional[str] = None
    processing_date: str = field(default_factory=lambda: datetime.now().isoformat())
    processing_steps: List[ProcessingStep] = field(default_factory=list)
    channel_mappings: Dict[str, str] = field(default_factory=dict)
    segmentation_sources: Dict[str, str] = field(default_factory=dict)
    intensity_filter_sources: Dict[str, str] = field(default_factory=dict)
    notes: Optional[str] = None
    
    def add_step(self, step: ProcessingStep):
        """Add a processing step to the history."""
        self.processing_steps.append(step)
    
    def to_json(self, filepath: Path):
        """Save metadata to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def from_json(cls, filepath: Path):
        """Load metadata from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        # Convert processing steps back to ProcessingStep objects
        data['processing_steps'] = [ProcessingStep(**step) for step in data['processing_steps']]
        return cls(**data)


@dataclass
class DataPaths:
    """
    Enhanced data paths manager with automatic path discovery and metadata tracking.
    """
    base_dir: Path
    base_name: str
    metadata: ImageMetadata
    
    # Channel suffixes
    channel_suffixes: Dict[str, str] = field(default_factory=lambda: {
        'cy5': '_CY5.tif',
        'cy3': '_CY3.tif',
        'dapi': '_DAPI.tif',
        'cd11b': '_CD11b.tif'  # Can be customized
    })
    
    # Segmentation suffixes
    seg_suffixes: Dict[str, str] = field(default_factory=lambda: {
        'cy5_seg': '_CY5_cellpose_masks.npy',
        'cy3_seg': '_CY3_cellpose_masks.npy',
        'dapi_seg': '_DAPI_cellpose_masks.npy',
        'qupath_seg': '_qupath_mask.npy'
    })
    
    # Channel-specific directories (optional)
    channel_dirs: Optional[Dict[str, str]] = None
    seg_dirs: Optional[Dict[str, str]] = None
    
    # Processed output paths
    output_dir: Optional[Path] = None
    _processed_masks: Dict[str, Path] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize paths and create output directory if needed."""
        self.base_dir = Path(self.base_dir)
        if self.output_dir is None:
            self.output_dir = self.base_dir / 'processed'
        else:
            self.output_dir = Path(self.output_dir)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Update metadata channel mappings
        for channel, suffix in self.channel_suffixes.items():
            self.metadata.channel_mappings[channel] = f"{self.base_name}{suffix}"
    
    def get_image_path(self, channel: str) -> Path:
        """Get image path for a specific channel."""
        suffix = self.channel_suffixes.get(channel)
        if suffix is None:
            raise ValueError(f"Unknown channel: {channel}")
        
        # Check if channel-specific directory is defined
        if self.channel_dirs and channel in self.channel_dirs:
            channel_dir = self.base_dir / self.channel_dirs[channel]
            path = channel_dir / f"{self.base_name}{suffix}"
        else:
            path = self.base_dir / f"{self.base_name}{suffix}"
            
        if not path.exists():
            # Try alternative naming patterns
            search_dirs = []
            if self.channel_dirs and channel in self.channel_dirs:
                search_dirs.append(self.base_dir / self.channel_dirs[channel])
            else:
                search_dirs.append(self.base_dir)
                
            alt_patterns = [
                f"{self.base_name}_{channel.upper()}.tif",
                f"{self.base_name}_{channel.upper()}.tiff",
                f"{self.base_name}-{channel.upper()}.tif",
                f"{self.base_name}-{channel.upper()}.tiff",
                f"{channel.upper()}_{self.base_name}.tif",
                f"{channel.upper()}_{self.base_name}.tiff"
            ]
            
            found = False
            for search_dir in search_dirs:
                for pattern in alt_patterns:
                    alt_path = search_dir / pattern
                    if alt_path.exists():
                        path = alt_path
                        found = True
                        break
                if found:
                    break
            
            if not found:
                # Build list of all searched paths for error message
                searched_paths = [str(path)]  # Original path
                for search_dir in search_dirs:
                    for pattern in alt_patterns:
                        searched_paths.append(str(search_dir / pattern))
                
                error_msg = (f"Could not find {channel} image with base name '{self.base_name}'\n"
                           f"Searched in the following locations:\n" + 
                           "\n".join(f"  - {p}" for p in searched_paths))
                raise FileNotFoundError(error_msg)
        
        return path
    
    def get_seg_path(self, seg_type: str) -> Path:
        """Get segmentation path for a specific type."""
        suffix = self.seg_suffixes.get(seg_type)
        if suffix is None:
            raise ValueError(f"Unknown segmentation type: {seg_type}")
        
        # Check if seg-specific directory is defined
        if self.seg_dirs and seg_type in self.seg_dirs:
            seg_dir = self.base_dir / self.seg_dirs[seg_type]
            path = seg_dir / f"{self.base_name}{suffix}"
        else:
            path = self.base_dir / f"{self.base_name}{suffix}"
            
        if not path.exists():
            # Try alternative locations
            search_dirs = []
            if self.seg_dirs and seg_type in self.seg_dirs:
                search_dirs.append(self.base_dir / self.seg_dirs[seg_type])
            else:
                search_dirs.append(self.base_dir)
                
            # Try without base name for some seg types
            searched_paths = [str(path)]  # Original path
            
            if seg_type == 'qupath_seg':
                for search_dir in search_dirs:
                    alt_path = search_dir / 'qupath_mask.npy'
                    searched_paths.append(str(alt_path))
                    if alt_path.exists():
                        path = alt_path
                        break
                else:
                    error_msg = (f"Could not find {seg_type} with base name '{self.base_name}'\n"
                               f"Searched in the following locations:\n" + 
                               "\n".join(f"  - {p}" for p in searched_paths))
                    raise FileNotFoundError(error_msg)
            else:
                error_msg = (f"Could not find {seg_type} with base name '{self.base_name}'\n"
                           f"Searched in the following locations:\n" + 
                           "\n".join(f"  - {p}" for p in searched_paths))
                raise FileNotFoundError(error_msg)
        
        # Track segmentation source
        self.metadata.segmentation_sources[seg_type] = str(path)
        return path
    
    def save_processed_mask(self, mask: np.ndarray, name: str, processing_info: Optional[ProcessingStep] = None) -> Path:
        """Save a processed mask and track it."""
        output_path = self.output_dir / f"{self.base_name}_{name}.npy"
        np.save(output_path, mask)
        
        self._processed_masks[name] = output_path
        
        # Add processing step to metadata
        if processing_info:
            processing_info.output_data = str(output_path)
            self.metadata.add_step(processing_info)
        
        return output_path
    
    def save_metadata(self, filename: Optional[str] = None):
        """Save metadata to JSON file."""
        if filename is None:
            filename = f"{self.base_name}_metadata.json"
        
        metadata_path = self.output_dir / filename
        self.metadata.to_json(metadata_path)
        return metadata_path
    
    @property
    def cy5_img(self) -> Path:
        return self.get_image_path('cy5')
    
    @property
    def cy3_img(self) -> Path:
        return self.get_image_path('cy3')
    
    @property
    def dapi_img(self) -> Path:
        return self.get_image_path('dapi')
    
    @property
    def cd11b_img(self) -> Path:
        return self.get_image_path('cd11b')
    
    @property
    def cy5_seg(self) -> Path:
        return self.get_seg_path('cy5_seg')
    
    @property
    def cy3_seg(self) -> Path:
        return self.get_seg_path('cy3_seg')
    
    @property
    def dapi_seg(self) -> Path:
        return self.get_seg_path('dapi_seg')
    
    @property
    def qupath_seg(self) -> Path:
        return self.get_seg_path('qupath_seg')
    
    def load_segs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns tuple of segmentation masks: (CY5, DAPI, QuPath).
        """
        seg_cy5 = load_segmentation(self.cy5_seg)
        seg_dapi = load_segmentation(self.dapi_seg)
        seg_qupath = load_segmentation(self.qupath_seg)
        return seg_cy5, seg_dapi, seg_qupath

    def load_imgs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns tuple of raw image arrays: (CY5, DAPI, CD11b).
        """
        img_cy5 = load_image(self.cy5_img)
        img_dapi = load_image(self.dapi_img)
        img_cd11b = load_image(self.cd11b_img)
        return img_cy5, img_dapi, img_cd11b
    
    def get_all_paths_dict(self) -> Dict[str, Path]:
        """Get dictionary of all available paths."""
        paths = {
            'cy5_img': self.cy5_img,
            'cy3_img': self.cy3_img,
            'dapi_img': self.dapi_img,
            'cd11b_img': self.cd11b_img,
            'cy5_seg': self.cy5_seg,
            'dapi_seg': self.dapi_seg,
            'qupath_seg': self.qupath_seg,
            'output_dir': self.output_dir
        }
        paths.update({f'processed_{k}': v for k, v in self._processed_masks.items()})
        return paths


def filter_by_overlap(
    seg_mask: np.ndarray,
    ref_mask: np.ndarray,
    min_overlap_ratio: float,
    data_paths: Optional[DataPaths] = None,
    step_name: str = "overlap_filter"
) -> np.ndarray:
    """
    Keep only objects with overlap(ref_mask)/area >= min_overlap_ratio.
    """
    filtered = seg_mask.copy()
    objects = find_objects(seg_mask)
    labels = np.unique(seg_mask)[1:]
    removed_count = 0
    
    for lab in labels:
        slc = objects[lab-1]
        if slc is None:
            continue
        region = (seg_mask[slc]==lab)
        overlap = (region & (ref_mask[slc]>0)).sum()
        area = region.sum()
        if area > 0 and overlap/area < min_overlap_ratio:
            filtered[slc][region] = 0
            removed_count += 1
    
    # Track processing step if data_paths provided
    if data_paths:
        step = ProcessingStep(
            step_name=step_name,
            timestamp=datetime.now().isoformat(),
            parameters={
                "min_overlap_ratio": min_overlap_ratio,
                "removed_objects": removed_count,
                "remaining_objects": len(labels) - removed_count
            },
            input_data=["segmentation_mask", "reference_mask"],
            notes=f"Filtered objects with overlap ratio < {min_overlap_ratio}"
        )
        data_paths.metadata.add_step(step)
    
    return filtered


def size_and_dapi_filter(
    seg_mask: np.ndarray,
    dapi_mask: np.ndarray,
    dapi_img: np.ndarray,
    min_size: int,
    min_overlap_ratio: float,
    min_dapi_intensity: float = None
) -> np.ndarray:
    """
    1) Remove objects smaller than `min_size` always.
    2) Remove objects with DAPI-overlap < `min_overlap_ratio`, unless the object's
       mean DAPI intensity >= `min_dapi_intensity` (if provided).
    """
    filtered = seg_mask.copy()
    objects = find_objects(seg_mask)
    labels = np.unique(seg_mask)[1:]

    for lab in labels:
        slc = objects[lab - 1]
        if slc is None:
            continue
        region = (seg_mask[slc] == lab)
        size = region.sum()

        # always enforce min_size
        if size < min_size:
            filtered[slc][region] = 0
            continue

        # compute overlap ratio
        overlap = (region & (dapi_mask[slc] > 0)).sum()
        overlap_ratio = overlap / size if size > 0 else 0.0

        # compute mean DAPI intensity if threshold is set
        mean_intensity = None
        if min_dapi_intensity is not None:
            pix = dapi_img[slc][region]
            mean_intensity = pix.mean() if pix.size > 0 else 0.0

        # remove by overlap ratio unless intensity safeguard applies
        if overlap_ratio < min_overlap_ratio:
            if min_dapi_intensity is None or mean_intensity < min_dapi_intensity:
                filtered[slc][region] = 0

    return filtered

def compute_average_intensity(
    seg_mask: np.ndarray,
    intensity_img: np.ndarray,
    use_donut: bool=False,
    erode_radius: int=1
) -> dict:
    """
    Returns dict {label: mean intensity} per object; ring if use_donut.
    """
    avgs = {}
    objs = find_objects(seg_mask)
    labs = np.unique(seg_mask)[1:]
    for lab in labs:
        slc = objs[lab-1]
        if slc is None:
            continue
        region = (seg_mask[slc]==lab)
        if use_donut:
            region_ring = generate_2D_donut(region, erode_radius)
            pix = intensity_img[slc][region_ring]
        else:
            pix = intensity_img[slc][region]
        if pix.size>0:
            avgs[lab] = pix.mean()
    return avgs

def threshold_otsu_values(values: np.ndarray) -> float:
    return threshold_otsu(values)

def intensity_filter(
    seg_mask: np.ndarray,
    avg_int: dict,
    upper_thresh: float,
    lower_thresh: float=0,
    data_paths: Optional[DataPaths] = None,
    intensity_channel: Optional[str] = None,
    step_name: str = "intensity_filter"
) -> np.ndarray:
    filt = seg_mask.copy()
    objs = find_objects(seg_mask)
    removed_count = 0
    
    for lab, mean in avg_int.items():
        slc = objs[lab-1]
        if slc is None:
            continue
        region = (seg_mask[slc]==lab)
        if mean>upper_thresh or mean<lower_thresh:
            filt[slc][region]=0
            removed_count += 1
    
    # Track processing step and intensity source
    if data_paths:
        if intensity_channel:
            data_paths.metadata.intensity_filter_sources[step_name] = intensity_channel
        
        step = ProcessingStep(
            step_name=step_name,
            timestamp=datetime.now().isoformat(),
            parameters={
                "upper_threshold": upper_thresh,
                "lower_threshold": lower_thresh,
                "intensity_channel": intensity_channel,
                "removed_objects": removed_count,
                "remaining_objects": len(avg_int) - removed_count
            },
            input_data=["segmentation_mask", f"intensity_image_{intensity_channel}" if intensity_channel else "intensity_image"],
            notes=f"Filtered objects with intensity outside [{lower_thresh}, {upper_thresh}]"
        )
        data_paths.metadata.add_step(step)
    
    return filt

def reassign_labels(mask: np.ndarray) -> np.ndarray:
    return sk_label(mask>0)

def relabel_sequential_labels(mask: np.ndarray) -> np.ndarray:
    nm, _, _ = relabel_sequential(mask)
    return nm

def count_cells(mask: np.ndarray) -> int:
    return len(np.unique(mask)[1:])


#### Visualize
def display_example(
    image: np.ndarray,
    seg_mask: np.ndarray,
    slice_idx: int=0
):
    fig, axes = plt.subplots(1,2,figsize=(10,5))
    img = image[slice_idx] if image.ndim==3 else image
    seg = seg_mask[slice_idx] if seg_mask.ndim==3 else seg_mask
    axes[0].imshow(img, cmap='gray'); axes[0].set_title(f'Image slice {slice_idx}')
    axes[1].imshow(seg, cmap='nipy_spectral'); axes[1].set_title(f'Seg slice {slice_idx}')
    plt.tight_layout()
    plt.show()

def plot_dirc_distribution(intensity_dict):
    """
    Plot the distribution of intensity values from a dictionary
    
    Args:
        intensity_dict: Dictionary with intensity values
    """
    # Extract intensity values from the dictionary
    intensity_values = list(intensity_dict.values())
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(intensity_values, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel('Mean Intensity')
    plt.ylabel('Frequency')
    plt.title('Distribution of Intensity Values')
    plt.grid(alpha=0.3)
    
    # Add vertical line for mean and median
    mean_intensity = np.mean(intensity_values)
    median_intensity = np.median(intensity_values)
    plt.axvline(mean_intensity, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_intensity:.2f}')
    plt.axvline(median_intensity, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median_intensity:.2f}')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return mean_intensity, median_intensity


def create_data_paths(
    base_dir: Union[str, Path],
    base_name: str,
    experiment_name: str,
    sample_id: str,
    acquisition_date: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
    channel_suffixes: Optional[Dict[str, str]] = None,
    seg_suffixes: Optional[Dict[str, str]] = None,
    channel_dirs: Optional[Dict[str, str]] = None,
    seg_dirs: Optional[Dict[str, str]] = None,
    notes: Optional[str] = None
) -> DataPaths:
    """
    Convenience function to create a DataPaths instance with metadata.
    
    Parameters
    ----------
    base_dir : str or Path
        Base directory containing the image and segmentation files
    base_name : str
        Base name for files (e.g., "Sample01" or "Slide 8 of 1_Region 001")
    experiment_name : str
        Name of the experiment
    sample_id : str
        Sample identifier
    acquisition_date : str, optional
        Date when images were acquired
    output_dir : str or Path, optional
        Output directory for processed files (default: base_dir/processed)
    channel_suffixes : dict, optional
        Custom channel suffixes mapping
    seg_suffixes : dict, optional
        Custom segmentation suffixes mapping
    channel_dirs : dict, optional
        Channel-specific subdirectories (e.g., {'cy5': 'raw_images/CY5'})
    seg_dirs : dict, optional
        Segmentation-specific subdirectories (e.g., {'cy5_seg': 'segmentations/CY5'})
    notes : str, optional
        Additional notes about the experiment
    
    Returns
    -------
    DataPaths
        Configured DataPaths instance
    """
    # Create metadata
    metadata = ImageMetadata(
        experiment_name=experiment_name,
        sample_id=sample_id,
        acquisition_date=acquisition_date,
        notes=notes
    )
    
    # Create DataPaths instance
    data_paths = DataPaths(
        base_dir=base_dir,
        base_name=base_name,
        metadata=metadata,
        output_dir=output_dir,
        channel_dirs=channel_dirs,
        seg_dirs=seg_dirs
    )
    
    # Update custom suffixes if provided
    if channel_suffixes:
        data_paths.channel_suffixes.update(channel_suffixes)
    if seg_suffixes:
        data_paths.seg_suffixes.update(seg_suffixes)
    
    return data_paths


# Example usage function
def example_usage():
    """
    Example of how to use the enhanced DataPaths class with metadata tracking.
    """
    # Create data paths with metadata
    data_paths = create_data_paths(
        base_dir="/path/to/data",
        base_name="Sample01",
        experiment_name="Immune Cell Analysis",
        sample_id="Mouse_01_Brain_Section_03",
        acquisition_date="2024-01-15",
        notes="60x oil immersion, Z-stack 50 slices"
    )
    
    # Load images and segmentations
    img_cy5, img_dapi, img_cd11b = data_paths.load_imgs()
    seg_cy5, seg_dapi, seg_qupath = data_paths.load_segs()
    
    # Process with overlap filter
    filtered_cy5 = filter_by_overlap(
        seg_cy5, 
        seg_dapi, 
        min_overlap_ratio=0.5,
        data_paths=data_paths,
        step_name="cy5_dapi_overlap_filter"
    )
    
    # Save processed mask
    data_paths.save_processed_mask(
        filtered_cy5, 
        "cy5_dapi_filtered",
        ProcessingStep(
            step_name="save_cy5_filtered",
            timestamp=datetime.now().isoformat(),
            parameters={"format": "numpy"},
            input_data=["cy5_dapi_overlap_filter"],
            notes="Saved CY5 cells with >50% DAPI overlap"
        )
    )
    
    # Compute intensities and filter
    avg_cd11b = compute_average_intensity(filtered_cy5, img_cd11b, use_donut=True)
    
    # Apply intensity filter
    final_mask = intensity_filter(
        filtered_cy5,
        avg_cd11b,
        upper_thresh=1000,
        lower_thresh=100,
        data_paths=data_paths,
        intensity_channel="cd11b",
        step_name="cd11b_intensity_filter"
    )
    
    # Save final result and metadata
    data_paths.save_processed_mask(final_mask, "final_immune_cells")
    metadata_path = data_paths.save_metadata()
    
    print(f"Processing complete. Metadata saved to: {metadata_path}")
    print(f"All paths: {data_paths.get_all_paths_dict()}")
    
    return data_paths