'''
author: zyx
date: 2025-06-10
last_modified: 2025-06-10
description: 
Metadata and Data Path Management for LMTools
===========================================

This module provides classes for tracking processing metadata and managing
file paths in microscopy image processing workflows.

Classes:
    ProcessingStep: Record a single processing step in the pipeline
    ImageMetadata: Metadata for tracking image processing pipeline
    DataPaths: Enhanced data paths manager with automatic path discovery

Functions:
    create_data_paths: Convenience function to create DataPaths with metadata
    create_data_paths_from_organized: Create DataPaths from organize_data output
'''

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field, asdict
import numpy as np
import tifffile


def load_segmentation(path: Path) -> np.ndarray:
    '''Load a .npy segmentation mask.'''
    return np.load(path, allow_pickle=True)


def load_image(path: Path) -> np.ndarray:
    '''Load an image volume or slice (TIFF).'''
    return tifffile.imread(path)


@dataclass
class ProcessingStep:
    '''Record a single processing step in the pipeline.'''
    step_name: str
    timestamp: str
    parameters: Dict[str, Any]
    input_data: List[str]
    output_data: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class ImageMetadata:
    '''Metadata for tracking image processing pipeline.'''
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
        '''Add a processing step to the history.'''
        self.processing_steps.append(step)
    
    def to_json(self, filepath: Path):
        '''Save metadata to JSON file.'''
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def from_json(cls, filepath: Path):
        '''Load metadata from JSON file.'''
        with open(filepath, 'r') as f:
            data = json.load(f)
        # Convert processing steps back to ProcessingStep objects
        data['processing_steps'] = [ProcessingStep(**step) for step in data['processing_steps']]
        return cls(**data)


@dataclass
class DataPaths:
    '''
    Enhanced data paths manager with automatic path discovery and metadata tracking.
    '''
    base_dir: Path
    base_name: str
    metadata: ImageMetadata
    
    # Channel suffixes
    channel_suffixes: Dict[str, str] = field(default_factory=lambda: {
        'cy5': '_CY5.tif',
        'cy3': '_CY3.tif',
        'dapi': '_DAPI.tif'
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
        '''Initialize paths and create output directory if needed.'''
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
        '''Get image path for a specific channel.'''
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
        '''Get segmentation path for a specific type.'''
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
        '''Save a processed mask and track it.'''
        output_path = self.output_dir / f"{self.base_name}_{name}.npy"
        np.save(output_path, mask)
        
        self._processed_masks[name] = output_path
        
        # Add processing step to metadata
        if processing_info:
            processing_info.output_data = str(output_path)
            self.metadata.add_step(processing_info)
        
        return output_path
    
    def load_processed_mask(self, name: str) -> np.ndarray:
        '''Load a previously saved processed mask by name.'''
        if name not in self._processed_masks:
            available = list(self._processed_masks.keys())
            raise ValueError(f"Processed mask '{name}' not found. Available: {available}")
        
        return load_segmentation(self._processed_masks[name])
    
    def get_latest_mask(self, base_name: str = "cy5") -> np.ndarray:
        '''Get the most recently processed mask for a given base name.'''
        # Find all masks that contain the base_name
        matching_masks = [name for name in self._processed_masks.keys() if base_name in name]
        
        if not matching_masks:
            raise ValueError(f"No processed masks found containing '{base_name}'")
        
        # Return the last one (assumes they were saved in order)
        latest_name = matching_masks[-1]
        return self.load_processed_mask(latest_name)
    
    def save_metadata(self, filename: Optional[str] = None):
        '''Save metadata to JSON file.'''
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
    def cy3_img(self) -> Path:
        return self.get_image_path('cy3')
    
    # Backward compatibility alias
    @property
    def cd11b_img(self) -> Path:
        '''Deprecated: Use cy3_img instead. CD11b is a marker, not a channel.'''
        return self.get_image_path('cy3')
    
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
    
    def load_segs(self, seg_types: Optional[List[str]] = None) -> Tuple[np.ndarray, ...]:
        '''
        Returns tuple of segmentation masks in specified order.
        
        Parameters
        ----------
        seg_types : List[str], optional
            List of segmentation types to load in desired order.
            Options: 'cy5', 'cy3', 'dapi', 'qupath'
            Default: ['cy5', 'dapi', 'qupath']
            
        Returns
        -------
        tuple of np.ndarray or None
            Segmentation masks in the order specified, None for missing segmentations
            
        Examples
        --------
        >>> # Default order
        >>> seg_cy5, seg_dapi, seg_qupath = data_paths.load_segs()
        >>> 
        >>> # Only CY5 and DAPI
        >>> seg_cy5, seg_dapi = data_paths.load_segs(['cy5', 'dapi'])
        >>> 
        >>> # Custom order
        >>> seg_dapi, seg_cy5 = data_paths.load_segs(['dapi', 'cy5'])
        '''
        if seg_types is None:
            seg_types = ['cy5', 'dapi', 'qupath']  # Default order for backward compatibility
        
        segs = []
        for seg_type in seg_types:
            try:
                # Add '_seg' suffix if not present
                if not seg_type.endswith('_seg') and seg_type != 'qupath':
                    seg_type = f'{seg_type}_seg'
                elif seg_type == 'qupath':
                    seg_type = 'qupath_seg'
                    
                seg = load_segmentation(self.get_seg_path(seg_type))
                segs.append(seg)
            except FileNotFoundError:
                segs.append(None)
        return tuple(segs)

    def load_imgs(self, channels: Optional[List[str]] = None) -> Tuple[np.ndarray, ...]:
        '''
        Returns tuple of raw image arrays in specified order.
        
        Parameters
        ----------
        channels : List[str], optional
            List of channels to load in desired order. 
            Options: 'cy5', 'cy3', 'dapi'
            Default: ['cy5', 'dapi', 'cy3']
        
        Returns
        -------
        tuple of np.ndarray or None
            Images in the order specified, None for missing channels
            
        Examples
        --------
        >>> # Default order
        >>> img_cy5, img_dapi, img_cy3 = data_paths.load_imgs()
        >>> 
        >>> # Custom order
        >>> img_dapi, img_cy5 = data_paths.load_imgs(['dapi', 'cy5'])
        >>> 
        >>> # All available channels
        >>> imgs = data_paths.load_imgs(['cy5', 'cy3', 'dapi'])
        '''
        if channels is None:
            channels = ['cy5', 'dapi', 'cy3']  # Default order
        
        imgs = []
        for channel in channels:
            try:
                img = load_image(self.get_image_path(channel))
                imgs.append(img)
            except FileNotFoundError:
                imgs.append(None)
        return tuple(imgs)
    
    def get_tissue_mask_path(self) -> Optional[Path]:
        '''Get path to tissue mask GeoJSON file.
        
        The tissue mask is typically created from DAPI channel and applies to all channels.
        Looks for any .geojson file in the tissue_masks folder.
        
        Returns
        -------
        Path or None
            Path to GeoJSON file if exists, None otherwise
        '''
        # Check if this is an organized sample with tissue_masks folder
        tissue_mask_dir = self.base_dir / "tissue_masks"
        if not tissue_mask_dir.exists():
            return None
        
        # Look for any GeoJSON file (usually created from DAPI)
        geojson_files = list(tissue_mask_dir.glob("*.geojson"))
        
        if geojson_files:
            # If multiple files, prefer one with DAPI in name
            dapi_files = [f for f in geojson_files if 'DAPI' in f.name.upper()]
            if dapi_files:
                return dapi_files[0]
            # Otherwise return the first one
            return geojson_files[0]
        
        return None
    
    def has_tissue_mask(self) -> bool:
        '''Check if tissue mask exists.
        
        Returns
        -------
        bool
            True if tissue mask exists
        '''
        return self.get_tissue_mask_path() is not None
    
    def get_all_paths_dict(self) -> Dict[str, Path]:
        '''Get dictionary of all available paths.'''
        paths = {}
        
        # Try to add each path, skip if it doesn't exist
        # Images
        for channel in ['cy5', 'cy3', 'dapi']:
            try:
                path = self.get_image_path(channel)
                paths[f'{channel}_img'] = path
            except FileNotFoundError:
                pass  # Skip missing channels
        
        # Segmentations
        for seg_type in ['cy5_seg', 'cy3_seg', 'dapi_seg', 'qupath_seg']:
            try:
                path = self.get_seg_path(seg_type)
                paths[seg_type] = path
            except FileNotFoundError:
                pass  # Skip missing segmentations
        
        # Tissue mask (single mask for all channels)
        tissue_mask_path = self.get_tissue_mask_path()
        if tissue_mask_path:
            paths['tissue_mask'] = tissue_mask_path
        
        # Always include output directory
        paths['output_dir'] = self.output_dir
        
        # Add processed masks
        paths.update({f'processed_{k}': v for k, v in self._processed_masks.items()})
        
        return paths


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
    '''
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
    '''
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


def create_data_paths_from_organized(
    organized_sample_dir: Union[str, Path],
    experiment_name: str,
    acquisition_date: Optional[str] = None,
    notes: Optional[str] = None
) -> DataPaths:
    '''
    Create a DataPaths instance from organize_data output structure.
    
    This helper function automatically configures DataPaths to work with
    the directory structure created by lmtools.io.organize_data().
    
    Parameters
    ----------
    organized_sample_dir : str or Path
        Path to a sample directory created by organize_data
        (e.g., "/organized/samples/Sample01")
    experiment_name : str
        Name of the experiment
    acquisition_date : str, optional
        Date when images were acquired
    notes : str, optional
        Additional notes about the experiment
    
    Returns
    -------
    DataPaths
        Configured DataPaths instance that works with organize_data structure
    
    Example
    -------
    >>> # After running organize_data
    >>> from lmtools.io import create_data_paths_from_organized
    >>> 
    >>> # Point to the organized sample directory
    >>> data_paths = create_data_paths_from_organized(
    ...     organized_sample_dir="/data/organized/samples/Sample01",
    ...     experiment_name="Immune Cell Analysis"
    ... )
    >>> 
    >>> # Now you can use it normally
    >>> img_cy5, img_dapi, img_cd11b = data_paths.load_imgs()
    >>> seg_cy5, seg_dapi, seg_qupath = data_paths.load_segs()
    '''
    organized_dir = Path(organized_sample_dir)
    
    # Extract sample_id from directory name
    sample_id = organized_dir.name
    
    # Check if sample_metadata.json exists and load it
    metadata_file = organized_dir / "sample_metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            sample_meta = json.load(f)
            # Use the sample_id from metadata if available
            sample_id = sample_meta.get('sample_id', sample_id)
    
    # Configure for organize_data directory structure
    return create_data_paths(
        base_dir=organized_dir,
        base_name=sample_id,
        experiment_name=experiment_name,
        sample_id=sample_id,
        acquisition_date=acquisition_date,
        output_dir=organized_dir / "results",  # Use existing results directory
        channel_suffixes={
            'cy5': '_CY5.tiff',   # organize_data uses .tiff
            'cy3': '_CY3.tiff',
            'dapi': '_DAPI.tiff'
        },
        seg_suffixes={
            'cy5_seg': '_CY5_masks.npy',      # organize_data naming
            'cy3_seg': '_CY3_masks.npy',
            'dapi_seg': '_DAPI_masks.npy',
            'qupath_seg': '_qupath_mask.npy'
        },
        channel_dirs={
            'cy5': 'raw_images',
            'cy3': 'raw_images',
            'dapi': 'raw_images'
        },
        seg_dirs={
            'cy5_seg': 'segmentations',
            'cy3_seg': 'segmentations',
            'dapi_seg': 'segmentations',
            'qupath_seg': 'segmentations'
        },
        notes=notes
    )