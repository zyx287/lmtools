'''
Cellpose Output Helper Functions
================================

Helper functions to standardize cellpose output naming conventions
for compatibility with lmtools data organization workflow.

Author: LMTools
'''

import os
from pathlib import Path
from typing import Union, List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def standardize_cellpose_masks(
    directory: Union[str, Path],
    dry_run: bool = False,
    verbose: bool = True
) -> Dict[str, List[Path]]:
    '''
    Standardize cellpose mask naming by removing .tiff/.tif from mask filenames.
    
    Cellpose sometimes outputs masks as 'image.tiff_masks.npy' when the input
    has .tiff extension. This function renames them to 'image_masks.npy' for
    compatibility with lmtools organization workflow.
    
    Parameters
    ----------
    directory : str or Path
        Directory containing cellpose output files
    dry_run : bool, default=False
        If True, only show what would be renamed without actually renaming
    verbose : bool, default=True
        Print renaming operations
    
    Returns
    -------
    Dict[str, List[Path]]
        Dictionary with 'renamed' and 'skipped' lists of file paths
    
    Example
    -------
    >>> from lmtools.io import standardize_cellpose_masks
    >>> 
    >>> # Fix naming in CY5 directory after cellpose
    >>> results = standardize_cellpose_masks('organized_output/channels_for_segmentation/CY5')
    >>> print(f"Renamed {len(results['renamed'])} files")
    '''
    directory = Path(directory)
    
    if not directory.exists():
        raise ValueError(f"Directory does not exist: {directory}")
    
    results = {
        'renamed': [],
        'skipped': []
    }
    
    # Patterns to fix: .tiff_masks.npy or .tif_masks.npy
    patterns = ['*.tiff_masks.npy', '*.tif_masks.npy']
    
    for pattern in patterns:
        for mask_file in directory.glob(pattern):
            # Determine new name by removing .tiff or .tif
            if '.tiff_masks.npy' in mask_file.name:
                new_name = mask_file.name.replace('.tiff_masks.npy', '_masks.npy')
            elif '.tif_masks.npy' in mask_file.name:
                new_name = mask_file.name.replace('.tif_masks.npy', '_masks.npy')
            else:
                results['skipped'].append(mask_file)
                continue
            
            new_path = mask_file.parent / new_name
            
            # Check if target already exists
            if new_path.exists() and new_path != mask_file:
                if verbose:
                    logger.warning(f"Target already exists, skipping: {new_name}")
                results['skipped'].append(mask_file)
                continue
            
            if dry_run:
                if verbose:
                    print(f"Would rename: {mask_file.name} -> {new_name}")
            else:
                mask_file.rename(new_path)
                if verbose:
                    print(f"Renamed: {mask_file.name} -> {new_name}")
                results['renamed'].append(new_path)
    
    # Also check for correct naming
    correct_masks = list(directory.glob('*_masks.npy'))
    correct_masks = [m for m in correct_masks if '.tiff' not in m.name and '.tif' not in m.name]
    
    if verbose:
        print(f"\nSummary for {directory.name}:")
        print(f"  Files renamed: {len(results['renamed'])}")
        print(f"  Files skipped: {len(results['skipped'])}")
        print(f"  Correctly named masks: {len(correct_masks)}")
    
    return results


def standardize_all_channels(
    channels_dir: Union[str, Path],
    channels: Optional[List[str]] = None,
    dry_run: bool = False,
    verbose: bool = True
) -> Dict[str, Dict]:
    '''
    Standardize cellpose masks across all channel directories.
    
    Parameters
    ----------
    channels_dir : str or Path
        Parent directory containing channel subdirectories
        (e.g., 'organized_output/channels_for_segmentation')
    channels : List[str], optional
        List of channel names to process. If None, processes ['CY5', 'CY3', 'DAPI']
    dry_run : bool, default=False
        If True, only show what would be renamed without actually renaming
    verbose : bool, default=True
        Print renaming operations
    
    Returns
    -------
    Dict[str, Dict]
        Results for each channel
    
    Example
    -------
    >>> from lmtools.io import standardize_all_channels
    >>> 
    >>> # Fix all channels after cellpose batch processing
    >>> results = standardize_all_channels('organized_output/channels_for_segmentation')
    '''
    channels_dir = Path(channels_dir)
    
    if channels is None:
        channels = ['CY5', 'CY3', 'DAPI']
    
    all_results = {}
    
    for channel in channels:
        channel_dir = channels_dir / channel
        if channel_dir.exists():
            if verbose:
                print(f"\nProcessing {channel} channel...")
            all_results[channel] = standardize_cellpose_masks(
                channel_dir, dry_run=dry_run, verbose=verbose
            )
        else:
            if verbose:
                print(f"\nSkipping {channel} - directory not found")
            all_results[channel] = {'renamed': [], 'skipped': []}
    
    return all_results


def check_cellpose_output(
    directory: Union[str, Path],
    expected_pattern: str = '*_masks.npy'
) -> Dict[str, List[Path]]:
    '''
    Check cellpose output files and categorize them.
    
    Parameters
    ----------
    directory : str or Path
        Directory to check
    expected_pattern : str
        Expected pattern for mask files
    
    Returns
    -------
    Dict[str, List[Path]]
        Dictionary with categorized files:
        - 'correct_masks': Properly named mask files
        - 'incorrect_masks': Mask files needing renaming
        - 'other_cellpose': Other cellpose output files
        - 'images': Input image files
    
    Example
    -------
    >>> from lmtools.io import check_cellpose_output
    >>> 
    >>> # Check what cellpose created
    >>> files = check_cellpose_output('organized_output/channels_for_segmentation/CY5')
    >>> print(f"Correct masks: {len(files['correct_masks'])}")
    >>> print(f"Need renaming: {len(files['incorrect_masks'])}")
    '''
    directory = Path(directory)
    
    results = {
        'correct_masks': [],
        'incorrect_masks': [],
        'other_cellpose': [],
        'images': []
    }
    
    if not directory.exists():
        return results
    
    for file in directory.iterdir():
        if file.is_file():
            # Check for masks
            if file.suffix == '.npy' and '_masks' in file.name:
                if '.tiff_masks' in file.name or '.tif_masks' in file.name:
                    results['incorrect_masks'].append(file)
                else:
                    results['correct_masks'].append(file)
            # Other cellpose outputs
            elif any(pattern in file.name for pattern in ['_cp_', '_seg.npy', '_flows.tif']):
                results['other_cellpose'].append(file)
            # Image files
            elif file.suffix in ['.tif', '.tiff', '.png', '.jpg']:
                results['images'].append(file)
    
    return results


def prepare_for_step2(
    output_dir: Union[str, Path],
    verbose: bool = True
) -> bool:
    '''
    Prepare cellpose output for lmtools organize_data step 2.
    
    This function checks and fixes mask naming in all channel directories.
    
    Parameters
    ----------
    output_dir : str or Path
        The output directory used in organize_data step 1
    verbose : bool, default=True
        Print progress information
    
    Returns
    -------
    bool
        True if successful, False if errors occurred
    
    Example
    -------
    >>> from lmtools.io import organize_data, prepare_for_step2
    >>> 
    >>> # After running cellpose on step 1 output
    >>> prepare_for_step2('organized_output')
    >>> 
    >>> # Now run step 2
    >>> sample_df = organize_data('input_data', 'organized_output', step=2)
    '''
    output_dir = Path(output_dir)
    channels_dir = output_dir / 'channels_for_segmentation'
    
    if not channels_dir.exists():
        logger.error(f"Channels directory not found: {channels_dir}")
        return False
    
    try:
        # First check the current state
        if verbose:
            print("Checking cellpose output...")
            print("=" * 50)
        
        all_checks = {}
        for channel in ['CY5', 'CY3', 'DAPI']:
            channel_dir = channels_dir / channel
            if channel_dir.exists():
                all_checks[channel] = check_cellpose_output(channel_dir)
                if verbose:
                    incorrect = len(all_checks[channel]['incorrect_masks'])
                    correct = len(all_checks[channel]['correct_masks'])
                    print(f"{channel}: {correct} correct, {incorrect} need renaming")
        
        # Fix naming issues
        if verbose:
            print("\nStandardizing mask names...")
            print("=" * 50)
        
        results = standardize_all_channels(channels_dir, verbose=verbose)
        
        # Verify everything is ready
        total_renamed = sum(len(r['renamed']) for r in results.values())
        
        if verbose:
            print("\n" + "=" * 50)
            print(f"Preparation complete! Renamed {total_renamed} files.")
            print("You can now run organize_data step 2.")
        
        return True
        
    except Exception as e:
        logger.error(f"Error preparing for step 2: {e}")
        return False