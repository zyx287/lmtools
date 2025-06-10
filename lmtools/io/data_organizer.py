'''
Data Organization Module for LMTools
====================================

This module provides functionality to organize microscopy data in a two-step workflow:
1. Organize raw images by channel for batch cellpose segmentation
2. Reorganize data by sample ID after segmentation

Author: LMTools
'''

import os
import shutil
import re
from pathlib import Path
import pandas as pd
from datetime import datetime
import json
from typing import Dict, List, Optional, Tuple, Union
import logging

# Setup module logger
logger = logging.getLogger(__name__)


class DataOrganizer:
    '''Handles the organization of microscopy data for processing.
    
    This class provides a two-step workflow:
    1. organize_by_channel(): Organize images by channel for cellpose batch processing
    2. organize_by_sample(): Reorganize data by sample ID after segmentation
    
    Attributes:
        source_dir (Path): Source directory containing raw images
        output_dir (Path): Output directory for organized data
        sample_info (List[Dict]): Information about processed samples
    '''
    
    def __init__(self, source_dir: Union[str, Path], output_dir: Union[str, Path]):
        '''Initialize the DataOrganizer.
        
        Args:
            source_dir: Directory containing raw images
            output_dir: Directory for organized output
        '''
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.sample_info = []
        
        # Validate source directory
        if not self.source_dir.exists():
            raise ValueError(f"Source directory does not exist: {source_dir}")
        
    def extract_sample_info(self, filename: str) -> Tuple[Optional[str], Optional[str]]:
        '''Extract sample ID and channel from filename.
        
        Expected format: id_CHANNEL.tiff
        Example: Sample01_CY5.tiff -> (Sample01, CY5)
        
        Args:
            filename: Image filename
            
        Returns:
            Tuple of (sample_id, channel) or (None, None) if pattern doesn't match
        '''
        pattern = r'^(.+?)_(CY5|CY3|DAPI)\.tiff?$'
        match = re.match(pattern, filename, re.IGNORECASE)
        
        if match:
            sample_id = match.group(1)
            channel = match.group(2).upper()
            return sample_id, channel
        return None, None
    
    def organize_by_channel(self) -> pd.DataFrame:
        '''Step 1: Organize images by channel for cellpose batch processing.
        
        Creates the following structure:
        output_dir/
        ├── channels_for_segmentation/
        │   ├── CY5/
        │   ├── CY3/
        │   └── DAPI/
        ├── sample_list.csv
        └── run_cellpose_batch.sh
        
        Returns:
            DataFrame with information about organized files
        '''
        logger.info("Step 1: Organizing images by channel...")
        
        # Create channel directories
        channel_dir = self.output_dir / "channels_for_segmentation"
        channel_dirs = {}
        for channel in ['CY5', 'CY3', 'DAPI']:
            channel_dirs[channel] = channel_dir / channel
            channel_dirs[channel].mkdir(parents=True, exist_ok=True)
        
        # Process all image files
        processed_files = []
        for file_path in self.source_dir.rglob('*.tif*'):
            if file_path.is_file():
                sample_id, channel = self.extract_sample_info(file_path.name)
                
                if sample_id and channel:
                    # Copy file to channel directory
                    dest_path = channel_dirs[channel] / file_path.name
                    shutil.copy2(file_path, dest_path)
                    
                    # Record file info
                    file_info = {
                        'sample_id': sample_id,
                        'channel': channel,
                        'original_path': str(file_path),
                        'channel_path': str(dest_path),
                        'file_size': file_path.stat().st_size,
                        'timestamp': datetime.now().isoformat()
                    }
                    processed_files.append(file_info)
                    self.sample_info.append(file_info)
                    
                    logger.debug(f"Copied {file_path.name} -> {channel}/{file_path.name}")
        
        # Create sample summary DataFrame
        df = pd.DataFrame(processed_files)
        
        # Save summary
        summary_path = channel_dir / "organization_summary.csv"
        df.to_csv(summary_path, index=False)
        
        # Save sample list in output root
        sample_list_path = self.output_dir / "sample_list.csv"
        df[['sample_id', 'channel', 'original_path']].to_csv(sample_list_path, index=False)
        
        # Create summary report
        self._create_step1_report(df, channel_dir)
        
        # Create cellpose batch script
        self._create_cellpose_script(channel_dir)
        
        # Log summary
        logger.info(f"Step 1 Complete:")
        logger.info(f"  Total files processed: {len(processed_files)}")
        if not df.empty:
            logger.info(f"  Unique samples: {df['sample_id'].nunique()}")
            logger.info(f"  Files per channel:")
            for channel, count in df['channel'].value_counts().items():
                logger.info(f"    {channel}: {count}")
        
        return df
    
    def find_segmentation_masks(self, channel_dir: Path) -> Dict[str, Path]:
        '''Find cellpose segmentation masks in a channel directory.
        
        Args:
            channel_dir: Directory containing cellpose output
            
        Returns:
            Dictionary mapping sample_channel keys to mask file paths
        '''
        masks = {}
        
        # Look for cellpose output patterns
        mask_patterns = [
            '*_cp_masks.npy',
            '*_masks.npy',
            '*_cellpose_masks.npy'
        ]
        
        for pattern in mask_patterns:
            for mask_file in channel_dir.glob(pattern):
                # Extract sample ID from mask filename
                base_name = mask_file.stem
                # Remove common suffixes
                for suffix in ['_cp_masks', '_masks', '_cellpose_masks']:
                    if base_name.endswith(suffix):
                        base_name = base_name[:-len(suffix)]
                        break
                
                # Try to match with original image name
                sample_id, channel = self.extract_sample_info(base_name + '.tiff')
                if sample_id:
                    masks[f"{sample_id}_{channel}"] = mask_file
                    
        return masks
    
    def organize_by_sample(self, include_masks: bool = True) -> pd.DataFrame:
        '''Step 2: Organize data by sample ID after segmentation.
        
        Creates the following structure:
        output_dir/
        └── samples/
            ├── Sample01/
            │   ├── raw_images/
            │   ├── segmentations/
            │   ├── results/
            │   └── sample_metadata.json
            ├── Sample02/
            │   └── ...
            ├── master_sample_list.csv
            ├── master_sample_list.xlsx
            └── sample_ids.txt
        
        Args:
            include_masks: Whether to copy segmentation masks
            
        Returns:
            DataFrame with sample organization information
        '''
        logger.info("Step 2: Organizing data by sample ID...")
        
        # Create samples directory
        samples_dir = self.output_dir / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        
        # Load previous organization info if available
        channel_dir = self.output_dir / "channels_for_segmentation"
        summary_path = channel_dir / "organization_summary.csv"
        
        if summary_path.exists():
            df = pd.read_csv(summary_path)
        else:
            # If no previous summary, scan channel directories
            df = pd.DataFrame(self.sample_info) if self.sample_info else pd.DataFrame()
        
        if df.empty:
            logger.warning("No sample information found. Please run organize_by_channel() first.")
            return df
        
        # Find all segmentation masks
        all_masks = {}
        if include_masks:
            for channel in ['CY5', 'CY3', 'DAPI']:
                channel_path = channel_dir / channel
                if channel_path.exists():
                    masks = self.find_segmentation_masks(channel_path)
                    all_masks.update(masks)
        
        # Group by sample ID
        sample_groups = df.groupby('sample_id')
        reorganized_info = []
        
        for sample_id, group in sample_groups:
            # Create sample directory
            sample_dir = samples_dir / sample_id
            sample_dir.mkdir(exist_ok=True)
            
            # Create subdirectories
            raw_dir = sample_dir / "raw_images"
            raw_dir.mkdir(exist_ok=True)
            
            if include_masks:
                seg_dir = sample_dir / "segmentations"
                seg_dir.mkdir(exist_ok=True)
            
            results_dir = sample_dir / "results"
            results_dir.mkdir(exist_ok=True)
            
            sample_data = {
                'sample_id': sample_id,
                'channels': [],
                'raw_images': {},
                'segmentations': {},
                'created_at': datetime.now().isoformat()
            }
            
            # Copy files for this sample
            for _, file_info in group.iterrows():
                channel = file_info['channel']
                channel_path = Path(file_info['channel_path'])
                
                if channel_path.exists():
                    # Copy raw image
                    dest_raw = raw_dir / f"{sample_id}_{channel}.tiff"
                    shutil.copy2(channel_path, dest_raw)
                    sample_data['channels'].append(channel)
                    sample_data['raw_images'][channel] = str(dest_raw)
                    
                    # Copy segmentation mask if available
                    if include_masks:
                        mask_key = f"{sample_id}_{channel}"
                        if mask_key in all_masks:
                            mask_path = all_masks[mask_key]
                            dest_mask = seg_dir / f"{sample_id}_{channel}_masks.npy"
                            shutil.copy2(mask_path, dest_mask)
                            sample_data['segmentations'][channel] = str(dest_mask)
            
            # Save sample metadata
            metadata_path = sample_dir / "sample_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(sample_data, f, indent=2)
            
            reorganized_info.append({
                'sample_id': sample_id,
                'sample_dir': str(sample_dir),
                'channels': ','.join(sample_data['channels']),
                'has_cy5': 'CY5' in sample_data['channels'],
                'has_cy3': 'CY3' in sample_data['channels'],
                'has_dapi': 'DAPI' in sample_data['channels'],
                'num_channels': len(sample_data['channels']),
                'has_segmentations': len(sample_data['segmentations']) > 0
            })
            
            logger.debug(f"Created sample directory: {sample_id}")
            logger.debug(f"  Channels: {', '.join(sample_data['channels'])}")
            if sample_data['segmentations']:
                logger.debug(f"  Segmentations: {', '.join(sample_data['segmentations'].keys())}")
        
        # Create reorganization summary
        reorg_df = pd.DataFrame(reorganized_info)
        summary_path = samples_dir / "sample_organization_summary.csv"
        reorg_df.to_csv(summary_path, index=False)
        
        # Create master sample list
        self._create_master_sample_list(samples_dir, reorg_df)
        
        # Log summary
        logger.info(f"Step 2 Complete:")
        logger.info(f"  Total samples organized: {len(reorg_df)}")
        logger.info(f"  Samples with all channels: {len(reorg_df[(reorg_df['has_cy5']) & (reorg_df['has_cy3']) & (reorg_df['has_dapi'])])}")
        logger.info(f"  Samples missing CY3: {len(reorg_df[~reorg_df['has_cy3']])}")
        
        return reorg_df
    
    def _create_step1_report(self, df: pd.DataFrame, channel_dir: Path):
        '''Create detailed report for step 1.'''
        report_path = self.output_dir / "step1_organization_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("Step 1: Channel Organization Report\n")
            f.write("===================================\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Source Directory: {self.source_dir}\n")
            f.write(f"Output Directory: {self.output_dir}\n")
            f.write("\n")
            f.write("Files Organized by Channel:\n")
            
            for channel in ['CY5', 'CY3', 'DAPI']:
                count = len(list((channel_dir / channel).glob("*.tif*")))
                f.write(f"  {channel}: {count} images\n")
            
            f.write(f"\nTotal images processed: {len(df)}\n")
            f.write("\n")
            f.write("Next Steps:\n")
            f.write("1. Run cellpose segmentation on each channel folder:\n")
            f.write(f"   - {channel_dir}/CY5/\n")
            f.write(f"   - {channel_dir}/CY3/\n")
            f.write(f"   - {channel_dir}/DAPI/\n")
            f.write("\n")
            f.write("2. After segmentation, run Step 2 to organize by sample ID:\n")
            f.write(f"   from lmtools.io import DataOrganizer\n")
            f.write(f"   organizer = DataOrganizer('{self.source_dir}', '{self.output_dir}')\n")
            f.write(f"   organizer.organize_by_sample()\n")
    
    def _create_cellpose_script(self, channel_dir: Path):
        '''Create cellpose batch processing script.'''
        script_path = self.output_dir / "run_cellpose_batch.sh"
        
        script_content = f'''#!/bin/bash
# Batch cellpose segmentation script
# Generated by lmtools DataOrganizer

CHANNEL_DIR="{channel_dir}"

echo "Running cellpose segmentation on all channels..."

# Process each channel
for channel in CY5 CY3 DAPI; do
    if [ -d "$CHANNEL_DIR/$channel" ] && [ "$(ls -A $CHANNEL_DIR/$channel/*.tif* 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "Processing $channel channel..."
        
        # Adjust parameters based on channel
        if [ "$channel" = "DAPI" ]; then
            MODEL="nuclei"
            DIAMETER=20
        else
            MODEL="cyto"
            DIAMETER=30
        fi
        
        # Example cellpose command - adjust parameters as needed
        cellpose --dir "$CHANNEL_DIR/$channel" \\
                 --pretrained_model $MODEL \\
                 --diameter $DIAMETER \\
                 --save_npy \\
                 --no_npy  # Remove this to save numpy arrays
        
        echo "  Completed $channel"
    else
        echo "Skipping $channel - no images found"
    fi
done

echo ""
echo "After segmentation completes, run Step 2:"
echo "python -c 'from lmtools.io import DataOrganizer; organizer = DataOrganizer(\"{self.source_dir}\", \"{self.output_dir}\"); organizer.organize_by_sample()'"
'''
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        os.chmod(script_path, 0o755)
    
    def _create_master_sample_list(self, samples_dir: Path, reorg_df: pd.DataFrame):
        '''Create master sample list with all relevant information.'''
        logger.info("Creating master sample list...")
        
        # Enhance the DataFrame with additional information
        master_list = reorg_df.copy()
        master_list['processing_status'] = 'ready'
        master_list['notes'] = ''
        master_list['last_updated'] = datetime.now().isoformat()
        
        # Save as both CSV and Excel for flexibility
        master_list.to_csv(samples_dir / "master_sample_list.csv", index=False)
        
        try:
            master_list.to_excel(samples_dir / "master_sample_list.xlsx", index=False)
        except ImportError:
            logger.warning("openpyxl not installed, skipping Excel output")
        
        # Create a simple text list of sample IDs
        with open(samples_dir / "sample_ids.txt", 'w') as f:
            for sample_id in master_list['sample_id']:
                f.write(f"{sample_id}\n")
        
        logger.info(f"  Created master_sample_list.csv")
        logger.info(f"  Created sample_ids.txt with {len(master_list)} samples")


def organize_data(source_dir: Union[str, Path], 
                  output_dir: Union[str, Path],
                  step: Optional[int] = None,
                  include_masks: bool = True) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    '''Organize microscopy data for processing.
    
    This function provides a convenient interface to the DataOrganizer class.
    
    Args:
        source_dir: Directory containing raw images
        output_dir: Directory for organized output
        step: Which step to run (1, 2, or None for both)
        include_masks: Whether to copy segmentation masks in step 2
        
    Returns:
        If step=1: DataFrame from organize_by_channel
        If step=2: DataFrame from organize_by_sample
        If step=None: Tuple of (channel_df, sample_df)
        
    Example:
        >>> from lmtools.io import organize_data
        >>> # Run both steps
        >>> channel_df, sample_df = organize_data('/raw/images', '/organized')
        >>> # Run only step 1
        >>> channel_df = organize_data('/raw/images', '/organized', step=1)
    '''
    organizer = DataOrganizer(source_dir, output_dir)
    
    if step == 1:
        return organizer.organize_by_channel()
    elif step == 2:
        return organizer.organize_by_sample(include_masks=include_masks)
    else:
        # Run both steps
        channel_df = organizer.organize_by_channel()
        sample_df = organizer.organize_by_sample(include_masks=include_masks)
        return channel_df, sample_df