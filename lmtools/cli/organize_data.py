#!/usr/bin/env python3
'''
Organize microscopy data for processing.

This module provides command-line interface for the data organization workflow.
'''

import argparse
import logging
from pathlib import Path
from lmtools.io import organize_data


def add_arguments(parser):
    '''Add arguments to the parser for the organize-data command.'''
    parser.add_argument(
        '-s', '--source', 
        required=True, 
        help='Source directory with raw images'
    )
    parser.add_argument(
        '-o', '--output', 
        required=True, 
        help='Output directory for organized data'
    )
    parser.add_argument(
        '--step', 
        type=int, 
        choices=[1, 2], 
        help='Run specific step (1 or 2)'
    )
    parser.add_argument(
        '--all', 
        action='store_true', 
        help='Run both steps sequentially'
    )
    parser.add_argument(
        '--no-masks', 
        action='store_true', 
        help='Skip copying segmentation masks in step 2'
    )
    parser.add_argument(
        '-v', '--verbose', 
        action='store_true', 
        help='Enable verbose output'
    )


def main(args):
    '''Main CLI function for data organization.'''
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Validate arguments
    if not args.all and args.step is None:
        raise ValueError("Either --step or --all must be specified")
    
    try:
        # Run organization
        if args.all:
            print("Running complete data organization workflow...")
            channel_df, sample_df = organize_data(
                args.source, 
                args.output,
                include_masks=not args.no_masks
            )
            print(f"\nWorkflow complete!")
            print(f"  Organized {len(channel_df)} files by channel")
            print(f"  Created {len(sample_df)} sample directories")
        elif args.step == 1:
            print("Step 1: Organizing images by channel...")
            df = organize_data(args.source, args.output, step=1)
            print(f"\nStep 1 complete! Organized {len(df)} files")
            print("\nNext: Run cellpose on channel directories, then use --step 2")
        else:  # step == 2
            print("Step 2: Organizing by sample ID...")
            df = organize_data(
                args.source, 
                args.output, 
                step=2,
                include_masks=not args.no_masks
            )
            print(f"\nStep 2 complete! Created {len(df)} sample directories")
        
        print(f"\nOutput directory: {args.output}")
        
    except Exception as e:
        logging.error(f"Error during organization: {e}")
        raise