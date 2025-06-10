'''
CLI command to split multi-channel images into separate channel files.
'''
import logging
from pathlib import Path
from lmtools.io.channel_splitting import split_channels, batch_split_channels

logger = logging.getLogger(__name__)

def add_arguments(parser):
    '''
    Add command line arguments for the split_channels command.
    '''
    parser.add_argument(
        "input", 
        type=str, 
        help="Input image or directory"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        help="Output directory (defaults to input directory)"
    )
    parser.add_argument(
        "--sequence", 
        type=str, 
        nargs="+", 
        help="Sequence of channel names (e.g. 'R G CY5')"
    )
    parser.add_argument(
        "--separator", 
        type=str, 
        default="_", 
        help="Separator between filename and channel name (default: _)"
    )
    parser.add_argument(
        "--recursive", 
        action="store_true",
        help="Process subdirectories recursively"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose logging"
    )

def main(args):
    '''
    Execute the split_channels command.
    '''
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Check if input is a file or directory
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else None
    
    if input_path.is_file():
        # Process single file
        logger.info(f"Processing single file: {input_path}")
        try:
            output_files = split_channels(
                input_path,
                output_dir=output_path,
                channel_names=args.sequence,
                suffix_separator=args.separator
            )
            
            logger.info(f"Split {len(output_files)} channels from {input_path.name}")
            for channel, path in output_files.items():
                logger.info(f"  - Channel {channel}: {path}")
                
        except Exception as e:
            logger.error(f"Error: {e}")
            raise
    
    elif input_path.is_dir():
        # Process directory
        logger.info(f"Processing directory: {input_path}")
        try:
            num_processed = batch_split_channels(
                input_path,
                output_dir=output_path,
                channel_names=args.sequence,
                suffix_separator=args.separator,
                recursive=args.recursive
            )
            logger.info(f"Processed {num_processed} images")
        except Exception as e:
            logger.error(f"Error: {e}")
            raise
    
    else:
        logger.error(f"Input path {input_path} does not exist")
        raise FileNotFoundError(f"Input path {input_path} does not exist")