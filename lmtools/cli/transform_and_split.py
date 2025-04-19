"""
CLI command to transform and split unusual multi-dimensional images.
"""
import logging
from pathlib import Path
from lmtools.io.dimension_transform import transform_and_split, batch_transform_and_split

logger = logging.getLogger(__name__)

def add_arguments(parser):
    """
    Add command line arguments for the transform_and_split command.
    """
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
        "--channel-axis", 
        type=int, 
        help="Channel axis (dimension index, e.g. 0 for first dimension)"
    )
    parser.add_argument(
        "--transpose", 
        type=int, 
        nargs="+", 
        help="Transpose axes order (e.g. '0 2 1' to swap Y and Z dimensions)"
    )
    parser.add_argument(
        "--separator", 
        type=str, 
        default="_", 
        help="Separator between filename and channel name (default: _)"
    )
    parser.add_argument(
        "--normalize", 
        action="store_true", 
        help="Normalize channel intensities to 0-1 range"
    )
    parser.add_argument(
        "--no-metadata", 
        action="store_false", 
        dest="metadata", 
        help="Don't save metadata JSON file"
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
    """
    Execute the transform_and_split command.
    """
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
            output_files = transform_and_split(
                input_path,
                output_dir=output_path,
                channel_names=args.sequence,
                channel_axis=args.channel_axis,
                suffix_separator=args.separator,
                transpose_axes=args.transpose,
                normalize=args.normalize,
                save_metadata=args.metadata
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
            num_processed = batch_transform_and_split(
                input_path,
                output_dir=output_path,
                channel_names=args.sequence,
                channel_axis=args.channel_axis,
                suffix_separator=args.separator,
                transpose_axes=args.transpose,
                normalize=args.normalize,
                save_metadata=args.metadata,
                recursive=args.recursive
            )
            logger.info(f"Processed {num_processed} images")
        except Exception as e:
            logger.error(f"Error: {e}")
            raise
    
    else:
        logger.error(f"Input path {input_path} does not exist")
        raise FileNotFoundError(f"Input path {input_path} does not exist")