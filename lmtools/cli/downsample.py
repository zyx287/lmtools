"""
CLI command to downsample TIFF images.
"""
import logging
from pathlib import Path
from lmtools.io.image_downsampling import downsample_image, batch_downsample

logger = logging.getLogger(__name__)

def add_arguments(parser):
    """
    Add command line arguments for the downsample command.
    """
    parser.add_argument(
        "input", 
        type=str, 
        help="Input TIFF image or directory"
    )
    parser.add_argument(
        "output", 
        type=str, 
        help="Output TIFF image or directory"
    )
    parser.add_argument(
        "--scale", 
        type=float, 
        default=0.5, 
        help="Scale factor (default: 0.5)"
    )
    parser.add_argument(
        "--method", 
        type=str, 
        default="bicubic", 
        choices=["nearest", "bilinear", "bicubic", "lanczos", "area", "gaussian", "median"],
        help="Downsampling method (default: bicubic)"
    )
    parser.add_argument(
        "--library", 
        type=str, 
        default="auto",
        choices=["auto", "pillow", "opencv", "skimage"],
        help="Library to use for downsampling (default: auto)"
    )
    parser.add_argument(
        "--no-preserve-range", 
        action="store_false", 
        dest="preserve_range",
        help="Do not preserve intensity range"
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
    Execute the downsample command.
    """
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Check if input is a file or directory
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if input_path.is_file():
        # Process single file
        logger.info(f"Processing single file: {input_path}")
        try:
            downsample_image(
                input_path,
                output_path,
                scale_factor=args.scale,
                method=args.method,
                preserve_range=args.preserve_range,
                library=args.library
            )
            logger.info(f"Image downsampled successfully: {output_path}")
        except Exception as e:
            logger.error(f"Error: {e}")
            raise
    
    elif input_path.is_dir():
        # Process directory
        logger.info(f"Processing directory: {input_path}")
        try:
            num_processed = batch_downsample(
                input_path,
                output_path,
                scale_factor=args.scale,
                method=args.method,
                preserve_range=args.preserve_range,
                library=args.library,
                recursive=args.recursive
            )
            logger.info(f"Processed {num_processed} images")
        except Exception as e:
            logger.error(f"Error: {e}")
            raise
    
    else:
        logger.error(f"Input path {input_path} does not exist")
        raise FileNotFoundError(f"Input path {input_path} does not exist")