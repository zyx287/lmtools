'''
CLI command to run cellpose segmentation pipeline using a YAML configuration file.
'''
import logging
from lmtools.seg import run_pipeline

logger = logging.getLogger(__name__)

def add_arguments(parser):
    '''
    Add command line arguments for the cellpose_segment command.
    '''
    parser.add_argument(
        "config", 
        type=str, 
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose logging"
    )

def main(args):
    '''
    Execute the cellpose_segment command.
    '''
    # Set logging level
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info(f"Starting cellpose segmentation with config: {args.config}")
    
    try:
        # Run the pipeline with the specified config file
        output_files = run_pipeline(args.config)
        
        if output_files:
            logger.info(f"Successfully generated {len(output_files)} mask files")
        else:
            logger.warning("No mask files were generated")
    
    except Exception as e:
        logger.error(f"Error running cellpose segmentation: {e}")
        raise