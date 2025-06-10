'''
CLI command for filtering segmentation objects based on intensity.
'''
import logging
from pathlib import Path
from lmtools.seg.intensity_filter import intensity_filter, visualize_intensity_regions

logger = logging.getLogger(__name__)

def add_arguments(parser):
    '''
    Add command line arguments for the intensity_filter command.
    '''
    parser.add_argument(
        "segmentation", 
        type=str, 
        help="Path to segmentation mask"
    )
    parser.add_argument(
        "intensity", 
        type=str, 
        help="Path to intensity image"
    )
    parser.add_argument(
        "output", 
        type=str, 
        help="Path to save filtered segmentation"
    )
    parser.add_argument(
        "--threshold", 
        type=float, 
        help="Intensity threshold (if not specified, calculated automatically)"
    )
    parser.add_argument(
        "--threshold-method", 
        type=str, 
        default="otsu", 
        choices=["otsu", "percentile"], 
        help="Method for automatic threshold calculation"
    )
    parser.add_argument(
        "--percentile", 
        type=float, 
        default=25.0, 
        help="Percentile for threshold calculation (used with percentile method)"
    )
    parser.add_argument(
        "--region", 
        type=str, 
        default="whole", 
        choices=["whole", "membrane", "inner", "outer"],
        help="Region to consider for intensity calculation"
    )
    parser.add_argument(
        "--membrane-width", 
        type=int, 
        default=2,
        help="Width of membrane/border in pixels"
    )
    parser.add_argument(
        "--no-histogram", 
        action="store_false", 
        dest="histogram",
        help="Don't generate histogram"
    )
    parser.add_argument(
        "--figure", 
        type=str, 
        help="Path to save histogram figure"
    )
    parser.add_argument(
        "--visualize-regions", 
        action="store_true",
        help="Visualize the different regions"
    )
    parser.add_argument(
        "--label", 
        type=int, 
        help="Label ID to visualize (for --visualize-regions)"
    )
    parser.add_argument(
        "--vis-output", 
        type=str, 
        help="Path to save region visualization"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose logging"
    )

def main(args):
    '''
    Execute the intensity_filter command.
    '''
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Visualize regions if requested
    if args.visualize_regions:
        logger.info("Visualizing intensity regions")
        visualize_intensity_regions(
            args.segmentation,
            args.intensity,
            output_path=args.vis_output,
            label_id=args.label,
            membrane_width=args.membrane_width
        )
    
    # Filter segmentation based on intensity
    logger.info(f"Filtering segmentation: {args.segmentation}")
    logger.info(f"Using intensity image: {args.intensity}")
    
    intensity_filter(
        args.segmentation,
        args.intensity,
        output_path=args.output,
        threshold=args.threshold,
        threshold_method=args.threshold_method,
        percentile=args.percentile,
        region_type=args.region,
        membrane_width=args.membrane_width,
        plot_histogram=args.histogram,
        figure_path=args.figure
    )
    
    logger.info(f"Saved filtered segmentation to {args.output}")