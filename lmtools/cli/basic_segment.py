'''
CLI command for basic segmentation methods.
'''
import logging
from pathlib import Path
from lmtools.seg.basic_segmentation import (
    threshold_segment,
    watershed_segment,
    region_growing_segment
)

logger = logging.getLogger(__name__)

def add_arguments(parser):
    '''
    Add command line arguments for the basic_segment command.
    '''
    subparsers = parser.add_subparsers(dest="method", help="Segmentation method", required=True)
    
    # Threshold segmentation
    threshold_parser = subparsers.add_parser("threshold", help="Threshold-based segmentation")
    threshold_parser.add_argument("input", type=str, help="Input image path")
    threshold_parser.add_argument("output", type=str, help="Output segmentation path")
    threshold_parser.add_argument("--threshold-method", type=str, default="otsu",
                                choices=["simple", "otsu", "adaptive", "local", "yen", "li", "triangle"],
                                help="Thresholding method")
    threshold_parser.add_argument("--threshold", type=float, help="Manual threshold value (0-1)")
    threshold_parser.add_argument("--block-size", type=int, default=35, 
                                help="Block size for adaptive methods")
    threshold_parser.add_argument("--offset", type=float, default=0.0,
                                help="Offset for adaptive methods")
    threshold_parser.add_argument("--no-remove-small", action="store_false", dest="remove_small",
                                help="Don't remove small objects")
    threshold_parser.add_argument("--min-size", type=int, default=50,
                                help="Minimum object size in pixels")
    threshold_parser.add_argument("--no-fill-holes", action="store_false", dest="fill_holes",
                                help="Don't fill holes")
    threshold_parser.add_argument("--connectivity", type=int, default=1, choices=[1, 2],
                                help="Connectivity for determining connected components")
    threshold_parser.add_argument("--binary", action="store_false", dest="return_labels",
                                help="Return binary mask instead of labeled objects")
    
    # Watershed segmentation
    watershed_parser = subparsers.add_parser("watershed", help="Watershed segmentation")
    watershed_parser.add_argument("input", type=str, help="Input image path")
    watershed_parser.add_argument("output", type=str, help="Output segmentation path")
    watershed_parser.add_argument("--threshold-method", type=str, default="otsu",
                                choices=["simple", "otsu", "yen", "li", "triangle"],
                                help="Method for initial thresholding")
    watershed_parser.add_argument("--threshold", type=float, help="Manual threshold value (0-1)")
    watershed_parser.add_argument("--no-distance-transform", action="store_false", dest="distance_transform",
                                help="Don't use distance transform for markers")
    watershed_parser.add_argument("--distance-threshold", type=float,
                                help="Threshold for distance transform (fraction of max)")
    watershed_parser.add_argument("--min-distance", type=int, default=10,
                                help="Minimum distance between peaks")
    watershed_parser.add_argument("--connectivity", type=int, default=1, choices=[1, 2],
                                help="Connectivity for watershed")
    watershed_parser.add_argument("--no-remove-small", action="store_false", dest="remove_small",
                                help="Don't remove small objects")
    watershed_parser.add_argument("--min-size", type=int, default=50,
                                help="Minimum object size in pixels")
    watershed_parser.add_argument("--no-fill-holes", action="store_false", dest="fill_holes",
                                help="Don't fill holes")
    
    # Region growing segmentation
    region_parser = subparsers.add_parser("region", help="Region growing segmentation")
    region_parser.add_argument("input", type=str, help="Input image path")
    region_parser.add_argument("output", type=str, help="Output segmentation path")
    region_parser.add_argument("--seed-method", type=str, default="intensity",
                             choices=["intensity", "grid", "random"],
                             help="Method to generate seeds")
    region_parser.add_argument("--num-seeds", type=int, default=100,
                             help="Number of seed points or regions")
    region_parser.add_argument("--threshold-method", type=str, default="otsu",
                             choices=["otsu", "yen", "li", "triangle"],
                             help="Method for intensity thresholding")
    region_parser.add_argument("--compactness", type=float, default=0.001,
                             help="Compactness parameter for SLIC")
    region_parser.add_argument("--no-remove-small", action="store_false", dest="remove_small",
                             help="Don't remove small objects")
    region_parser.add_argument("--min-size", type=int, default=50,
                             help="Minimum object size in pixels")
    
    # Common options
    for p in [threshold_parser, watershed_parser, region_parser]:
        p.add_argument("--verbose", action="store_true", help="Enable verbose logging")


def main(args):
    '''
    Execute the basic_segment command.
    '''
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if args.method == "threshold":
        logger.info(f"Running threshold segmentation on {args.input}")
        threshold_segment(
            args.input,
            args.output,
            method=args.threshold_method,  # FIXED: use threshold_method instead of method
            threshold_value=args.threshold,
            block_size=args.block_size,
            offset=args.offset,
            remove_small_objects=args.remove_small,
            min_size=args.min_size,
            fill_holes=args.fill_holes,
            connectivity=args.connectivity,
            return_labels=args.return_labels
        )
    
    elif args.method == "watershed":
        logger.info(f"Running watershed segmentation on {args.input}")
        watershed_segment(
            args.input,
            args.output,
            threshold_method=args.threshold_method,
            threshold_value=args.threshold,
            distance_transform=args.distance_transform,
            distance_threshold=args.distance_threshold,
            min_distance=args.min_distance,
            watershed_connectivity=args.connectivity,
            remove_small_objects=args.remove_small,
            min_size=args.min_size,
            fill_holes=args.fill_holes
        )
    
    elif args.method == "region":
        logger.info(f"Running region growing segmentation on {args.input}")
        region_growing_segment(
            args.input,
            args.output,
            seed_method=args.seed_method,
            num_seeds=args.num_seeds,
            threshold_method=args.threshold_method,
            compactness=args.compactness,
            remove_small_objects=args.remove_small,
            min_size=args.min_size
        )
    
    else:
        raise ValueError(f"Unknown segmentation method: {args.method}")