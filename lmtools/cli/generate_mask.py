'''
CLI command to generate segmentation masks from QuPath GeoJSON files.
'''
from lmtools.seg import generate_segmentation_mask

def add_arguments(parser):
    '''
    Add command line arguments for the generate_mask command.
    '''
    parser.add_argument(
        "geojson_path", 
        type=str, 
        help="Path to the QuPath .geojson file."
    )
    parser.add_argument(
        "output_dir", 
        type=str, 
        help="Directory to save output masks."
    )
    parser.add_argument(
        "image_width", 
        type=int, 
        help="Width of the original image."
    )
    parser.add_argument(
        "image_height", 
        type=int, 
        help="Height of the original image."
    )
    parser.add_argument(
        "--inner_holes", 
        action="store_true", 
        help="Include inner holes in the mask."
    )
    parser.add_argument(
        "--downsample_factor", 
        type=int, 
        default=4, 
        help="Downsampling factor (default: 4)."
    )

def main(args):
    '''
    Execute the generate_mask command.
    '''
    success = generate_segmentation_mask(
        geojson_path=args.geojson_path,
        output_dir=args.output_dir,
        image_width=args.image_width,
        image_height=args.image_height,
        inner_holes=args.inner_holes,
        downsample_factor=args.downsample_factor
    )
    
    if not success:
        raise RuntimeError("Failed to generate segmentation mask.")