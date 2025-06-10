'''
CLI command to extract masks from Cellpose segmentation outputs.
'''
from lmtools.seg import maskExtract

def add_arguments(parser):
    '''
    Add command line arguments for the extract_mask command.
    '''
    parser.add_argument(
        "--file_path", 
        type=str, 
        required=True, 
        help="Path to the mask.npy file"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        required=True, 
        help="Path to save the extracted mask"
    )

def main(args):
    '''
    Execute the extract_mask command.
    '''
    maskExtract(
        file_path=args.file_path,
        output_path=args.output_path
    )