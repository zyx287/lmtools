"""
CLI command to load and visualize ND2 files with Napari.
"""
import argparse
from lmtools.io import load_nd2

def add_arguments(parser):
    """
    Add command line arguments for the load_nd2 command.
    """
    parser.add_argument(
        "file_path", 
        type=str, 
        help="Path to the .nd2 file."
    )

def main(args):
    """
    Execute the load_nd2 command.
    """
    load_nd2(args.file_path)