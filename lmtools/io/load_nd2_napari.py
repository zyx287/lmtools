'''
author: zyx
date: 2025-03-31
last_modified: 2025-03-31
description: 
    Load .nd2 file and display it using Napari
'''
import napari
import nd2
import numpy as np
from typing import Optional

def load_nd2(file_path: str) -> Optional[napari.Viewer]:
    '''
    Load .nd2 file and display it using Napari
    
    Parameters
    ----------
    file_path : str
        Path to the .nd2 file
        
    Returns
    -------
    napari.Viewer or None
        The Napari viewer instance, or None if loading failed
    '''
    try:
        with nd2.ND2File(file_path) as nd2_file:
            image_data = nd2_file.asarray()  # Convert to NumPy array

        # Launch Napari viewer
        viewer = napari.Viewer()
        viewer.add_image(image_data, name="ND2 Image")
        return viewer
    except Exception as e:
        print(f"Error loading ND2 file: {e}")
        return None

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "path/to/your/file.nd2"  # Replace with your .nd2 file path
    load_nd2(file_path)