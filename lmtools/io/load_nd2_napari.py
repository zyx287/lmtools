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

def load_nd2(file_path:str)->None:
    """
    Load .nd2 file and display it using Napari
    """
    with nd2.ND2File(file_path) as nd2_file:
        image_data = nd2_file.asarray()  # Convert to NumPy array

    # Launch Napari viewer
    viewer = napari.Viewer()
    viewer.add_image(image_data, name="ND2 Image")

if __name__ == "__main__":
    file_path = "path/to/your/file.nd2"  # Replace with your .nd2 file path
    load_nd2(file_path)
