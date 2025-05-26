'''
author: zyx
date: 2025-05-23
last modified: 2025-05-25
description: 
    Functions for morphological operations
'''
import numpy as np

from skimage.morphology import erosion, ball

def erode_mask_2D_with_ball(mask: np.ndarray,
                            radius: int) -> np.ndarray:
    """
    Erode a 2D binary_mask by a disk of given radius.
    """
    disk = ball(radius)[radius]
    return erosion(mask > 0, disk)

def generate_2D_donut(mask: np.ndarray,
                      radius: int) -> np.ndarray:
    """
    Generate a 2D donut mask by eroding the input mask with a disk of given radius.
    """
    eroded_mask = erode_mask_2D_with_ball(mask, radius)
    donut_mask = mask > 0
    donut_mask[eroded_mask] = False
    return donut_mask
