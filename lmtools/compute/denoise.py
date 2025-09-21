'''
author: zyx
date: 2025-09-20
last_modified: 2025-09-20
description:
    Functions for imaging denoising with various algorithms:
        CLAHE
'''
import numpy as np
from typing import Union, Tuple, Optional

import cv2
from skimage import exposure
from skimage.util import img_as_ubyte, img_as_float
from skimage.filters import gaussian

# GPU support with PyTorch
try:
    import torch
    import torch.nn.functional as F
    HAS_GPU = torch.cuda.is_available()
except ImportError:
    HAS_GPU = False
    torch = None
    F = None


def clahe_denoise(image: np.ndarray,
                  clip_limit: float = 2.0,
                  tile_grid_size: Tuple[int, int] = (8, 8),
                  gaussian_sigma: float = 0.5,
                  use_gpu: bool = True) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for image denoising.

    CLAHE enhances local contrast while limiting noise amplification, making it effective
    for denoising by improving signal-to-noise ratio in local regions.

    Parameters
    ----------
    image : np.ndarray
        Input image (2D grayscale or 3D with channels as last dimension)
    clip_limit : float, default=2.0
        Threshold for contrast limiting. Higher values allow more contrast enhancement
    tile_grid_size : tuple of int, default=(8, 8)
        Size of the grid for adaptive histogram equalization
    gaussian_sigma : float, default=0.5
        Standard deviation for Gaussian smoothing applied after CLAHE
    use_gpu : bool, default=True
        Whether to use GPU acceleration if available

    Returns
    -------
    np.ndarray
        Denoised image with same shape and dtype as input

    Raises
    ------
    ValueError
        If input image has unsupported dimensions
    RuntimeError
        If PyTorch is not available and GPU processing is requested
    """
    if image.ndim not in [2, 3]:
        raise ValueError(f"Input image must be 2D or 3D, got {image.ndim}D")

    # Store original dtype for output conversion
    original_dtype = image.dtype

    # Use GPU if available and requested
    if use_gpu and HAS_GPU:
        return _clahe_denoise_gpu(image, clip_limit, tile_grid_size,
                                  gaussian_sigma, original_dtype)
    else:
        return _clahe_denoise_cpu(image, clip_limit, tile_grid_size,
                                  gaussian_sigma, original_dtype)


def _clahe_denoise_cpu(image: np.ndarray,
                       clip_limit: float,
                       tile_grid_size: Tuple[int, int],
                       gaussian_sigma: float,
                       original_dtype) -> np.ndarray:
    """CPU implementation of CLAHE denoising."""

    # Convert to float for processing
    if image.dtype != np.float64:
        image_float = img_as_float(image)
    else:
        image_float = image.copy()

    if image.ndim == 2:
        # 2D grayscale image
        if HAS_OPENCV:
            # Use OpenCV CLAHE (generally faster)
            image_uint8 = img_as_ubyte(image_float)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            enhanced = clahe.apply(image_uint8)
            enhanced_float = img_as_float(enhanced)
        else:
            # Use scikit-image CLAHE
            enhanced_float = exposure.equalize_adapthist(
                image_float,
                clip_limit=clip_limit/100.0,  # scikit-image uses different scale
                nbins=256
            )

        # Apply light Gaussian smoothing to reduce residual noise
        if gaussian_sigma > 0:
            denoised = gaussian(enhanced_float, sigma=gaussian_sigma, preserve_range=True)
        else:
            denoised = enhanced_float

    elif image.ndim == 3:
        # 3D image (assumed channels-last)
        denoised = np.zeros_like(image_float)

        for c in range(image_float.shape[2]):
            channel = image_float[:, :, c]

            if HAS_OPENCV:
                channel_uint8 = img_as_ubyte(channel)
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
                enhanced_channel = clahe.apply(channel_uint8)
                enhanced_channel_float = img_as_float(enhanced_channel)
            else:
                enhanced_channel_float = exposure.equalize_adapthist(
                    channel,
                    clip_limit=clip_limit/100.0,
                    nbins=256
                )

            # Apply light Gaussian smoothing
            if gaussian_sigma > 0:
                denoised[:, :, c] = gaussian(enhanced_channel_float,
                                           sigma=gaussian_sigma,
                                           preserve_range=True)
            else:
                denoised[:, :, c] = enhanced_channel_float

    # Convert back to original dtype
    if original_dtype == np.uint8:
        return img_as_ubyte(denoised)
    elif original_dtype == np.uint16:
        return (denoised * 65535).astype(np.uint16)
    else:
        return denoised.astype(original_dtype)


def _clahe_denoise_gpu(image: np.ndarray,
                       clip_limit: float,
                       tile_grid_size: Tuple[int, int],
                       gaussian_sigma: float,
                       original_dtype) -> np.ndarray:
    """GPU implementation of CLAHE denoising using PyTorch."""

    if not torch:
        raise RuntimeError("PyTorch is required for GPU CLAHE processing")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert to PyTorch tensor and move to GPU
    if image.dtype != np.float64:
        image_float = img_as_float(image)
    else:
        image_float = image.copy()

    if image.ndim == 2:
        # CLAHE on CPU (OpenCV), then Gaussian smoothing on GPU
        image_uint8 = img_as_ubyte(image_float)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced = clahe.apply(image_uint8)
        enhanced_float = img_as_float(enhanced)

        # Apply Gaussian smoothing on GPU using PyTorch
        if gaussian_sigma > 0:
            # Convert to PyTorch tensor and move to GPU
            enhanced_tensor = torch.from_numpy(enhanced_float).float().unsqueeze(0).unsqueeze(0).to(device)

            # Create Gaussian kernel
            kernel_size = int(2 * np.ceil(2 * gaussian_sigma) + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1

            # Apply Gaussian blur using PyTorch
            denoised_tensor = _gaussian_blur_torch(enhanced_tensor, gaussian_sigma, kernel_size)
            denoised = denoised_tensor.squeeze().cpu().numpy()
        else:
            denoised = enhanced_float

    elif image.ndim == 3:
        # Process each channel
        denoised = np.zeros_like(image_float)

        for c in range(image_float.shape[2]):
            channel = image_float[:, :, c]

            # CLAHE on CPU
            channel_uint8 = img_as_ubyte(channel)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            enhanced_channel = clahe.apply(channel_uint8)
            enhanced_channel_float = img_as_float(enhanced_channel)

            # Apply Gaussian smoothing on GPU
            if gaussian_sigma > 0:
                enhanced_tensor = torch.from_numpy(enhanced_channel_float).float().unsqueeze(0).unsqueeze(0).to(device)
                kernel_size = int(2 * np.ceil(2 * gaussian_sigma) + 1)
                if kernel_size % 2 == 0:
                    kernel_size += 1

                denoised_tensor = _gaussian_blur_torch(enhanced_tensor, gaussian_sigma, kernel_size)
                denoised[:, :, c] = denoised_tensor.squeeze().cpu().numpy()
            else:
                denoised[:, :, c] = enhanced_channel_float

    # Convert back to original dtype
    if original_dtype == np.uint8:
        return img_as_ubyte(denoised)
    elif original_dtype == np.uint16:
        return (denoised * 65535).astype(np.uint16)
    else:
        return denoised.astype(original_dtype)


def _gaussian_blur_torch(tensor: 'torch.Tensor', sigma: float, kernel_size: int) -> 'torch.Tensor':
    """Apply Gaussian blur using PyTorch."""
    # Create 1D Gaussian kernel
    x = torch.arange(kernel_size, dtype=torch.float32, device=tensor.device)
    x = x - kernel_size // 2
    kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

    # Reshape for separable convolution
    kernel_x = kernel_1d.view(1, 1, 1, -1)
    kernel_y = kernel_1d.view(1, 1, -1, 1)

    # Apply separable Gaussian convolution
    padding = kernel_size // 2
    blurred = F.conv2d(tensor, kernel_x, padding=(0, padding))
    blurred = F.conv2d(blurred, kernel_y, padding=(padding, 0))

    return blurred


def adaptive_clahe_denoise(image: np.ndarray,
                          min_clip_limit: float = 1.0,
                          max_clip_limit: float = 4.0,
                          tile_grid_size: Tuple[int, int] = (8, 8),
                          gaussian_sigma: float = 0.5,
                          noise_threshold: float = 0.1,
                          use_gpu: bool = True) -> np.ndarray:
    """
    Apply adaptive CLAHE denoising that adjusts clip limit based on local noise estimation.

    This function estimates local noise levels and adapts the CLAHE clip limit accordingly,
    providing stronger denoising in noisier regions while preserving detail in cleaner areas.

    Parameters
    ----------
    image : np.ndarray
        Input image (2D grayscale or 3D with channels as last dimension)
    min_clip_limit : float, default=1.0
        Minimum clip limit for low-noise regions
    max_clip_limit : float, default=4.0
        Maximum clip limit for high-noise regions
    tile_grid_size : tuple of int, default=(8, 8)
        Size of the grid for adaptive histogram equalization
    gaussian_sigma : float, default=0.5
        Standard deviation for Gaussian smoothing
    noise_threshold : float, default=0.1
        Threshold for noise level estimation (0-1 scale)
    use_gpu : bool, default=True
        Whether to use GPU acceleration if available

    Returns
    -------
    np.ndarray
        Denoised image with same shape and dtype as input
    """

    # Estimate local noise using Laplacian variance
    if image.ndim == 2:
        laplacian_var = cv2.Laplacian(img_as_ubyte(img_as_float(image)), cv2.CV_64F).var()
    else:
        # For 3D images, compute average variance across channels
        laplacian_vars = []
        for c in range(image.shape[2]):
            channel = img_as_ubyte(img_as_float(image[:, :, c]))
            laplacian_vars.append(cv2.Laplacian(channel, cv2.CV_64F).var())
        laplacian_var = np.mean(laplacian_vars)

    # Normalize variance to [0, 1] range (empirically determined scaling)
    normalized_noise = min(laplacian_var / 1000.0, 1.0)

    # Adapt clip limit based on noise level
    if normalized_noise < noise_threshold:
        clip_limit = min_clip_limit
    else:
        # Linear interpolation between min and max clip limits
        clip_limit = min_clip_limit + (max_clip_limit - min_clip_limit) * normalized_noise

    return clahe_denoise(image, clip_limit=clip_limit, tile_grid_size=tile_grid_size,
                        gaussian_sigma=gaussian_sigma, use_gpu=use_gpu)
