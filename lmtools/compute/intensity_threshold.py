'''
author: zyx
date: 2025-06-10
last_modified: 2025-06-10
description: 
    Functions for computing intensity thresholds
'''
import numpy as np

from skimage.filters import threshold_otsu
from sklearn.mixture import GaussianMixture

def compute_otsu_threshold(values: np.ndarray) -> float:
    """
    Compute Otsu's threshold for a given array of values.
    Parameters:
        values (np.ndarray): 1D array of intensity values.
    Returns:
        float: Otsu's threshold value.
    """
    return threshold_otsu(values)

def compute_gmm_component(values: np.ndarray,
                          n_components: None)-> GaussianMixture:
    '''
    Compute Gaussian Mixture Model (GMM) for intensity values.
    Parameters:
        values (np.ndarray): 1D array of intensity values.
        n_components (int, optional): Number of GMM components. If None, it will be determined using BIC.
    Returns:
        GaussianMixture: Fitted GMM model.
    '''
    if n_components is None:
        bic_scores = []
        components_range = range(1, 6)
        for n_component in components_range:
            gmm = GaussianMixture(n_components=n_component, random_state=42)
            gmm.fit(values.reshape(-1, 1))
            bic_scores.append(gmm.bic(values.reshape(-1, 1)))
        # Choose optimal number of components (lowest BIC)
        optimal_components = components_range[np.argmin(bic_scores)]
        n_components = optimal_components
    
    gmm_model = GaussianMixture(n_components=n_components, random_state=42)
    gmm_model.fit(values.reshape(-1, 1))

    return gmm_model

    