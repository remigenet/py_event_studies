"""
event_studies_test_statistics.py

This module provides low-level functions for computing various test statistics
in event studies. These functions are designed for efficient computation across
multiple portfolios, clusters, and methods simultaneously.

The functions in this module are not intended for direct use by end-users of the
package. Instead, they serve as the computational backbone for higher-level
functions that provide a more user-friendly interface.

Input array shapes:
- event_residual: (n_portfolios, n_clusters, n_methods, n_stocks)
- estim_residuals: (n_portfolios, n_clusters, n_methods, estimation_period_length, n_stocks)
"""

import numpy as np

def standard_test(event_residual: np.ndarray, estim_residuals: np.ndarray) -> np.ndarray:
    """
    Compute the standard t-test for abnormal returns.

    This function calculates the t-statistic for each portfolio, cluster, and method
    using the event residuals and the standard deviation of the estimation period residuals.

    Args:
        event_residual (np.ndarray): Abnormal returns at the event date.
            Shape: (n_portfolios, n_clusters, n_methods, n_stocks)
        estim_residuals (np.ndarray): Residuals from the estimation period.
            Shape: (n_portfolios, n_clusters, n_methods, estimation_period_length, n_stocks)

    Returns:
        np.ndarray: T-statistics for each portfolio, cluster, and method.
            Shape: (n_portfolios, n_clusters, n_methods)
    """
    return np.mean(event_residual, axis=-1) / np.sqrt(np.var(estim_residuals, axis=(3, 4)) / estim_residuals.shape[-1])

def ordinary_cross_sec_test(event_residual: np.ndarray) -> np.ndarray:
    """
    Perform a cross-sectional t-test on abnormal returns.

    This function computes the cross-sectional t-statistic for each portfolio,
    cluster, and method. It uses only the event date residuals, making it robust 
    to event-induced volatility changes but potentially less powerful than other tests.

    Args:
        event_residual (np.ndarray): Abnormal returns at the event date.
            Shape: (n_portfolios, n_clusters, n_methods, n_stocks)

    Returns:
        np.ndarray: Cross-sectional t-statistics for each portfolio, cluster, and method.
            Shape: (n_portfolios, n_clusters, n_methods)
    """
    event_residual_mean = np.mean(event_residual, axis=-1)
    event_residual_var = np.var(event_residual, axis=-1)
    return event_residual_mean / np.sqrt(event_residual_var / event_residual.shape[-1])

def bmp_test(event_residual: np.ndarray, estim_residuals: np.ndarray, event_d: float) -> np.ndarray:
    """
    Implement the Boehmer, Masumeci, and Poulsen (1991) test.

    This test adjusts for event-induced volatility changes by standardizing the
    abnormal returns using the estimation period standard deviation and then
    performing a cross-sectional t-test on these standardized returns.

    Args:
        event_residual (np.ndarray): Abnormal returns at the event date.
            Shape: (n_portfolios, n_clusters, n_methods, n_stocks)
        estim_residuals (np.ndarray): Residuals from the estimation period.
            Shape: (n_portfolios, n_clusters, n_methods, estimation_period_length, n_stocks)
        event_d (float): Adjustment factor for the event window.

    Returns:
        np.ndarray: BMP test statistics for each portfolio, cluster, and method.
            Shape: (n_portfolios, n_clusters, n_methods)
    """
    nb_expl_var_map = (np.array([1, 2, 4, 6, 1, 3, 5, 1, 1, 1]) + 1).reshape(1,1,-1,1)
    
    T1 = estim_residuals.shape[3]
    sigma_raw = np.std(estim_residuals, axis=3)
    nb_expl_var = np.broadcast_to(nb_expl_var_map, sigma_raw.shape)
    sigma_adj = sigma_raw * np.sqrt(T1 - 1) / (np.sqrt(T1 - nb_expl_var) * np.sqrt(1 + event_d))
    event_residual_bar = event_residual / sigma_adj
    event_residual_bar_mean = np.mean(event_residual_bar, axis=-1)
    event_residual_bar_var = np.var(event_residual_bar, axis=-1)
    event_residual_bar_count = event_residual_bar.shape[-1]
    return event_residual_bar_mean / np.sqrt(event_residual_bar_var / event_residual_bar_count)

def create_avg_corrs(estim_residuals: np.ndarray) -> np.ndarray:
    """
    Calculate average correlations between residuals.

    This function computes the average pairwise correlation of residuals for each
    portfolio, cluster, and method. These correlations are used in the Kolari and
    Pynnönen (2010) test to account for cross-sectional correlation.

    Args:
        estim_residuals (np.ndarray): Residuals from the estimation period.
            Shape: (n_portfolios, n_clusters, n_methods, estimation_period_length, n_stocks)

    Returns:
        np.ndarray: Average correlations for each portfolio, cluster, and method.
            Shape: (n_portfolios, n_clusters, n_methods)
    """
    avg_cors = np.zeros(estim_residuals.shape[:3])
    for i in range(estim_residuals.shape[0]):
        for j in range(estim_residuals.shape[1]):
            for k in range(estim_residuals.shape[2]):
                residuals = estim_residuals[i, j, k]
                rho_mat = np.corrcoef(residuals.T)
                mask = ~np.eye(rho_mat.shape[0], dtype=bool)
                avg_cors[i, j, k] = np.sum(rho_mat[mask]) / np.sum(mask)
    return avg_cors

def kp_test(event_residual: np.ndarray, estim_residuals: np.ndarray, avg_cors: np.ndarray) -> np.ndarray:
    """
    Implement the Kolari and Pynnönen (2010) test.

    This test extends the BMP test by accounting for cross-sectional correlation
    in the abnormal returns. It uses the average correlations computed by the
    create_avg_corrs function to adjust the test statistic.

    Args:
        event_residual (np.ndarray): Abnormal returns at the event date.
            Shape: (n_portfolios, n_clusters, n_methods, n_stocks)
        estim_residuals (np.ndarray): Residuals from the estimation period.
            Shape: (n_portfolios, n_clusters, n_methods, estimation_period_length, n_stocks)
        avg_cors (np.ndarray): Average correlations for each portfolio, cluster, and method.
            Shape: (n_portfolios, n_clusters, n_methods)

    Returns:
        np.ndarray: KP test statistics for each portfolio, cluster, and method.
            Shape: (n_portfolios, n_clusters, n_methods)
    """
    nb_expl_var_map = (np.array([1, 2, 4, 6, 1, 3, 5, 1, 1, 1]) + 1).reshape(1,1,-1,1)

    T1 = estim_residuals.shape[3]
    sigma_raw = np.sqrt(np.sum(np.square(estim_residuals - np.mean(estim_residuals, axis=3, keepdims=True)), axis=3) / (T1 - 1))
    nb_expl_var = np.broadcast_to(nb_expl_var_map, sigma_raw.shape)
    sigma_adj = sigma_raw * np.sqrt(T1 - 1) / np.sqrt(T1 - nb_expl_var)
    event_residual_bar = event_residual / sigma_adj
    event_residual_bar_mean = np.mean(event_residual_bar, axis=-1)
    event_residual_bar_var = np.var(event_residual_bar, axis=-1)
    event_residual_bar_count = event_residual_bar.shape[-1]
    t_test = event_residual_bar_mean / np.sqrt(event_residual_bar_var) * np.sqrt(event_residual_bar_count) * \
             np.sqrt((1 - avg_cors) / (1 + (event_residual_bar_count - 1) * avg_cors))
    return t_test