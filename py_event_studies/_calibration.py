"""
Private Computational Functions for Event Study Analysis

This module contains highly optimized computational functions for event study analysis.
These functions are not intended for direct use by end users due to their complexity
and focus on performance optimization. The code prioritizes memory efficiency and 
computational speed, which may impact readability.

Note: These functions use advanced techniques like Numba JIT compilation and parallel
processing to achieve high performance. Modifications should be made with caution.
"""
from typing import List, Optional, Tuple

import numpy as np
import numba as nb

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet

from joblib import Parallel, delayed


@nb.njit(forceinline=True, looplift=True, inline='always', no_cfunc_wrapper=True, no_rewrites=True, nogil=True, cache=True)
def reg(X_estim: np.ndarray, y_estim: np.ndarray, X_event: np.ndarray, y_event: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform regression analysis for event study.

    Args:
        X_estim (np.ndarray): Estimation period explanatory variables.
        y_estim (np.ndarray): Estimation period dependent variable.
        X_event (np.ndarray): Event period explanatory variables.
        y_event (np.ndarray): Event period dependent variable.

    Returns:
        Tuple containing estimation residuals, event residuals, and diagnostics.
    """
    X_estim_with_intercept = np.hstack((np.ones((X_estim.shape[0], 1)), X_estim))
    X_event_with_intercept = np.hstack((np.ones((X_event.shape[0], 1)), X_event))
    
    X_combined = np.vstack((X_estim_with_intercept, X_event_with_intercept))
    X_combined_without_intercept = X_combined[:, 1:]
    
    X_estim_inv = np.linalg.inv(X_estim_with_intercept.T @ X_estim_with_intercept)
    
    d = np.diag(X_combined @ X_estim_inv @ X_combined.T)
    
    beta = X_estim_inv @ X_estim_with_intercept.T @ y_estim
    
    y_estim_pred = X_estim_with_intercept @ beta
    y_event_pred = X_event_with_intercept @ beta
    
    ARs_estim = y_estim - y_estim_pred
    ARs_event = y_event - y_event_pred

    return ARs_estim, ARs_event, d[:X_estim.shape[0]], d[X_estim.shape[0]:]

# With numba use automatic parallelisation for the computation of the regression !
@nb.njit(parallel=True, forceinline=True, looplift=True, inline='always', no_cfunc_wrapper=True, no_rewrites=True ,nogil=True, cache=True)
def make_ptf_reg(output_estim_residuals: np.ndarray, output_event_residuals: np.ndarray, 
                 output_estim_d: np.ndarray, output_event_d: np.ndarray, 
                 ret_array_c_valid: np.ndarray, ptf_in_valid_index: np.ndarray, 
                 associated_cluster_returns: np.ndarray, other_factors: np.ndarray, 
                 estim_period: int, event_period: int, use_cluster: bool) -> None:
    for idx in nb.prange(len(ptf_in_valid_index)):
        if use_cluster:
            X = np.concatenate((associated_cluster_returns[idx][:,np.newaxis], other_factors), axis=1)
        else:
            X = other_factors
        X_train, X_test, y_train, y_test = X[:estim_period], X[estim_period:], ret_array_c_valid[ptf_in_valid_index[idx], :estim_period], ret_array_c_valid[ptf_in_valid_index[idx], estim_period:]
        ars_estim, ars_event, d_estim, d_event = reg(X_train, y_train, X_test, y_test)
        output_estim_residuals[:,idx] = ars_estim
        output_event_residuals[:,idx] = ars_event
        output_estim_d[:,idx] = d_estim
        output_event_d[:,idx] = d_event

@nb.njit(parallel=True, forceinline=True, looplift=True, inline='always', no_cfunc_wrapper=True, no_rewrites=True ,nogil=True, cache=True)
def make_all_reg(output_estim_residuals: np.ndarray, output_event_residuals: np.ndarray, 
                 output_estim_d: np.ndarray, output_event_d: np.ndarray, 
                 ret_array_c_valid: np.ndarray, stocks_cluster_label: np.ndarray, 
                 ptf_stocks_label: np.ndarray, ptf_in_valid_index: np.ndarray, 
                 fama_french_factors: np.ndarray, vwretd_arr: np.ndarray, 
                 estim_period: int, event_period: int) -> None:
    n_stock = len(ptf_in_valid_index)
    associated_cluster_returns = np.zeros((n_stock,ret_array_c_valid.shape[1]))

    for idx in nb.prange(n_stock):
        mask = (stocks_cluster_label == ptf_stocks_label[idx]) & (np.arange(ret_array_c_valid.shape[0]) != ptf_in_valid_index[idx])
        associated_cluster_returns[idx] = np.sum(ret_array_c_valid[mask], axis=0) / np.sum(mask)

    for reg_num in nb.prange(7):
        if reg_num==0:
            other_factor = np.zeros((vwretd_arr.shape[0], 0)) # fake it for working similarly
            use_cluster = True
        elif reg_num==1:
            other_factor = vwretd_arr
            use_cluster = True
        elif reg_num==2:
            other_factor = np.concatenate((vwretd_arr, fama_french_factors[:,:2]), axis=1)
            use_cluster = True
        elif reg_num==3:
            other_factor = np.concatenate((vwretd_arr, fama_french_factors), axis=1)
            use_cluster = True
        elif reg_num==4:
            other_factor = vwretd_arr
            use_cluster = False
        elif reg_num==5:
            other_factor = np.concatenate((vwretd_arr, fama_french_factors[:,:2]), axis=1)
            use_cluster = False
        elif reg_num==6:
            other_factor = np.concatenate((vwretd_arr, fama_french_factors), axis=1)
            use_cluster = False
        
        make_ptf_reg(output_estim_residuals[reg_num], output_event_residuals[reg_num], output_estim_d[reg_num], output_event_d[reg_num], ret_array_c_valid, ptf_in_valid_index, associated_cluster_returns, other_factor, estim_period, event_period, use_cluster)


def process_single_stock(idx: int, ret_array_c_valid: np.ndarray, stocks_cluster_label: np.ndarray, 
                         ptf_stocks_label: np.ndarray, ptf_in_valid_index: np.ndarray, 
                         estim_period: int, event_period: int) -> Optional[List[Tuple[np.ndarray, np.ndarray]]]:
    mask = (stocks_cluster_label == ptf_stocks_label[idx]) & (np.arange(ret_array_c_valid.shape[0]) != ptf_in_valid_index[idx])
    cluster_returns = ret_array_c_valid[mask]

    X_train = cluster_returns[:, :estim_period].T
    X_test = cluster_returns[:, estim_period:].T
    y_train = ret_array_c_valid[ptf_in_valid_index[idx], :estim_period]
    y_test = ret_array_c_valid[ptf_in_valid_index[idx], estim_period:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Calculate default alpha values
    n_samples, n_features = X_train.shape
    ridge_alpha = 1.0
    lasso_alpha = 1e-3
    elastic_alpha = 1e-3

    models = [
        Ridge(alpha=ridge_alpha),
        Lasso(alpha=lasso_alpha, max_iter=10000, tol=1e-4),
        ElasticNet(alpha=elastic_alpha, l1_ratio=0.5, max_iter=10000, tol=1e-4)
    ]

    results = []
    for model in models:
        model.fit(X_train_scaled, y_train)
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        estim_residuals = y_train - y_train_pred
        event_residuals = y_test - y_test_pred
        results.append((estim_residuals, event_residuals))
    
    return results

def make_all_ml_models(output_estim_residuals: np.ndarray, output_event_residuals: np.ndarray, 
                       ret_array_c_valid: np.ndarray, stocks_cluster_label: np.ndarray, 
                       ptf_stocks_label: np.ndarray, ptf_in_valid_index: np.ndarray, 
                       estim_period: int, event_period: int) -> Tuple[np.ndarray, np.ndarray]:

    n_stocks = len(ptf_in_valid_index)
    n_models = 3  # RidgeCV, LassoCV, ElasticNetCV

    results = Parallel(n_jobs=-1)(
        delayed(process_single_stock)(
            idx, ret_array_c_valid, stocks_cluster_label, 
            ptf_stocks_label, ptf_in_valid_index, estim_period, event_period
        ) for idx in range(n_stocks)
    )

    for idx, stock_results in enumerate(results):
        if stock_results is None:
            continue
        for model_idx, (estim_res, event_res) in enumerate(stock_results):
            output_estim_residuals[model_idx, :, idx] = estim_res
            output_event_residuals[model_idx, :, idx] = event_res

    return output_estim_residuals, output_event_residuals


def fit_kmeans(n_clusters: int, returns: np.ndarray) -> KMeans:
    """
    Fit KMeans clustering to returns data.

    Args:
        n_clusters (int): Number of clusters to use.
        returns (np.ndarray): Array of returns to cluster.

    Returns:
        KMeans: Fitted KMeans model.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(returns)
    return kmeans

def compute_residuals_of_portfolio_with_methods(start_date: int, ptf: np.ndarray, 
                                                ret_array_c: np.ndarray, valid_array_c: np.ndarray, 
                                                estim_period: int, ff_array: np.ndarray, 
                                                vwretd_arr: np.ndarray, event_period: int, 
                                                cluster_num_list: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute residuals for a portfolio using various methods including clustering.

    Args:
        start_date (int): Start date index.
        ptfs (np.ndarray): Portfolio array (indexes in columns not PERMNO).
        ret_array_c (np.ndarray): Returns array.
        valid_array_c (np.ndarray): Validity array.
        estim_period (int): Length of estimation period.
        fama_french_factors (np.ndarray): Fama-French factors.
        vwretd_arr (np.ndarray): Value-weighted market returns.
        event_period (int): Length of event period.
        cluster_num_list (List[int]): List of number of clusters to try.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Estimation residuals, event residuals, and corresponding diagnostics.
    """
    #Select all valid stocks over studying period
    n_stocks = len(ptf)
    ret_array_c_valid = ret_array_c[valid_array_c[:,start_date+event_period-1],start_date-estim_period:start_date+event_period]
    ptf_in_valid_index = (valid_array_c[:,start_date+event_period-1].cumsum() - 1)[ptf]
    kmeans_results = Parallel(n_jobs=-1)(
        delayed(fit_kmeans)(n_clusters, ret_array_c_valid[:, :estim_period])
        for n_clusters in cluster_num_list
    )

    # 7 is for cluster only, cluster + vwreted, cluster + FF3, cluster + FF5, Market Only, FF3 and FF5, + 3 is for ML models
    estim_residuals, event_residuals = np.zeros((len(cluster_num_list), 7 + 3, estim_period, n_stocks)), np.zeros((len(cluster_num_list), 7 + 3, event_period, n_stocks))
    estim_d, event_d = np.zeros(estim_residuals.shape), np.zeros(event_residuals.shape)
    
    for kmeans in kmeans_results:

        
        n_clusters = kmeans.n_clusters
        cluster_idx = [idx for idx, k in enumerate(cluster_num_list) if k==n_clusters][0]
        stocks_cluster_label = kmeans.predict(ret_array_c_valid[:, :estim_period])
        # Counts number of stocks per cluster and remove the cluster with only one stock if any
        values, counts = np.unique(stocks_cluster_label, return_counts=True)
        final_cluster_centers = kmeans.cluster_centers_[counts > 1]

        # Reaffect the stocks
        stocks_cluster_label = np.argmin(
            np.sum((final_cluster_centers[:, np.newaxis, :] - ret_array_c_valid[:, :estim_period]) ** 2, axis=(2,)),
            axis=0
        )
        ptf_stocks_label = stocks_cluster_label[ptf_in_valid_index]

        make_all_reg(
            estim_residuals[cluster_idx, :7],
            event_residuals[cluster_idx, :7],
            estim_d[cluster_idx, :7],
            event_d[cluster_idx, :7],
            ret_array_c_valid,
            stocks_cluster_label,
            ptf_stocks_label,
            ptf_in_valid_index,
            ff_array[start_date-estim_period:start_date+event_period],
            vwretd_arr[start_date-estim_period:start_date+event_period],
            estim_period,
            event_period
        )

        make_all_ml_models(
            estim_residuals[cluster_idx, 7:],
            event_residuals[cluster_idx, 7:],
            ret_array_c_valid, 
            stocks_cluster_label, ptf_stocks_label, ptf_in_valid_index, estim_period, event_period
        )

    return estim_residuals, event_residuals, estim_d, event_d