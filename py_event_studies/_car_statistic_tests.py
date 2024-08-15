import numpy as np

def standard_test(event_residual: np.ndarray, estim_residuals: np.ndarray) -> np.ndarray:
    """
    Compute the standard t-test for cumulative abnormal returns.

    Args:
        event_residual (np.ndarray): Abnormal returns over the event period.
            Shape: (n_clusters, n_methods, event_period_length, n_stocks)
        estim_residuals (np.ndarray): Residuals from the estimation period.
            Shape: (n_clusters, n_methods, estimation_period_length, n_stocks)

    Returns:
        np.ndarray: T-statistics for each cluster and method.
            Shape: (n_clusters, n_methods)
    """
    car = np.sum(event_residual, axis=2)  # Cumulative abnormal returns
    car_mean = np.mean(car, axis=-1)
    estimation_var = np.var(estim_residuals, axis=(2, 3))
    t = event_residual.shape[2]  # Event period length
    return car_mean / np.sqrt(t * estimation_var / estim_residuals.shape[-1])

def ordinary_cross_sec_test(event_residual: np.ndarray) -> np.ndarray:
    """
    Perform a cross-sectional t-test on cumulative abnormal returns.

    Args:
        event_residual (np.ndarray): Abnormal returns over the event period.
            Shape: (n_clusters, n_methods, event_period_length, n_stocks)

    Returns:
        np.ndarray: Cross-sectional t-statistics for each cluster and method.
            Shape: (n_clusters, n_methods)
    """
    car = np.sum(event_residual, axis=2)  # Cumulative abnormal returns
    car_mean = np.mean(car, axis=-1)
    car_var = np.var(car, axis=-1)
    return car_mean / np.sqrt(car_var / car.shape[-1])

def bmp_test(event_residual: np.ndarray, estim_residuals: np.ndarray, event_d: np.ndarray) -> np.ndarray:
    """
    Implement the modified Boehmer, Masumeci, and Poulsen (1991) test for cumulative returns.

    Args:
        event_residual (np.ndarray): Abnormal returns over the event period.
            Shape: (n_clusters, n_methods, event_period_length, n_stocks)
        estim_residuals (np.ndarray): Residuals from the estimation period.
            Shape: (n_clusters, n_methods, estimation_period_length, n_stocks)
        event_d (np.ndarray): Adjustment factor for the event window.
            Shape: (n_clusters, n_methods, event_period_length, n_stocks)

    Returns:
        np.ndarray: BMP test statistics for each cluster and method.
            Shape: (n_clusters, n_methods)
    """
    nb_expl_var_map = np.array([1, 2, 4, 6, 1, 3, 5, 1, 1, 1]) + 1
    
    T1 = estim_residuals.shape[2]
    t = event_residual.shape[2]  # Event period length
    sigma_raw = np.std(estim_residuals, axis=2)
    
    # Adjust nb_expl_var_map to match the number of methods
    nb_expl_var = nb_expl_var_map[:sigma_raw.shape[1]]
    nb_expl_var = nb_expl_var[np.newaxis, :, np.newaxis]
    
    # Calculate sigma_adj for each time point in the event window
    sigma_adj = sigma_raw[:, :, np.newaxis, :] * np.sqrt(T1 - 1) / (np.sqrt(T1 - nb_expl_var) * np.sqrt(1 + event_d))
    
    # Calculate standardized abnormal returns for each time point
    standardized_ar = event_residual / sigma_adj
    
    # Calculate cumulative standardized abnormal returns
    car_standardized = np.sum(standardized_ar, axis=2)
    
    car_standardized_mean = np.mean(car_standardized, axis=-1)
    car_standardized_var = np.var(car_standardized, axis=-1)
    return car_standardized_mean / np.sqrt(car_standardized_var / car_standardized.shape[-1])

def create_avg_corrs(estim_residuals: np.ndarray) -> np.ndarray:
    """
    Calculate average correlations between residuals.

    Args:
        estim_residuals (np.ndarray): Residuals from the estimation period.
            Shape: (n_clusters, n_methods, estimation_period_length, n_stocks)

    Returns:
        np.ndarray: Average correlations for each cluster and method.
            Shape: (n_clusters, n_methods)
    """
    # This function remains unchanged as it only uses estimation period residuals
    avg_cors = np.zeros(estim_residuals.shape[:2])
    for j in range(estim_residuals.shape[0]):
        for k in range(estim_residuals.shape[1]):
            residuals = estim_residuals[j, k]
            rho_mat = np.corrcoef(residuals.T)
            mask = ~np.eye(rho_mat.shape[0], dtype=bool)
            avg_cors[j, k] = np.sum(rho_mat[mask]) / np.sum(mask)
    return avg_cors

def kp_test(event_residual: np.ndarray, estim_residuals: np.ndarray, avg_cors: np.ndarray) -> np.ndarray:
    """
    Implement the modified Kolari and Pynnönen (2010) test for cumulative returns.

    Args:
        event_residual (np.ndarray): Abnormal returns over the event period.
            Shape: (n_clusters, n_methods, event_period_length, n_stocks)
        estim_residuals (np.ndarray): Residuals from the estimation period.
            Shape: (n_clusters, n_methods, estimation_period_length, n_stocks)
        avg_cors (np.ndarray): Average correlations for each cluster and method.
            Shape: (n_clusters, n_methods)

    Returns:
        np.ndarray: KP test statistics for each cluster and method.
            Shape: (n_clusters, n_methods)
    """
    nb_expl_var_map = (np.array([1, 2, 4, 6, 1, 3, 5, 1, 1, 1]) + 1).reshape(1,-1,1)

    T1 = estim_residuals.shape[2]
    t = event_residual.shape[2]  # Event period length
    sigma_raw = np.sqrt(np.sum(np.square(estim_residuals - np.mean(estim_residuals, axis=2, keepdims=True)), axis=2) / (T1 - 1))
    nb_expl_var = np.broadcast_to(nb_expl_var_map, sigma_raw.shape)
    sigma_adj = sigma_raw * np.sqrt(T1 - 1) / np.sqrt(T1 - nb_expl_var)
    car = np.sum(event_residual, axis=2)  # Cumulative abnormal returns
    car_standardized = car / (np.sqrt(t) * sigma_adj)
    car_standardized_mean = np.mean(car_standardized, axis=-1)
    car_standardized_var = np.var(car_standardized, axis=-1)
    n = car_standardized.shape[-1]
    t_test = car_standardized_mean / np.sqrt(car_standardized_var) * np.sqrt(n) * \
             np.sqrt((1 - avg_cors) / (1 + (n - 1) * avg_cors))
    return t_test