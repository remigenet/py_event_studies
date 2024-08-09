import os
import pickle
from typing import Union, Iterable, List, Dict, Any
import hashlib
from functools import cached_property, cache
import importlib.util

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from py_event_studies import data_store, config
from py_event_studies._utils import to_date_index_format
from py_event_studies._data_loader import compute_validity, read_data_file, create_pivot_dataframes
from py_event_studies._calibration import compute_residuals_of_portfolio_with_methods
from py_event_studies._statistic_tests import standard_test, bmp_test, ordinary_cross_sec_test, create_avg_corrs, kp_test


def update_config(**kwargs):
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise AttributeError(f"Config object has no attribute '{key}'")
        if key in ['estim_period', 'event_period'] and data_store.df_valid is not None:
            print('Updating the period length requires to reprocess partially the datas. If possible update config before loading the datas.')
            compute_validity()
        elif key == 'min_obs_per_permno':
            load_data(data_store.data_path)


def get_valid_permno_at_date(date: Any) -> List[int]:
    date = to_date_index_format(date)
    return data_store.df_valid_stock.loc[date][data_store.df_valid_stock.loc[date]].index.values

def _load_data_no_cache(path: str) -> None:
    print(f"Loading and preprocessing data from {path}")
    data_store.data_path = path
    cache_path = generate_cache_path(path)
    read_data_file(path)
    create_pivot_dataframes()
    compute_validity()

    # Cache the processed data
    data_to_cache = {
        'df_valid': data_store.df_valid,
        'df_primexch': data_store.df_primexch,
        'df_siccd': data_store.df_siccd,
        'df_ret': data_store.df_ret,
        'df_prc': data_store.df_prc,
        'vwretd': data_store.vwretd,
        'primexch_mapping': data_store.primexch_mapping,
        'df_valid_stock': data_store.df_valid_stock,
        'ret_array_c': data_store.ret_array_c,
        'vwretd_arr': data_store.vwretd_arr,
        'valid_array_c': data_store.valid_array_c,
        'min_obs_per_permno': config.min_obs_per_permno,
        'estim_period': config.estim_period,
        'event_period': config.event_period,
    }

    with open(cache_path, 'wb') as cache_file:
        pickle.dump(data_to_cache, cache_file)
    
    print(f"Cached preprocessed data for {path}")

@cache
def generate_cache_path(path: str) -> str:
    # Create a cache directory in the package directory if it doesn't exist
    cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
    os.makedirs(cache_dir, exist_ok=True)

    # Create a unique cache file name based on the input file path
    file_hash = hashlib.md5(path.encode()).hexdigest()
    cache_file_name = f"{file_hash}.cache"
    return os.path.join(cache_dir, cache_file_name)

def load_data(path: str, no_cache: bool = False) -> None:
    """
    Main function to load and process CRSP data with caching support.

    Args:
        path (str): Path to the data file.
        no_cache (bool): If True, ignore cached data and reload from scratch. Defaults to False.

    Raises:
        ValueError: If the file format is not supported.
    """
    cache_path = generate_cache_path(path)
    
    if no_cache or not os.path.exists(cache_path):
        _load_data_no_cache(path)
    with open(cache_path, 'rb') as cache_file:
        cached_data = pickle.load(cache_file)

    if any(k not in cached_data for k in ['min_obs_per_permno', 'estim_period', 'event_period']) or cached_data['min_obs_per_permno'] != config.min_obs_per_permno:
        print('The min_obs_per_permno parameter has changed. The data must be reprocessed.')
        _load_data_no_cache(path)
    
    print(f"Using cached data for {path}")
    with open(cache_path, 'rb') as cache_file:
        cached_data = pickle.load(cache_file)
    
    # Restore cached data to data_store
    for key, value in cached_data.items():
        setattr(data_store, key, value)

    if config.estim_period != cached_data['estim_period'] or config.event_period != cached_data['event_period']:
        print('The estimation or event period has changed. The data must partially reprocessed.')
        compute_validity()    

# Function to clear the cache
def clear_cache():
    """
    Clear all cached data files.
    """
    cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
    if os.path.exists(cache_dir):
        for cache_file in os.listdir(cache_dir):
            os.remove(os.path.join(cache_dir, cache_file))
        print("Cache cleared successfully.")
    else:
        print("No cache directory found.")


def load_ff_factors(path: str) -> None:
    """
    Load Fama-French factors from a CSV or Parquet file.

    Args:
        path (str): Path to the Fama-French factors file.

    Raises:
        ValueError: If the file format is not supported.
    """
    data_store.ff_factors_path = path
    if path.endswith('.parquet'):
        ff_factors = pd.read_parquet(path)
    elif path.endswith('.csv'):
        ff_factors = pd.read_csv(path)
    else:
        raise ValueError("Unsupported file format. Use .csv or .parquet files.")

    if 'date' in ff_factors.columns:
        ff_factors = ff_factors.set_index('date')

    data_store.ff_factors = ff_factors / 100.0
    data_store.ff_array = data_store.ff_factors.values

class Results:
    def __init__(self, date: str, ptf: Union[int, List[int]], test_results: Dict[str, np.ndarray],
                 estim_stock_returns: np.ndarray, event_stock_returns: np.ndarray,
                 estim_residuals: np.ndarray, event_residuals: np.ndarray):
        self.date = date
        self.ptf = [ptf] if isinstance(ptf, int) else ptf
        self.test_results = test_results
        self.estim_stock_returns = estim_stock_returns
        self.event_stock_returns = event_stock_returns
        self.estim_residuals = estim_residuals
        self.event_residuals = event_residuals
        self.n_stocks = len(self.ptf)
        
        self.cluster_num = config.cluster_num_list
        self.model_names = ['Cluster only', 'Cluster + Market', 'Cluster + FF3', 'Cluster + FF5',
                            'Market Model', 'FF3', 'FF5', 'RidgeCV', 'LassoCV', 'ElasticNetCV']
        self.test_names = ['std', 'CS', 'BMP', 'KP']
        self.models_degree_of_freedom = np.array([1, 2, 4, 6, 1, 3, 5, 1, 1, 1]) + 1
        self.n_stocks = len(self.ptf)
        

    @cached_property
    def estim_preds(self):
        return np.broadcast_to(np.expand_dims(self.estim_stock_returns, axis = (0,1)), self.estim_residuals.shape) + self.estim_residuals

    @cached_property
    def event_preds(self):
        return np.broadcast_to(np.expand_dims(self.event_stock_returns, axis = (0,1)), self.event_residuals.shape) + self.event_residuals

    @cached_property
    def std_test_stats(self):
        return pd.DataFrame(self.test_results['std_test'], index=self.cluster_num, columns=self.model_names)

    @cached_property
    def cs_test_stats(self):
        return pd.DataFrame(self.test_results['cs_test'], index=self.cluster_num, columns=self.model_names)

    @cached_property
    def bmp_test_stats(self):
        return pd.DataFrame(self.test_results['bmp_test'], index=self.cluster_num, columns=self.model_names)

    @cached_property
    def kp_test_stats(self):
        return pd.DataFrame(self.test_results['kp_test'], index=self.cluster_num, columns=self.model_names)

    @staticmethod
    def _t_distribution_cdf(x, df):
        """
        Approximation of the CDF of the t-distribution.
        This is a simple approximation and might not be as accurate as scipy's implementation for all cases.
        """
        return 0.5 + 0.5 * np.sign(x) * (1 - np.exp(-0.5 * x**2 * (df + 1) / (x**2 + df)))

    def _calculate_p_values(self, test_stats: pd.DataFrame) -> pd.DataFrame:
        """Calculate p-values for given test statistics."""
        df = self.n_stocks - self.models_degree_of_freedom
        p_values = 2 * (1 - self._t_distribution_cdf(np.abs(test_stats), df=df[np.newaxis, :]))
        return pd.DataFrame(p_values, index=self.cluster_num, columns=self.model_names)

    @cached_property
    def std_p_values(self):
        return self._calculate_p_values(self.std_test_stats)

    @cached_property
    def cs_p_values(self):
        return self._calculate_p_values(self.cs_test_stats)

    @cached_property
    def bmp_p_values(self):
        return self._calculate_p_values(self.bmp_test_stats)

    @cached_property
    def kp_p_values(self):
        return self._calculate_p_values(self.kp_test_stats)

    def get_test_result(self, test_name: str) -> pd.DataFrame:
        """Get the results for a specific test as a DataFrame."""
        if test_name not in self.test_names:
            raise ValueError(f"Invalid test name. Choose from {self.test_names}")
        
        return getattr(self, f"{test_name.lower()}_test_stats")

    def get_p_values(self, test_name: str) -> pd.DataFrame:
        """Get p-values for a specific test."""
        if test_name not in self.test_names:
            raise ValueError(f"Invalid test name. Choose from {self.test_names}")
        
        return getattr(self, f"{test_name.lower()}_p_values")

    def summary(self) -> None:
        """Print a summary of the results."""
        print(f"Event Date: {self.date}")
        print(f"Portfolio: {self.ptf}")
        print(f"Number of stocks: {self.n_stocks}")
        print(f"Estimation period: {self.estim_stock_returns.shape[0]} days")
        print(f"Event period: {self.event_stock_returns.shape[0]} days")
        print("\nTest Results:")
        for test in self.test_names:
            print(f"\n{test} Test:")
            print(self.get_test_result(test))
            print(f"\n{test} Test P-values:")
            print(self.get_p_values(test))

    def __str__(self):
        return f"Event Study Results\n" \
               f"Event Date: {self.date}\n" \
               f"Portfolio: {self.ptf[:5]}{'...' if len(self.ptf) > 5 else ''}\n" \
               f"Number of stocks: {self.n_stocks}\n" \
               f"Estimation period: {self.estim_stock_returns.shape[0]} days\n" \
               f"Event period: {self.event_stock_returns.shape[0]} days\n" \
               f"Number of cluster configurations: {len(self.cluster_num)}\n" \
               f"Number of models: {len(self.model_names)}\n" \
               f"Available tests: {', '.join(self.test_names)}"

    @staticmethod
    def help():
        print("Available methods and properties:")
        print("- estim_preds: Predicted returns for estimation period")
        print("- event_preds: Predicted returns for event period")
        print("- std_test_stats, cs_test_stats, bmp_test_stats, kp_test_stats: Test statistics")
        print("- std_p_values, cs_p_values, bmp_p_values, kp_p_values: P-values for tests")
        print("- get_test_result(test_name): Get test statistics for a specific test")
        print("- get_p_values(test_name): Get p-values for a specific test")
        print("- summary(): Print a detailed summary of results")
        print("- plot(cluster_idx, model_idx): Plot true vs predicted returns for a specific model and cluster configuration")
        print("- to_excel(filename): Export results to an Excel file")

    def plot(self, cluster_num: int, model_name: str):
        """
        Plot true vs predicted returns for a specific model and cluster configuration.
        
        Args:
        cluster_num (int): Number of clusters to plot
        model_name (str): Name of the model to plot
        """
        if cluster_num not in self.cluster_num:
            raise ValueError(f"cluster_num must be one of {self.cluster_num}")
        if model_name not in self.model_names:
            raise ValueError(f"model_name must be one of {self.model_names}")

        cluster_idx = self.cluster_num.index(cluster_num)
        model_idx = self.model_names.index(model_name)

        estim_true = self.estim_stock_returns
        estim_pred = self.estim_preds[cluster_idx, model_idx]
        event_true = self.event_stock_returns
        event_pred = self.event_preds[cluster_idx, model_idx]

        n_stocks = self.n_stocks
        n_rows = (n_stocks + 3) // 4  # Round up to nearest multiple of 4

        fig, axs = plt.subplots(n_rows, 4, figsize=(20, 5*n_rows))
        fig.suptitle(f"True vs Predicted Returns - Cluster: {cluster_num}, Model: {model_name}")

        for i in range(n_stocks):
            row = i // 4
            col = i % 4
            ax = axs[row, col] if n_rows > 1 else axs[col]

            # Plot estimation period
            ax.plot(range(len(estim_true)), estim_true[:, i], label='True (Estimation)', color='blue', alpha=0.7)
            ax.plot(range(len(estim_pred)), estim_pred[:, i], label='Predicted (Estimation)', color='red', alpha=0.7)

            # Plot event period
            offset = len(estim_true)
            ax.plot(range(offset, offset + len(event_true)), event_true[:, i], label='True (Event)', color='green', alpha=0.7)
            ax.plot(range(offset, offset + len(event_pred)), event_pred[:, i], label='Predicted (Event)', color='orange', alpha=0.7)

            ax.set_title(f"Stock {self.ptf[i]}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Returns")
            ax.legend(loc='upper left', fontsize='x-small')

        # Remove any unused subplots
        for i in range(n_stocks, n_rows * 4):
            row = i // 4
            col = i % 4
            fig.delaxes(axs[row, col] if n_rows > 1 else axs[col])

        plt.tight_layout()
        plt.show()

    def to_excel(self, filename: str):
        if importlib.util.find_spec("openpyxl") is None:
            raise ImportError("openpyxl is required for Excel export. "
                              "Install it with 'pip install openpyxl' or "
                              "'poetry install --with excel'")

        with pd.ExcelWriter(filename) as writer:
            for test in self.test_names:
                self.get_test_result(test).to_excel(writer, sheet_name=f"{test} Test")
                self.get_p_values(test).to_excel(writer, sheet_name=f"{test} P-values")
            
            pd.DataFrame(self.estim_stock_returns, columns=self.ptf).to_excel(writer, sheet_name="Estimation Returns")
            pd.DataFrame(self.event_stock_returns, columns=self.ptf).to_excel(writer, sheet_name="Event Returns")
            
            for i, cluster_num in enumerate(self.cluster_num):
                for j, model in enumerate(self.model_names):
                    sheet_name = f"Estim Residuals {cluster_num}_{model}"
                    pd.DataFrame(self.estim_residuals[i, j], columns=self.ptf).to_excel(writer, sheet_name=sheet_name)
                    
                    sheet_name = f"Event Residuals {cluster_num}_{model}"
                    pd.DataFrame(self.event_residuals[i, j], columns=self.ptf).to_excel(writer, sheet_name=sheet_name)

        

def compute(date: str, ptf: Union[int, Iterable[int]]):
    date_idx = data_store.df_primexch.index.get_loc(to_date_index_format(date))
    valid_permnos = get_valid_permno_at_date(date)
    ptf = [ptf] if isinstance(ptf, int) else ptf
    if any(p not in valid_permnos for p in ptf):
        raise ValueError("One or more stocks in the portfolio are not valid at the given date.\n Use function get_valid_permno_at_date(date) to get the list of valid permnos.")
    ptf_idxs = [data_store.df_primexch.columns.get_loc(p) for p in ptf]
    

    (estim_residuals,
     event_residuals,
     estim_d,
     event_d
    ) = compute_residuals_of_portfolio_with_methods(
        start_date = date_idx,
        ptf = ptf_idxs,
        ret_array_c = data_store.ret_array_c, 
        valid_array_c = data_store.valid_array_c, 
        estim_period = config.estim_period, 
        ff_array = data_store.ff_array, 
        vwretd_arr = data_store.vwretd_arr, 
        event_period = config.event_period, 
        cluster_num_list = config.cluster_num_list
    )

    estim_stock_returns = data_store.ret_array_c[ptf_idxs, date_idx - config.estim_period:date_idx].T
    event_stock_returns = data_store.ret_array_c[ptf_idxs, date_idx:date_idx + config.event_period].T

    estim_residuals = np.expand_dims(estim_residuals, axis=0)
    event_residuals = np.expand_dims(event_residuals, axis=0)
    estim_d = np.expand_dims(estim_d, axis=0)
    event_d = np.expand_dims(event_d, axis=0)

    #0 is for first date of event date - TO DO implement for cumulative event window
    std_test_result = standard_test(event_residuals[:, :, : , 0, :], estim_residuals) 
    bmp_test_result = bmp_test(event_residuals[:, :, : , 0, :], estim_residuals, event_d[:, :, : , 0, :])
    
    ocs_test_result = ordinary_cross_sec_test(event_residuals[:, :, : , 0, :])

    avg_cors = create_avg_corrs(estim_residuals)
    kp_test_results = kp_test(event_residuals[:, :, : , 0, :], estim_residuals, avg_cors)

    results = Results(
        date=date,
        ptf=ptf,
        test_results={
            'std_test': std_test_result[0],
            'bmp_test': bmp_test_result[0],
            'kp_test': kp_test_results[0],
            'cs_test': ocs_test_result[0]
        },
        estim_stock_returns=estim_stock_returns,
        event_stock_returns=event_stock_returns,
        estim_residuals=estim_residuals[0],
        event_residuals=event_residuals[0]
    )

    return results