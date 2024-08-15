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
from py_event_studies.results import Results
from py_event_studies._utils import to_date_index_format
from py_event_studies._data_loader import compute_validity, read_data_file, create_pivot_dataframes
from py_event_studies._calibration import compute_residuals_of_portfolio_with_methods
from py_event_studies.results import Results

CACHE_KEYS = {
    'df_valid', 
    'df_primexch',
    'df_siccd',
    'df_ret',
    'df_prc',
    'vwretd',
    'primexch_mapping',
    'df_valid_stock',
    'ret_array_c',
    'vwretd_arr',
    'valid_array_c',
    'delta_estim_event_period',
    'estim_period',
    'event_period',
}

def update_config(**kwargs):
    need_reload, need_recompute_validity_mask = False, False
    for key, value in kwargs.items():
        if not hasattr(config, key):
            raise AttributeError(f"Config object has no attribute '{key}'")

        if key in ['estim_period', 'event_period', 'delta_estim_event_period'] and data_store.data_path is not None and getattr(config, key) != value:
            need_reload = True
        if key == 'event_period' and value % 2 != 1:
            raise ValueError("The event period must be an uneven number.")
    
        setattr(config, key, value)

    if need_reload:
        print('Updating the period length requires to reprocess partially the datas. If possible update config before loading the datas.')
        load_data(data_store.data_path)

def get_valid_dates() -> List[str]:
    return data_store.df_primexch.index.values

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
    data_to_cache = {key: getattr(data_store, key) if hasattr(data_store, key) else getattr(config, key) for key in CACHE_KEYS}

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

    if any(k not in cached_data for k in CACHE_KEYS) or any(getattr(config, k) != cached_data[k] for k in ['estim_period', 'event_period', 'delta_estim_event_period']):
        print('The config has changed or older version used. The data must be reprocessed.')
        _load_data_no_cache(path)
    
    print(f"Using cached data for {path}")
    with open(cache_path, 'rb') as cache_file:
        cached_data = pickle.load(cache_file)
    
    # Restore cached data to data_store
    for key, value in cached_data.items():
        setattr(data_store, key, value)

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
        event_date = date_idx,
        ptf = ptf_idxs,
        ret_array_c = data_store.ret_array_c, 
        valid_array_c = data_store.valid_array_c, 
        estim_period = config.estim_period, 
        ff_array = data_store.ff_array, 
        vwretd_arr = data_store.vwretd_arr, 
        event_period = config.event_period, 
        delta_estim_event_period = config.delta_estim_event_period,
        cluster_num_list = config.cluster_num_list
    )

    estim_stock_returns = data_store.ret_array_c[ptf_idxs, date_idx - config.estim_period:date_idx].T
    event_stock_returns = data_store.ret_array_c[ptf_idxs, date_idx:date_idx + config.event_period].T

    estim_stock_returns = np.broadcast_to(np.expand_dims(estim_stock_returns, axis = (0,1)), estim_residuals.shape)
    event_stock_returns = np.broadcast_to(np.expand_dims(event_stock_returns, axis = (0,1)), event_residuals.shape)

    event_idx = int(config.estim_period/2) + 1

    results = Results(
        date=date,
        ptf=ptf,
        estim_stock_returns=estim_stock_returns,
        event_stock_returns=event_stock_returns,
        estim_residuals=estim_residuals,
        event_residuals=event_residuals, 
        event_d=event_d,
    )

    return results