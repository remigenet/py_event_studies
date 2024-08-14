from typing import Dict
import numpy as np
import pandas as pd
from py_event_studies import data_store, config

def read_data_file(path: str) -> None:
    """
    Read and preprocess CRSP data from a CSV or Parquet file.

    Args:
        path (str): Path to the data file.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.

    Raises:
        ValueError: If the file format is not supported.
    """
    config.data_path = path
    if path.endswith('.parquet'):
        df = pd.read_parquet(path)
    elif path.endswith('.csv'):
        df = pd.read_csv(path)
    else:
        raise ValueError("Unsupported file format. Use .csv or .parquet files.")

    df_OK = df[(df['RET'].isna()==False) & (df['PRC']>5)]
    byPERMNO = df_OK.groupby('PERMNO')
    nobsByPermno = pd.DataFrame(byPERMNO['PERMNO'].count())
    nobsByPermno.columns = ['nobs']
    nobsByPermno = nobsByPermno.reset_index()
    nobsByPermno_valid = nobsByPermno[nobsByPermno['nobs']>=config.min_obs_per_permno]
    df_valid = pd.merge(nobsByPermno_valid, df) 

    # Map PRIMEXCH to integer codes
    df_valid['PRIMEXCH_code'], _ = pd.factorize(df_valid['PRIMEXCH'])
    df_valid['PRIMEXCH_code'] = df_valid['PRIMEXCH_code'].astype('uint8')

    data_store.df_valid = df_valid
    

def create_pivot_dataframes() -> None:
    """
    Pivot the preprocessed DataFrame into several dataframe with date index and permno columns.
    Goal is to have information more contiguous in memory after.

    Args:
        df_valid (pd.DataFrame): Preprocessed DataFrame.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, Dict[int, str]]:
            Pivoted DataFrames for PRIMEXCH, SICCD, RET, PRC, VWRETD, and PRIMEXCH mapping.
    """
    # Create a dictionary to store the mapping of codes to exchange names
    data_store.primexch_mapping: Dict[int, str] = dict(zip(data_store.df_valid['PRIMEXCH_code'], data_store.df_valid['PRIMEXCH']))

    # Pivot the DataFrames
    data_store.df_primexch = data_store.df_valid.pivot(index='date', columns='PERMNO', values='PRIMEXCH_code')
    data_store.df_primexch = data_store.df_primexch.fillna(255).astype(np.uint8)
    data_store.df_siccd = data_store.df_valid.pivot(index='date', columns='PERMNO', values='SICCD')
    data_store.df_ret = data_store.df_valid.pivot(index='date', columns='PERMNO', values='RET')
    data_store.df_prc = data_store.df_valid.pivot(index='date', columns='PERMNO', values='PRC')
    data_store.vwretd = data_store.df_valid.groupby('date')['vwretd'].last().sort_index()
    data_store.ret_array_c = data_store.df_ret.values.T.copy(order='C')
    data_store.vwretd_arr = data_store.vwretd.values.reshape(-1,1)

def compute_validity() -> None:
    """
    Compute the validity mask for stocks based on various criteria.

    Args:
        df_primexch (pd.DataFrame): Pivoted PRIMEXCH DataFrame.
        df_siccd (pd.DataFrame): Pivoted SICCD DataFrame.
        df_ret (pd.DataFrame): Pivoted RET DataFrame.

    Returns:
        pd.DataFrame: Validity mask for stocks.
    """
    total_period = config.estim_period + config.event_period + config.delta_estim_event_period
    valid_return_mask = ~data_store.df_ret.rolling(total_period).max().isna()
    valid_primex_mask = data_store.df_primexch.rolling(total_period).max() == data_store.df_primexch.rolling(total_period).min()
    valid_sccid_mask = data_store.df_siccd.rolling(total_period).max() == data_store.df_siccd.rolling(total_period).min()
    data_store.df_valid_stock = valid_primex_mask & valid_sccid_mask & valid_return_mask
    data_store.valid_array_c = data_store.df_valid_stock.values.T.copy(order='F')
