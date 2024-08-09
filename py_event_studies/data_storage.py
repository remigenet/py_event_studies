from typing import List, Optional, Dict
from numpy import ndarray
from pandas import DataFrame, Series


class Config:
    cluster_num_list: List[int] = [5, 10, 15, 20, 25, 30, 35, 40, 50]
    event_period: int = 10
    estim_period: int = 249
    min_obs_per_permno = 259 # Change this only if you need to use a permno with very few datas, it is used to filter initial datas. Changing it implies reloading all datas.

class DataStorage:
    data_path: Optional[str] = None
    ff_factors_path: Optional[str] = None
    df_primexch: Optional[DataFrame] = None
    df_siccd: Optional[DataFrame] = None
    df_ret: Optional[DataFrame] = None
    df_prc: Optional[DataFrame] = None
    vwretd: Optional[Series] = None
    primexch_mapping: Optional[Dict[int, str]] = None
    df_valid_stock: Optional[DataFrame] = None
    ret_array_c: Optional[ndarray] = None
    vwretd_arr: Optional[ndarray] = None
    valid_array_c: Optional[ndarray] = None
    ff_array: Optional[ndarray] = None
    ff_factors: Optional[DataFrame] = None

data_store = DataStorage()
config = Config()