from typing import List, Optional, Dict
from numpy import ndarray
from pandas import DataFrame, Series


class Config:
    cluster_num_list: List[int] = [5, 10, 15, 20, 25, 30, 35, 40, 50]
    event_period: int = 10 # Symetric around event period, needs to be a uneven number
    estim_period: int = 249 # Size of period where we estimate the models
    delta_estim_event_period: int = 3 # Delta between the end of the estimation period and the event period

    @property
    def min_obs_per_permno(self):
        return self.event_period + self.estim_period + self.delta_estim_event_period

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
    df_valid: Optional[DataFrame] = None
    ret_array_c: Optional[ndarray] = None
    vwretd_arr: Optional[ndarray] = None
    valid_array_c: Optional[ndarray] = None
    ff_array: Optional[ndarray] = None
    ff_factors: Optional[DataFrame] = None

data_store = DataStorage()
config = Config()