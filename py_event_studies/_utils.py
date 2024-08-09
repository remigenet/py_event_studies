from typing import Any
from datetime import datetime
import numpy as np
import pandas as pd
from functools import cache

@cache
def to_date_index_format(value: Any) -> np.int32:
    if isinstance(value, (pd.Timestamp, datetime)):
        return np.int32(value.strftime('%Y%m%d'))
    elif isinstance(value, str):
        return np.int32(pd.Timestamp(value).strftime('%Y%m%d'))
    elif isinstance(value, (int, np.int32, np.int64)):  
        return np.int32(value)