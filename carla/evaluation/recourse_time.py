from typing import List

import numpy as np
import pandas as pd
import timeit
from carla.recourse_methods.api import RecourseMethod


def recourse_time_taken(
    recourse_method: RecourseMethod, factuals: pd.DataFrame
) -> List[List[int]]:
    """
    Time taken per counterfactual
    Parameters
    ----------
    recourse_method: carla.recourse_methods.RecourseMethod
    factuals: Not normalized and encoded factuals
    Returns
    -------
    """
    
    times = []
    for i in range(factuals.shape[0]):
      start = timeit.default_timer()      
      df = pd.DataFrame(columns=factuals.columns)
      df.loc[0] = factuals.iloc[i]
      recourse_method.get_counterfactuals(df)
      stop = timeit.default_timer()
      times.append([stop - start])
      
    return times
