from typing import List

import numpy as np
import pandas as pd

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
    self._counterfactuals = recourse_method.get_counterfactuals(factuals)
    stop = timeit.default_timer()
    self._timer = stop - start

    times = []
    for fact in factuals:
      start = timeit.default_timer()      
      recourse_method.get_counterfactuals(fact)
      stop = timeit.default_timer()
      times.append([stop - start])
      
    return times
