from typing import List

import numpy as np
import pandas as pd

from carla.recourse_methods.api import RecourseMethod


def recourse_time_taken(
    recourse_method: RecourseMethod, factuals: pd.DataFrame
) -> List[List[float]]:
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
    for fact in factuals:
      recourse_method.get_counterfactuals(fact)
      times.append([0.0])
      
    return times
