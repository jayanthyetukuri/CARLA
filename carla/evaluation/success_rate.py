import pandas as pd
from typing import List

def success_rate(counterfactuals: pd.DataFrame) -> float:
    """
    Computes success rate for all counterfactuals

    Parameters
    ----------
    counterfactuals: All counterfactual examples inclusive nan values

    Returns
    -------
    % non-null

    """
    return (counterfactuals.dropna().shape[0]) / counterfactuals.shape[0]

def individual_success_rate(counterfactuals: pd.DataFrame) -> List[int]:
    """
    Computes binary success eval for all given individuals

    Parameters
    ----------
    counterfactuals: All counterfactual examples inclusive nan values

    Returns
    -------
    list of 0/1 vals based on null/non-null
    """
    check_na = lambda x: any(pd.isna(x))
    return counterfactuals.apply(check_na, axis=1).astype(int)
