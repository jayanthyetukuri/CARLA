import pandas as pd
import numpy as np
from typing import List

def individual_diversity(counterfactuals: pd.DataFrame, factuals: pd.DataFrame) -> List[float]:
    """
    Computes success rate for all counterfactuals

    Parameters
    ----------
    counterfactuals: All counterfactual examples inclusive nan values

    Returns
    -------
    % non-null

    """
    if factuals.shape[0] != counterfactuals.shape[0] or factuals.shape[1] != counterfactuals.shape[1]:
        raise ValueError(
            "Counterfactuals and factuals should contain the same amount of samples with the same number of features"
        )
    diff = counterfactuals - factuals
    num_diff = lambda x: sum(x != 0)
    return diff.apply(num_diff, axis=1)

def avg_diversity(counterfactuals: pd.DataFrame, factuals: pd.DataFrame) -> float:
    """
    Computes success rate for all counterfactuals

    Parameters
    ----------
    counterfactuals: All counterfactual examples inclusive nan values

    Returns
    -------
    % non-null

    """
    if factuals.shape[0] != counterfactuals.shape[0] or factuals.shape[1] != counterfactuals.shape[1]:
        raise ValueError(
            "Counterfactuals and factuals should contain the same amount of samples with the same number of features"
        )
    diff = counterfactuals - factuals
    num_diff = lambda x: sum(x != 0)
    total_diff = diff.apply(num_diff, axis=1)
    return np.mean(total_diff)

