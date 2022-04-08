import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from typing import List

from carla.models.api import MLModel
from carla.recourse_methods.api import RecourseMethod
from carla.evaluation.distances import get_distances

def yNN(
    counterfactuals: pd.DataFrame,
    recourse_method: RecourseMethod,
    mlmodel: MLModel,
    y: int,
) -> float:
    """

    Parameters
    ----------
    counterfactuals: Generated counterfactual examples
    recourse_method: Method we want to benchmark
    y: Number of

    Returns
    -------
    float
    """
    number_of_diff_labels = 0
    N = counterfactuals.shape[0]

    df_enc_norm_data = recourse_method.encode_normalize_order_factuals(
        mlmodel.data.raw, with_target=True
    )
    nbrs = NearestNeighbors(n_neighbors=y).fit(df_enc_norm_data.values)

    for i, row in counterfactuals.iterrows():
        if pd.isna(row).any():
            continue
        knn = nbrs.kneighbors(row.values.reshape((1, -1)), y, return_distance=False)[0]
        cf_label = row[mlmodel.data.target]

        for idx in knn:
            neighbour = df_enc_norm_data.iloc[idx]
            neighbour = neighbour.drop(mlmodel.data.target)
            neighbour = neighbour.values.reshape((1, -1))
            neighbour_label = np.argmax(mlmodel.predict_proba(neighbour))

            number_of_diff_labels += np.abs(cf_label - neighbour_label)

    return 1 - (1 / (N * y)) * number_of_diff_labels

def yNN_prob(
    counterfactuals: pd.DataFrame,
    recourse_method: RecourseMethod,
    mlmodel: MLModel,
    y: int,
)-> List[List[float]]:
    """
    TODO
    Parameters
    ----------
    counterfactuals: Generated counterfactual examples
    recourse_method: Method we want to benchmark
    y: Number of

    Returns
    -------
    List[List[float]]
    """
    number_of_diff_labels = []
    N = counterfactuals.shape[0]

    df_enc_norm_data = recourse_method.encode_normalize_order_factuals(
        mlmodel.data.raw, with_target=True
    )
    nbrs = NearestNeighbors(n_neighbors=y).fit(df_enc_norm_data.values)

    for i, row in counterfactuals.iterrows():
        if pd.isna(row).any():
            number_of_diff_labels.append(np.nan)
            continue
        knn = nbrs.kneighbors(row.values.reshape((1, -1)), y, return_distance=False)[0]
        cf_label = row[mlmodel.data.target]
        number_of_diff_labels_local = 0
        
        for idx in knn:
            neighbour = df_enc_norm_data.iloc[idx]
            neighbour = neighbour.drop(mlmodel.data.target)
            neighbour = neighbour.values.reshape((1, -1))
            neighbour_label = np.argmax(mlmodel.predict_proba(neighbour))

            number_of_diff_labels_local += np.abs(cf_label - neighbour_label)
            
        number_of_diff_labels.append([1 - (1 / y) * number_of_diff_labels_local])
        
    return number_of_diff_labels

def yNN_dist(
    counterfactuals: pd.DataFrame,
    recourse_method: RecourseMethod,
    mlmodel: MLModel,
    y: int,
    dist_type: int = 0,
)-> List[List[float]]:
    """
    TODO
    Parameters
    ----------
    counterfactuals: Generated counterfactual examples
    recourse_method: Method we want to benchmark
    y: Number of

    Returns
    -------
    List[List[float]]
    """
    distances = []
    N = counterfactuals.shape[0]

    df_enc_norm_data = recourse_method.encode_normalize_order_factuals(
        mlmodel.data.raw, with_target=True
    )
    nbrs = NearestNeighbors(n_neighbors=y).fit(df_enc_norm_data.values)

    for i, row in counterfactuals.iterrows():
        if pd.isna(row).any():
            distances.append(np.nan)
            continue
        knn = nbrs.kneighbors(row.values.reshape((1, -1)), y, return_distance=False)[0]
        cf_label = row[mlmodel.data.target]
        distances_local = 0
        
        for idx in knn:
            neighbour = df_enc_norm_data.iloc[idx]
            neighbour = neighbour.drop(mlmodel.data.target)
            neighbour = neighbour.values.reshape((1, -1))
            row_copy = row.drop(mlmodel.data.target)
            row_copy = row_copy.values.reshape((1, -1))
            
            distances_local += get_distances(row_copy, neighbour)[0][dist_type]
            
        distances.append([(1 / y) * distances_local])
        
    return distances
