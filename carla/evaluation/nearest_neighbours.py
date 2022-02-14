import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from carla.models.api import MLModel
from carla.recourse_methods.api import RecourseMethod


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
) -> List[List[float]]:
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
) -> List[List[float]]:
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
        knn = nbrs.kneighbors(row.values.reshape((1, -1)), y, return_distance=False)[0]
        cf_label = row[mlmodel.data.target]
        distances_local = 0
        
        for idx in knn:
            neighbour = df_enc_norm_data.iloc[idx]
            neighbour = neighbour.drop(mlmodel.data.target)
            neighbour = neighbour.values.reshape((1, -1))

            distances_local += dist(row, neighbour)
            
        distances.append([(1 / y) * distances_local])
        
    return distances
