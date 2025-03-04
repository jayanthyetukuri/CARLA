import timeit
from typing import Union

import numpy as np
import pandas as pd
import copy

from carla.evaluation.distances import get_distances
from carla.evaluation.nearest_neighbours import yNN, yNN_prob, yNN_dist
from carla.evaluation.manifold import yNN_manifold, sphere_manifold
from carla.evaluation.process_nans import remove_nans
from carla.evaluation.redundancy import redundancy
from carla.evaluation.success_rate import success_rate, individual_success_rate
from carla.evaluation.diversity import individual_diversity, avg_diversity
from carla.evaluation.violations import constraint_violation
from carla.evaluation.recourse_time import recourse_time_taken
from carla.models.api import MLModel
from carla.models.catalog import MLModelCatalog
from carla.recourse_methods.api import RecourseMethod
from carla.recourse_methods.processing import get_drop_columns_binary


class Benchmark:
    """
    The benchmarking class contains all measurements.
    It is possible to run only individual evaluation metrics or all via one single call.

    For every given factual, the benchmark object will generate one counterfactual example with
    the given recourse method.

    Parameters
    ----------
    mlmodel: carla.models.MLModel
        Black Box model we want to explain
    recmodel: carla.recourse_methods.RecourseMethod
        Recourse method we want to benchmark
    factuals: pd.DataFrame
        Instances we want to find counterfactuals

    Methods
    -------
    compute_ynn:
        Computes y-Nearest-Neighbours for generated counterfactuals
    compute_average_time:
        Computes average time for generated counterfactual
    compute_distances:
        Calculates the distance measure and returns it as dataframe
    compute_constraint_violation:
        Computes the constraint violation per factual as dataframe
    compute_redundancy:
        Computes redundancy for each counterfactual
    compute_success_rate:
        Computes success rate for the whole recourse method.
    run_benchmark:
        Runs every measurement and returns every value as dict.
    """

    def __init__(
        self,
        mlmodel: Union[MLModel, MLModelCatalog],
        recourse_method: RecourseMethod,
        factuals: pd.DataFrame,
        dataset: pd.DataFrame = None
    ) -> None:

        self._mlmodel = mlmodel
        self._recourse_method = recourse_method
        self._full_dataset = dataset
        start = timeit.default_timer()
        self._counterfactuals = recourse_method.get_counterfactuals(factuals)
        stop = timeit.default_timer()
        self._timer = stop - start

        # Avoid using scaling and normalizing more than once
        if isinstance(mlmodel, MLModelCatalog):
            self._mlmodel.use_pipeline = False  # type: ignore

        self._factuals = copy.deepcopy(factuals)

        # Normalizing and encoding factual for later use
        self._enc_norm_factuals = recourse_method.encode_normalize_order_factuals(
            factuals, with_target=True
        )

    def compute_ynn(self) -> pd.DataFrame:
        """
        Computes y-Nearest-Neighbours for generated counterfactuals

        Returns
        -------
        pd.DataFrame
        """
        _, counterfactuals_without_nans = remove_nans(
            self._factuals, self._counterfactuals
        )

        if counterfactuals_without_nans.empty:
            ynn = np.nan
        else:
            ynn = yNN(
                counterfactuals_without_nans, self._recourse_method, self._mlmodel, 5
            )

        columns = ["y-Nearest-Neighbours"]

        return pd.DataFrame([[ynn]], columns=columns)

    def compute_average_time(self) -> pd.DataFrame:
        """
        Computes average time for generated counterfactual

        Returns
        -------
        pd.DataFrame
        """

        avg_time = self._timer / self._counterfactuals.shape[0]

        columns = ["Average_Time"]

        return pd.DataFrame([[avg_time]], columns=columns)

    def compute_distances(self) -> pd.DataFrame:
        """
        Calculates the distance measure and returns it as dataframe

        Returns
        -------
        pd.DataFrame
        """
        factual_without_nans, counterfactuals_without_nans = remove_nans(
            self._enc_norm_factuals, self._counterfactuals
        )

        columns = ["Distance_1", "Distance_2", "Distance_3", "Distance_4"]

        if counterfactuals_without_nans.empty:
            return pd.DataFrame(columns=columns)

        if self._mlmodel.encoder.drop is None:
            # To prevent double count of encoded features without drop if_binary
            binary_columns_to_drop = get_drop_columns_binary(
                self._mlmodel.data.categoricals,
                counterfactuals_without_nans.columns.tolist(),
            )
            counterfactuals_without_nans = counterfactuals_without_nans.drop(
                binary_columns_to_drop, axis=1
            )
            factual_without_nans = factual_without_nans.drop(
                binary_columns_to_drop, axis=1
            )

        arr_f = factual_without_nans.to_numpy()
        arr_cf = counterfactuals_without_nans.to_numpy()

        distances = get_distances(arr_f, arr_cf)

        output = pd.DataFrame(distances, columns=columns)

        return output

    def compute_constraint_violation(self) -> pd.DataFrame:
        """
        Computes the constraint violation per factual as dataframe

        Returns
        -------
        pd.Dataframe
        """
        factual_without_nans, counterfactuals_without_nans = remove_nans(
            self._factuals, self._counterfactuals
        )

        if counterfactuals_without_nans.empty:
            violations = []
        else:
            violations = constraint_violation(
                self._mlmodel, counterfactuals_without_nans, factual_without_nans
            )
        columns = ["Constraint_Violation"]

        return pd.DataFrame(violations, columns=columns)
    
    def compute_time_taken(self) -> pd.DataFrame:
        
        """
        TODO
        Computes time taken for generated counterfactual

        Returns
        -------
        pd.DataFrame
        """
        
        factual_without_nans, counterfactuals_without_nans = remove_nans(
            self._factuals, self._counterfactuals
        )

        if counterfactuals_without_nans.empty:
            time_taken = []
        else:
            time_taken = recourse_time_taken(
                self._recourse_method, self._factuals
            )
        columns = ["Time_taken"]

        return pd.DataFrame(time_taken, columns=columns)

    def compute_individual_diversity(self) -> pd.DataFrame:
        
        """
        TODO
        Computes instance-wise diveristy for generated counterfactual

        Returns
        -------
        pd.DataFrame
        """
        
        factual_without_nans, counterfactuals_without_nans = remove_nans(
            self._factuals, self._counterfactuals
        )

        if counterfactuals_without_nans.empty:
            diveristy = []
        else:
            diveristy = individual_diversity(
                counterfactuals_without_nans, factual_without_nans
            )
        columns = ["Individual_Diversity"]

        return pd.DataFrame(diveristy, columns=columns)

    def compute_avg_diversity(self) -> pd.DataFrame:
        
        """
        TODO
        Computes average diversity for generated counterfactual

        Returns
        -------
        pd.DataFrame
        """
        
        factual_without_nans, counterfactuals_without_nans = remove_nans(
            self._factuals, self._counterfactuals
        )

        if counterfactuals_without_nans.empty:
            diversity = []
        else:
            diversity = avg_diversity(
                counterfactuals_without_nans, factual_without_nans
            )
        columns = ["Average_Diversity"]

        return pd.DataFrame(diversity, columns=columns)
    
    def compute_ynn_dist(self) -> pd.DataFrame:
        """
        TODO
        Computes y-Nearest-Neighbours for generated counterfactuals

        Returns
        -------
        pd.DataFrame
        """
        _, counterfactuals_without_nans = remove_nans(
            self._factuals, self._counterfactuals
        )

        if counterfactuals_without_nans.empty:
            ynn = np.nan
        else:
            ynn = yNN_dist(
                counterfactuals_without_nans, self._recourse_method, self._mlmodel, 5
            )

        columns = ["y-Nearest-Neighbours-Distance"]

        output = pd.DataFrame(ynn, columns=columns)

        return output

    def compute_manifold_ynn(self) -> pd.DataFrame:
        """
        TODO
        Computes y-Nearest-Neighbours for generated counterfactuals with respect to positive class

        Returns
        -------
        pd.DataFrame
        """
        _, counterfactuals_without_nans = remove_nans(
            self._factuals, self._counterfactuals
        )

        if counterfactuals_without_nans.empty:
            ynn = np.nan
        else:
            ynn = yNN_manifold(
                counterfactuals_without_nans, self._recourse_method, self._mlmodel, 5
            )

        columns = ["y-Nearest-Neighbours-Manifold-Distance"]

        output = pd.DataFrame(ynn, columns=columns)

        return output

    def compute_manifold_sphere(self) -> pd.DataFrame:
        """
        TODO
        Computes neighbor distance for generated counterfactuals with respect to positive class within sphere

        Returns
        -------
        pd.DataFrame
        """
        _, counterfactuals_without_nans = remove_nans(
            self._factuals, self._counterfactuals
        )

        if counterfactuals_without_nans.empty:
            ynn = np.nan
        else:
            ynn = sphere_manifold(
                counterfactuals_without_nans, self._recourse_method, self._mlmodel
            )

        columns = ["Sphere-Manifold-Distance"]

        output = pd.DataFrame(ynn, columns=columns)

        return output
        
    def compute_ynn_prob(self) -> pd.DataFrame:
        """
        TODO
        Computes y-Nearest-Neighbours for generated counterfactuals

        Returns
        -------
        pd.DataFrame
        """
        _, counterfactuals_without_nans = remove_nans(
            self._factuals, self._counterfactuals
        )
        
        if counterfactuals_without_nans.empty:
            ynn = np.nan
        else:
            ynn = yNN_prob(
                counterfactuals_without_nans, self._recourse_method, self._mlmodel, 5
            )
        print(ynn)
        columns = ["y-Nearest-Neighbours-Probability"]

        output = pd.DataFrame(ynn, columns=columns)

        return output
    
    def compute_redundancy(self) -> pd.DataFrame:
        """
        Computes redundancy for each counterfactual

        Returns
        -------
        pd.Dataframe
        """
        factual_without_nans, counterfactuals_without_nans = remove_nans(
            self._enc_norm_factuals, self._counterfactuals
        )

        if counterfactuals_without_nans.empty:
            redundancies = []
        else:
            redundancies = redundancy(
                factual_without_nans, counterfactuals_without_nans, self._mlmodel
            )

        columns = ["Redundancy"]

        return pd.DataFrame(redundancies, columns=columns)

    def compute_success_rate(self) -> pd.DataFrame:
        """
        Computes success rate for the whole recourse method.

        Returns
        -------
        pd.Dataframe
        """

        rate = success_rate(self._counterfactuals)
        columns = ["Success_Rate"]

        return pd.DataFrame([[rate]], columns=columns)

    def compute_individual_success_rate(self) -> pd.DataFrame:
        """
        Computes success rate for the whole recourse method.

        Returns
        -------
        pd.Dataframe
        """

        rate = individual_success_rate(self._counterfactuals)
        columns = ["Individual_Success_Rate"]

        return pd.DataFrame([[rate]], columns=columns)

    def run_benchmark(self) -> pd.DataFrame:
        """
        Runs every measurement and returns every value as dict.

        Returns
        -------
        pd.DataFrame
        """
        pipeline = [
            self.compute_distances(),
            self.compute_constraint_violation(),
            self.compute_redundancy(),
            self.compute_ynn_prob(),
            self.compute_ynn_dist(),
            #self.compute_individual_success_rate(),
            #self.compute_individual_diversity(),
            self.compute_time_taken(),
            self.compute_manifold_ynn(),
            self.compute_manifold_sphere(),
            self.compute_success_rate(),
            self.compute_average_time(),
            self.compute_ynn()
            #self.compute_avg_diversity()
        ]

        output = pd.concat(pipeline, axis=1)

        return output
