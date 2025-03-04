from typing import Any, Dict, List

import pandas as pd

from carla.data.load_catalog import load_catalog

from ..api import Data
from .load_data import load_dataset


class DataCatalog(Data):
    """
    Use already implemented datasets.

    Parameters
    ----------
    data_name : {'adult', 'compas', 'give_me_some_credit'}
        Used to get the correct dataset from online repository.

    Returns
    -------
    None
    """

    def __init__(self, data_name: str):
        self.name = data_name

        catalog_content = ["continous", "categorical", "immutable", "target"]
        self.catalog: Dict[str, Any] = load_catalog(  # type: ignore
            "data_catalog.yaml", data_name, catalog_content
        )

        for key in ["continous", "categorical", "immutable"]:
            if self.catalog[key] is None:
                self.catalog[key] = []

        raw, train_raw, test_raw = load_dataset(data_name)
        self._raw: pd.DataFrame = raw
        self._train_raw: pd.DataFrame = train_raw
        self._test_raw: pd.DataFrame = test_raw

    @property
    def categoricals(self) -> List[str]:
        return self.catalog["categorical"]

    @property
    def continous(self) -> List[str]:
        return self.catalog["continous"]

    @property
    def immutables(self) -> List[str]:
        return self.catalog["immutable"]

    @property
    def target(self) -> str:
        return self.catalog["target"]

    @property
    def raw(self) -> pd.DataFrame:
        return self._raw.copy()

    @property
    def train_raw(self) -> pd.DataFrame:
        return self._train_raw.copy()

    @property
    def test_raw(self) -> pd.DataFrame:
        return self._test_raw.copy()
