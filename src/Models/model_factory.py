from enum import Enum
from typing import Iterable, Type
from Models.abc import ClassificationModel, HyperParameterModel
from Models.kan import MyKan, KANSearchSpace
from Models.mlp import MLP, MLPSearchSpace
from Models.rand_forest import RandomForestModel, RandomForestHyperParameters
from Models.xgboost import XGBoostModel, XGBoostHyperParameters


class ModelFactory:
    """A fancy dictionary that maps model names to ClassificationModel subclasses."""

    class Models(str, Enum):
        """Enum of available models."""

        KAN = "KAN"
        MLP = "MLP"
        RANDOMFOREST = "RANDOMFOREST"
        XGBOOST = "XGBOOST"

    # Dictionary mapping enum keys to model classes
    _models: dict[Models, Type[ClassificationModel]] = {
        Models.KAN: MyKan,
        Models.MLP: MLP,
        Models.RANDOMFOREST: RandomForestModel,
        Models.XGBOOST: XGBoostModel,
    }

    _params: dict[Models, Type[HyperParameterModel]] = {
        Models.KAN: KANSearchSpace,
        Models.MLP: MLPSearchSpace,
        Models.RANDOMFOREST: RandomForestHyperParameters,
        Models.XGBOOST: XGBoostHyperParameters,
    }

    def __getitem__(
        self, key: str
    ) -> tuple[Type[ClassificationModel], Type[HyperParameterModel]]:
        """Get a model class by enum key or string name."""
        if isinstance(key, str):
            assert key in list(self._models.keys()), (
                f"key {key} not in {self._models.keys()}"
            )
            key = self.Models[key.upper()]
        return self._models[key], self._params[key]

    def __setitem__(self, key, value):
        """Prevent modification of the factory."""
        raise TypeError("ModelFactory is immutable")

    def __iter__(self) -> Iterable:
        """Iterate over available model names."""
        return zip(self._models, self._params)

    def __len__(self):
        """Return number of available models."""
        return len(self._models)

    def get(
        self, key: str, default=None
    ) -> (
        tuple[Type[ClassificationModel], Type[HyperParameterModel]] | tuple[None, None]
    ):
        """Safely get a model class."""
        try:
            return self[key]
        except (KeyError, ValueError):
            return default, default
