from enum import Enum
from typing import Any, Dict

import optuna
import torch
from kan import KAN
from pydantic import ConfigDict

from Models.abc import ClassificationModel, HyperParameterModel


class KANSearchSpace(HyperParameterModel):
    """
    Search Space definition for KAN Model: Contains the Keys, Boundaries, and Logic.
    """

    model_config = ConfigDict(use_enum_values=True, arbitrary_types_allowed=True)

    # Internal Enumerator
    class Keys(str, Enum):
        GRID = "grid"
        SPLINE_POL_ORDER = "spline_pol_order"
        LR = "lr"
        BETA0 = "beta0"
        BETA1 = "beta1"
        WEIGHT_DECAY = "weight_decay"
        BATCH_SIZE = "batch_size"
        HIDDEN_DIMS = "hidn_dims"

    def suggest(self, values_dict: dict[Keys, float | int]) -> dict[str, float | int]:
        """
        Function to organize hyperparameter definition

        :param values_dict: dictionary with hyperparameters defined
        :type values_dict: dict[str, float | int]
        """
        K = self.Keys
        hypers = {K(key).value: value for key, value in values_dict.items()}
        return hypers

    # 3. Suggestion Logic
    def suggest_optuna(self, trial: optuna.Trial | None = None) -> Dict[str, Any]:
        """
        Maps trial suggestions to the internal Keys namespace.
        """
        K = self.Keys  # alias
        assert trial is not None, "trial nulo!"

        # Search Space dict
        return {
            K.BATCH_SIZE: trial.suggest_categorical(K.BATCH_SIZE, [8, 16, 32, 64]),
            K.HIDDEN_DIMS: trial.suggest_int(K.HIDDEN_DIMS, 32, 100, step=2),
            K.GRID: trial.suggest_int(K.GRID, 60, 154, step=2),
            K.SPLINE_POL_ORDER: trial.suggest_categorical(
                K.SPLINE_POL_ORDER, [3, 4, 5, 7, 9]
            ),
            K.LR: trial.suggest_float(K.LR, 1e-7, 1e-2, log=True),
            K.WEIGHT_DECAY: trial.suggest_float(K.WEIGHT_DECAY, 1e-7, 1e-2, log=True),
            K.BETA0: trial.suggest_float(K.BETA0, 0.900, 0.9999),
            K.BETA1: trial.suggest_float(K.BETA1, 0.900, 0.9999),
        }


class MyKan(ClassificationModel):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        recall_factor: list[float],
        **kwargs: Any,
    ):
        super().__init__(input_dim, num_classes, recall_factor)
        # Accessing hyperparameters using the Enum keys
        self.search_space = KANSearchSpace().Keys
        self.hyperparams = kwargs.get("hyperparameters", {})

        # Define KAN width (typically much thinner than MLP)
        # Using logic of hidden_dims // 16 compared to an MLP Hidden dims, to mantain model capacity equivalence
        assert self.hyperparams is not None

        key = self.search_space.HIDDEN_DIMS.value
        kan_width = int(self.hyperparams.get(key, -1))

        key = self.search_space.SPLINE_POL_ORDER.value
        spline_order = int(self.hyperparams.get(key, -1))

        key = self.search_space.GRID.value
        grid = int(self.hyperparams.get(key, -1))

        assert kan_width > 0, "ERROR AT MODEL PARAMETERS: kan_width must be > 0!"
        assert int(num_classes) > 0, (
            "ERROR AT MODEL PARAMETERS: num_classes must be > 0!"
        )
        assert spline_order > 0, "ERROR AT MODEL PARAMETERS: spline_order must be > 0!"
        assert grid > 0, "ERROR AT MODEL PARAMETERS: grid must be > 0!"

        width_arr = [input_dim, kan_width, num_classes]

        self.model = KAN(
            width=width_arr,
            grid=grid,
            k=spline_order,
            symbolic_enabled=False,
            auto_save=False,
        )

        self.example_input_array = torch.zeros(1, input_dim, dtype=torch.float32)

        # Log the calculated capacity for MLflow/Tensorboard
        self.save_hyperparameters()
