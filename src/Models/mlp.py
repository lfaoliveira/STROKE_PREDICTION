from enum import Enum
from typing import Any, Dict

import optuna
import torch
import torch.nn as nn
from pydantic import ConfigDict

from Models.abc import ClassificationModel, HyperParameterModel


class MLPSearchSpace(HyperParameterModel):
    """
    Search Space definition for KAN Model: Contains the Keys, Boundaries, and Logic.
    """

    model_config = ConfigDict(use_enum_values=True, arbitrary_types_allowed=True)

    # Internal Enumerator
    class Keys(str, Enum):
        LR = "lr"
        BETA0 = "beta0"
        BETA1 = "beta1"
        WEIGHT_DECAY = "weight_decay"
        BATCH_SIZE = "batch_size"
        HIDDEN_DIMS = "hidn_dims"
        N_LAYERS = "n_layers"

    def suggest(self, values_dict: dict[Keys, float | int]) -> dict[str, float | int]:
        """
        Function to organize hyperparameter definition

        :param values_dict: dictionary mapping Keys enum members to their values
        """
        K = self.Keys
        # Use K(key).value to ensure we return the string values defined in the Enum
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
            K.HIDDEN_DIMS: trial.suggest_int(K.HIDDEN_DIMS, 16, 256, step=16),
            K.LR: trial.suggest_float(K.LR, 1e-5, 1e-2, log=True),
            K.WEIGHT_DECAY: trial.suggest_float(K.WEIGHT_DECAY, 1e-7, 1e-2, log=True),
            K.BETA0: trial.suggest_float(K.BETA0, 0.900, 0.9999),
            K.BETA1: trial.suggest_float(K.BETA1, 0.900, 0.9999),
            K.N_LAYERS: trial.suggest_int(K.N_LAYERS, 4, 12),
        }


class MLP(ClassificationModel):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        recall_factor: float,
        **kwargs,
    ):
        super().__init__(input_dim, num_classes, recall_factor)
        # Accessing hyperparameters using the Enum keys
        self.search_space = MLPSearchSpace().Keys
        self.hyperparams = kwargs.get("hyperparameters", {})

        self.recall_factor = recall_factor

        hidden_dims = int(self.hyperparams.get(self.search_space.HIDDEN_DIMS, 256))
        n_layers = int(self.hyperparams.get(self.search_space.N_LAYERS, 4))

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dims, dtype=torch.float32),
            nn.ReLU(),
        )
        for _ in range(n_layers):
            self.model.append(nn.LazyLinear(hidden_dims, dtype=torch.float32))
            self.model.append(nn.SELU())

        self.model.append(nn.Linear(hidden_dims, num_classes, dtype=torch.float32))
        self.example_input_array = torch.zeros(input_dim, dtype=torch.float32)
        self.save_hyperparameters()
        # print(self.model)
