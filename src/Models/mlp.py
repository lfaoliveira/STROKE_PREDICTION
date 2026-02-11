##codigo dos modelos
from enum import Enum
from typing import Any, Dict

import optuna
import pandas as pd
from pydantic import ConfigDict
import torch
import torch.nn as nn
from torch import optim

from Models.interface import ClassificationModel, HyperParameterModel
from Models.utils import analyse_test, calc_metrics


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

    def suggest(self, values_dict: dict[str, float | int]) -> dict[str, float | int]:
        """
        Function to organize hyperparameter definition

        :param values_dict: dictionary with hyperparameters defined
        :type values_dict: dict[str, float | int]
        """
        K = self.Keys
        hypers = {str(K(key)): value for key, value in values_dict.items()}
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
        **kwargs,
    ):
        super().__init__()
        # Accessing hyperparameters using the Enum keys
        self.search_space = MLPSearchSpace().Keys
        self.hyperparams = kwargs.get("hyperparameters", {})

        hidden_dims = self.hyperparams.get(self.search_space.HIDDEN_DIMS)
        n_layers = self.hyperparams.get(self.search_space.N_LAYERS)

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

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        # training_step defines the train loop.
        # it is independent of forward
        data, labels = batch
        # print("TREINAMENTO: ", data.shape, labels.shape)

        logits = self.model(data)
        labels = torch.squeeze(labels.long())

        loss = nn.functional.cross_entropy(logits, labels)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, prog_bar=True)
        # self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        # training_step defines the train loop.
        # it is independent of forward

        data, labels = batch
        # print(data.shape, labels.shape)
        logits = self.model(data)
        labels = torch.squeeze(labels.long())

        loss = nn.functional.cross_entropy(logits, labels)

        f_beta, prec, rec, roc_auc = calc_metrics(labels, logits)

        self.log("val_loss", loss, prog_bar=False)
        self.log("val_prec", float(prec), prog_bar=False)
        self.log("val_rec", float(rec), prog_bar=False)
        self.log("val_f_beta", float(f_beta), prog_bar=False)
        self.log("val_roc_auc", float(roc_auc), prog_bar=False)
        return loss

    def configure_optimizers(self):
        lr = self.hyperparams.get(self.search_space.LR, 1e-3)
        b0 = self.hyperparams.get(self.search_space.BETA0, 0.9)
        b1 = self.hyperparams.get(self.search_space.BETA1, 0.999)
        wd = self.hyperparams.get(self.search_space.WEIGHT_DECAY, 1e-5)

        return optim.Adam(self.parameters(), lr=lr, betas=(b0, b1), weight_decay=wd)

    def forward(self, x) -> torch.Tensor:
        return self.model(x)

    def test_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        output_df: pd.DataFrame,
        test_dataset: torch.utils.data.Subset,
    ):
        analyse_test(self.model, batch, batch_idx, output_df, test_dataset)

        return output_df
