from enum import Enum
from typing import Any, Dict

import optuna
import torch
import torch.nn as nn
from kan import KAN
from pydantic import ConfigDict
from torch import optim

from Models.interface import ClassificationModel, HyperParameterModel
from Models.utils import analyse_test, calc_metrics


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
            K.HIDDEN_DIMS: trial.suggest_int(K.HIDDEN_DIMS, 2, 8, step=2),
            K.GRID: trial.suggest_categorical(K.GRID, [14, 19, 24, 29, 34, 40]),
            K.SPLINE_POL_ORDER: trial.suggest_categorical(
                K.SPLINE_POL_ORDER, [3, 5, 7]
            ),
            K.LR: trial.suggest_float(K.LR, 1e-5, 1e-2, log=True),
            K.WEIGHT_DECAY: trial.suggest_float(K.WEIGHT_DECAY, 1e-7, 1e-2, log=True),
            K.BETA0: trial.suggest_float(K.BETA0, 0.900, 0.9999),
            K.BETA1: trial.suggest_float(K.BETA1, 0.900, 0.9999),
        }


class MyKan(ClassificationModel):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        recall_factor: float,
        **kwargs: Any,
    ):
        super().__init__()
        # Accessing hyperparameters using the Enum keys
        self.search_space = KANSearchSpace().Keys
        self.hyperparams = kwargs.get("hyperparameters", {})

        self.recall_factor = recall_factor

        # Define KAN width (typically much thinner than MLP)
        # Using logic of hidden_dims // 16 for a thin KAN, to mantain model capacity equivalence
        kan_width = int(self.hyperparams.get(self.search_space.HIDDEN_DIMS, 24))
        width_arr = [input_dim, kan_width, num_classes]
        spline_order = int(self.hyperparams.get(self.search_space.SPLINE_POL_ORDER, 3))
        grid = int(self.hyperparams.get(self.search_space.GRID, 12))

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

    def forward(self, x) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        data, labels = batch
        logits = self.forward(data)
        labels = torch.squeeze(labels.long())
        loss = nn.functional.cross_entropy(logits, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        data, labels = batch
        logits = self.forward(data)
        labels = torch.squeeze(labels.long())
        loss = nn.functional.cross_entropy(logits, labels)

        f_beta, prec, rec, roc_auc = calc_metrics(labels, logits, self.recall_factor)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_prec", float(prec), prog_bar=False)
        self.log("val_rec", float(rec), prog_bar=False)
        self.log("val_f_beta", float(f_beta), prog_bar=False)
        self.log("val_roc_auc", float(roc_auc), prog_bar=False)
        return loss

    def configure_optimizers(self):
        # Using the Enum for safe access
        lr = self.hyperparams.get(self.search_space.LR, 1e-3)
        b0 = self.hyperparams.get(self.search_space.BETA0, 0.9)
        b1 = self.hyperparams.get(self.search_space.BETA1, 0.999)
        wd = self.hyperparams.get(self.search_space.WEIGHT_DECAY, 1e-5)

        return optim.Adam(self.parameters(), lr=lr, betas=(b0, b1), weight_decay=wd)

    def test_step(self, batch, batch_idx, output_df, test_dataset):
        # We pass self.model to ensure we are testing the KAN architecture
        analyse_test(self.model, batch, batch_idx, output_df, test_dataset)
        return output_df
