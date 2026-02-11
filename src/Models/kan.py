##codigo dos modelos
from typing import Any
import pandas as pd
from torch import optim
import torch.nn as nn
import torch
from sklearn.metrics import precision_recall_fscore_support
from kan import KAN
from Models.interface import ClassificationModel
from Models.utils import analyse_test, calc_metrics


class MyKan(ClassificationModel):
    hyperparams: dict[str, float | int]

    def __init__(
        self,
        input_dim: int,
        hidden_dims: int,
        n_layers: int,
        num_classes: int,
        **kwargs: dict[str, Any],
    ):
        super().__init__()
        self.hyperparams = kwargs.get("hyperparameters", {})

        width_arr: list[int] = [hidden_dims for _ in range(n_layers)]
        width_arr.insert(0, input_dim)
        width_arr.append(num_classes)

        grid = int(self.hyperparams.get("grid", 12))
        spline_pol_order = int(self.hyperparams.get("spline_pol_order", 5))

        self.model = KAN(
            width=width_arr,
            grid=grid,
            k=spline_pol_order,
            symbolic_enabled=False,
            auto_save=False,
        )

        self.example_input_array = torch.zeros(1, input_dim, dtype=torch.float32)
        self.save_hyperparameters()

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        # training_step defines the train loop.
        # it is independent of forward
        data, labels = batch
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
        logits = self.model(data)
        labels = torch.squeeze(labels.long())
        loss = nn.functional.cross_entropy(logits, labels)

        f_beta, prec, rec, roc_auc = calc_metrics(labels, logits)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_prec", float(prec), prog_bar=False)
        self.log("val_rec", float(rec), prog_bar=False)
        self.log("val_f_beta", float(f_beta), prog_bar=False)
        self.log("val_roc_auc", float(roc_auc), prog_bar=False)

        return loss

    def configure_optimizers(self):
        lr = self.hyperparams.get("lr", 1e-5)
        beta0 = self.hyperparams.get("beta0", 0.99)
        beta1 = self.hyperparams.get("beat1", 0.9999)
        weight_decay = self.hyperparams.get("weight_decay", 1e-5)
        optimizer = optim.Adam(
            self.parameters(), lr=lr, betas=(beta0, beta1), weight_decay=weight_decay
        )
        return optimizer

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
