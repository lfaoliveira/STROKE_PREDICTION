from typing import TYPE_CHECKING, Any, Dict, Tuple

import pandas as pd
import torch
import torch.nn as nn
from lightning import LightningModule
from pydantic import BaseModel
from torch import optim

from torchmetrics import ConfusionMatrix, MetricCollection
from torchmetrics.classification import (
    # TODO: add ROC calculation
    MulticlassAUROC,
    MulticlassFBetaScore,
    MulticlassPrecision,
    MulticlassRecall,
)

if TYPE_CHECKING:
    import torch.utils.data

from Models.utils import analyse_test


# classe comum para todos os definidores de hyperparametros
class HyperParameterModel(BaseModel):
    pass


class ClassificationModel(LightningModule):
    def __init__(
        self, input_dim: int, num_classes: int, recall_factor: float, **kwargs: Any
    ) -> None:
        super().__init__()
        self.recall_factor = recall_factor
        self.hyperparams: Dict[str, Any] = kwargs.get("hyperparameters", {})
        self.search_space: Any = None
        self.model: nn.Module = nn.Identity()

        # NOTE: when working outside of lightning, MetricCollection needs a manual reset()

        self.val_metrics = MetricCollection(
            {
                "f_beta": MulticlassFBetaScore(
                    num_classes=num_classes, beta=recall_factor, average="macro"
                ),
                "prec": MulticlassPrecision(num_classes=num_classes, average="macro"),
                "rec": MulticlassRecall(num_classes=num_classes, average="macro"),
            }
        )

        self.test_metrics = MetricCollection(
            {
                "f_beta": MulticlassFBetaScore(
                    num_classes=num_classes, beta=recall_factor, average="macro"
                ),
                "cm": ConfusionMatrix(task="binary", num_classes=num_classes),
                "prec": MulticlassPrecision(num_classes=num_classes, average="macro"),
                "rec": MulticlassRecall(num_classes=num_classes, average="macro"),
            }
        )

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        data, labels = batch
        logits = self.forward(data)
        labels = torch.squeeze(labels.long())
        loss = nn.functional.cross_entropy(logits, labels)

        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        data, labels = batch
        logits = self.forward(data)
        labels = torch.squeeze(labels.long())
        loss = nn.functional.cross_entropy(logits, labels)
        preds = logits.argmax(dim=-1)
        self.val_metrics.update(preds, labels)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        # NOTE: lightning automatically agregates metrics by MEAN when on_epoch=True.
        self.log_dict(self.val_metrics, on_epoch=True, prog_bar=False)

        return loss

    def on_validation_epoch_end(self):
        self.val_metrics.reset()

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        output_df: pd.DataFrame,
        test_dataset: torch.utils.data.Subset,
    ):
        # We pass self.model to ensure we are testing the architecture
        analyse_test(self.model, batch, batch_idx, output_df, test_dataset)

        return {"output_df": output_df}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        # Using the Enum for safe access
        lr = self.hyperparams.get(self.search_space.LR, 1e-3)
        b0 = self.hyperparams.get(self.search_space.BETA0, 0.9)
        b1 = self.hyperparams.get(self.search_space.BETA1, 0.999)
        wd = self.hyperparams.get(self.search_space.WEIGHT_DECAY, 1e-5)

        return optim.Adam(self.parameters(), lr=lr, betas=(b0, b1), weight_decay=wd)
