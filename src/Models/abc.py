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

        self.class_weight = torch.asarray([1.0, recall_factor], dtype=torch.float32)

        # NOTE: when working outside of lightning, MetricCollection needs a manual reset()

        self.val_metrics = MetricCollection(
            {
                "val_f_beta": MulticlassFBetaScore(
                    num_classes=num_classes, beta=1.0, average="macro"
                ),
                "val_prec": MulticlassPrecision(
                    num_classes=num_classes, average="macro"
                ),
                "val_rec": MulticlassRecall(num_classes=num_classes, average="macro"),
            }
        )

        self.test_metrics = MetricCollection(
            {
                "f_beta": MulticlassFBetaScore(
                    num_classes=num_classes, beta=1.0, average="macro"
                ),
                "prec": MulticlassPrecision(num_classes=num_classes, average="macro"),
                "rec": MulticlassRecall(num_classes=num_classes, average="macro"),
                "auroc": MulticlassAUROC(num_classes=num_classes, average="macro"),
                "cm": ConfusionMatrix(task="multiclass", num_classes=num_classes),
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
        loss = nn.functional.cross_entropy(logits, labels, weight=self.class_weight)

        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        data, labels = batch
        logits = self.forward(data)
        labels = torch.squeeze(labels.long())
        loss = nn.functional.cross_entropy(logits, labels, weight=self.class_weight)
        preds = logits.argmax(dim=-1)
        self.val_metrics.update(preds, labels)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        # NOTE: When logging a MetricCollection or Metric object, Lightning automatically
        # handles the compute() and reset() at the end of the epoch.
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=False)

        return loss

    def test_step(
        self,
        test_dataset: torch.utils.data.Subset,
        output_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        One-pass test step that updates metrics and analysis DataFrame.
        """
        output_df, logits, labels = analyse_test(self.model, test_dataset, output_df)
        # print(logits, "\n\n", labels)
        # Store data into test_metrics (accumulate results)
        self.test_metrics.update(logits, labels)

        return {"output_df": output_df}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        # Using the Enum for safe access
        lr = self.hyperparams.get(self.search_space.LR, 1e-3)
        b0 = self.hyperparams.get(self.search_space.BETA0, 0.9)
        b1 = self.hyperparams.get(self.search_space.BETA1, 0.999)
        wd = self.hyperparams.get(self.search_space.WEIGHT_DECAY, 1e-5)

        return optim.Adam(self.parameters(), lr=lr, betas=(b0, b1), weight_decay=wd)
