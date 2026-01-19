##codigo dos modelos
from torch import optim
import torch.nn as nn
import torch
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from kan import KAN
from Models.interface import ClassificationModel


class MyKan(ClassificationModel):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: int,
        n_layers: int,
        num_classes: int,
        **kwargs,
    ):
        super().__init__()
        self.hyperparams: dict[str, float | int] = kwargs.get("hyperparameters", {})

        width_arr: list[int] = [hidden_dims for _ in range(n_layers)]
        width_arr.insert(0, input_dim)
        width_arr.append(num_classes)
        self.model = KAN(
            width=width_arr, grid=12, k=5, symbolic_enabled=False, auto_save=False
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
        prec, rec, f1, support = precision_recall_fscore_support(
            labels.numpy(force=True),
            torch.argmax(logits, dim=1).numpy(force=True),
            zero_division=0,
        )

        prec = np.mean(prec) if isinstance(prec, np.ndarray) else float(prec)
        rec = np.mean(rec) if isinstance(rec, np.ndarray) else float(rec)
        f1 = np.mean(f1) if isinstance(f1, np.ndarray) else float(f1)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_prec", float(prec), prog_bar=False)
        self.log("val_rec", float(rec), prog_bar=False)
        self.log("val_f1", float(f1), prog_bar=False)
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
