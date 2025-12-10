##codigo dos modelos
from torch import optim
import torch.nn as nn
import torch
from lightning import LightningModule


class MLP(LightningModule):
    def __init__(
        self, input_dim: int, hidden_dims: int, n_layers: int, num_classes: int
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dims, dtype=torch.float32),
            nn.ReLU(),
        )
        for _ in range(n_layers):
            self.model.append(nn.LazyLinear(hidden_dims, dtype=torch.float32))
            self.model.append(nn.SELU())

        self.model.append(nn.Linear(hidden_dims, num_classes, dtype=torch.float32))
        self.save_hyperparameters()
        # print(self.model)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        data, labels = batch
        logits = self.model(data)
        loss = nn.functional.cross_entropy(logits, labels)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, prog_bar=True)
        # self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        data, labels = batch
        logits = self.model(data)
        loss = nn.functional.cross_entropy(logits, labels)
        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss, prog_bar=True)
        # self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    # def forward(self, x) -> Tensor:
    #     return self.model(x)
