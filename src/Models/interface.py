from lightning import LightningModule
from pydantic import BaseModel


class ClassificationModel(LightningModule):
    def validation_step(self, batch, batch_idx):
        raise NotImplementedError("validation_step must be implemented")

    def training_step(self, batch, batch_idx):
        raise NotImplementedError("training_step must be implemented")

    def configure_optimizers(self):
        raise NotImplementedError("configure_optimizers must be implemented")

    def forward(self, x):
        raise NotImplementedError("forward must be implemented")

    def test_step(self, batch, batch_idx, output_df, test_dataset):
        raise NotImplementedError("forward must be implemented")


# classe comum para todos os definidores de hyperparametros
class HyperParameterModel(BaseModel):
    pass
