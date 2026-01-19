from lightning import LightningModule


class ClassificationModel(LightningModule):
    def validation_step(self, batch, batch_idx):
        raise NotImplementedError("validation_step must be implemented")

    def training_step(self, batch, batch_idx):
        raise NotImplementedError("training_step must be implemented")

    def configure_optimizers(self):
        raise NotImplementedError("configure_optimizers must be implemented")

    def forward(self, x):
        raise NotImplementedError("forward must be implemented")
