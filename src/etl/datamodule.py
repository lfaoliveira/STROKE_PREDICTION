from lightning.pytorch.core.datamodule import LightningDataModule
from etl.dataset import StrokeDataset
from torch.utils.data import DataLoader, random_split


class StrokeDataModule(LightningDataModule):
    def __init__(self, BATCH_SIZE: int, WORKERS: int):
        super().__init__()

        self.BATCH_SIZE = BATCH_SIZE
        self.WORKERS = WORKERS
        self.input_dims = None

    # preparacao dos dados
    def prepare_data(self):
        # NOTE: cuidado para nao carregar dados pesados demais na memoria (estoura memoria da GPU!!!)
        self.dataset = StrokeDataset()

    # setup de transformacao e augmentation
    def setup(self, stage=None):
        DATA_SPLIT = [0.8, 0.2]
        data, label = self.dataset[0]
        self.input_dims = data.shape[0]
        self.stroke_train, self.stroke_val = random_split(self.dataset, DATA_SPLIT)

    def train_dataloader(self, BATCH_SIZE: int | None = None):
        train_loader = DataLoader(
            self.stroke_train,
            batch_size=self.BATCH_SIZE,
            num_workers=self.WORKERS,
            persistent_workers=True,
        )
        return train_loader

    def val_dataloader(self, BATCH_SIZE: int | None = None):
        val_loader = DataLoader(
            self.stroke_val,
            batch_size=self.BATCH_SIZE,
            num_workers=self.WORKERS,
            persistent_workers=True,
        )
        return val_loader

    def test_dataloader(self, BATCH_SIZE: int | None = None):
        """Dataloader de teste"""
        test_loader = DataLoader(
            self.stroke_val,
            batch_size=self.BATCH_SIZE,
            num_workers=self.WORKERS,
            persistent_workers=True,
        )
        return test_loader, self.stroke_val
