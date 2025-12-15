import os

if os.path.exists("/kaggle"):
    PATH_DATASET = "/kaggle/working/PROJETO_PESS_DADOS"
    os.chdir("/kaggle/working/PROJETO_PESS_DADOS/src")
    os.environ["AMBIENTE"] = "KAGGLE"
elif os.path.exists("/content"):
    os.environ["AMBIENTE"] = "COLAB"
else:
    PATH_DATASET = os.path.abspath(".")
    os.environ["AMBIENTE"] = "LOCAL"

import mlflow
from torch.utils.data import DataLoader
from DataProcesser.data import StrokeDataset
from Models.model import MLP
import torch
from lightning import seed_everything, Trainer
from lightning.pytorch.loggers import MLFlowLogger
from mlflow.pytorch import autolog


## -----------------------------COLAR NO KAGGLE------------------
def main():
    ## ----------VARIAVEIS TREINO-----------
    RAND_SEED = 42
    seed_everything(RAND_SEED)
    BATCH_SIZE = 8
    WORKERS = 1
    EPOCHS = 2
    EXP_NAME = "stroke_1"
    RUN_NAME = None #   "stroke_teste"
    MLF_TRACK_URI = "sqlite:///mlflow.db"
    AMBIENTE = os.environ["AMBIENTE"]
    # MLF_TRACK_URI = "file:./mlruns"
    # mlflow.set_tracking_uri("file:./mlruns")
    # mlflow.set_experiment(EXP_NAME)

    ## ----------VARIAVEIS MODELO-----------
    HIDN_DIMS = 32
    N_CLASSES = 2
    N_LAYERS = 5

    dataset = StrokeDataset()
    INPUT_DIMS = dataset.data.shape[1]

    model = MLP(INPUT_DIMS, HIDN_DIMS, N_LAYERS, N_CLASSES)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        persistent_workers=True,
    )
    mlflow_logger = MLFlowLogger(experiment_name=EXP_NAME, tracking_uri=MLF_TRACK_URI, log_model=True)

    trainer = Trainer(
        max_epochs=EPOCHS,
        devices=1,
        accelerator="gpu" if AMBIENTE == "KAGGLE" else "cpu",
        enable_autolog_hparams=False,
        logger=mlflow_logger,
        enable_checkpointing=False,
    )

    autolog(log_models=True, checkpoint=True)
    with mlflow.start_run(run_name=RUN_NAME):
        # log model hyperparams to MLflow manually
        # mlflow.log_params(dict(model.hparams))
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
