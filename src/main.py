import os

if os.path.exists("/kaggle"):
    os.chdir("/kaggle/working/PROJETO_PESS_DADOS/src")

import mlflow

# import numpy as np
# from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# from torch import Tensor, from_numpy
from DataProcesser.data import StrokeDataset
from Models.model import MLP
import torch

# import torch.nn as nn
import lightning as L
from lightning import seed_everything
from lightning.pytorch.loggers import MLFlowLogger


## AUX_VARS
BATCH_SIZE = 8
WORKERS = 1
RAND_SEED = 42
shuffle = False
seed_everything(RAND_SEED)
## COLAR NO KAGGLE


# def data_init():
#     dataset = StrokeDataset()
#     data, labels = dataset.data.to_numpy(), dataset.labels.to_numpy()
#     print(f"N_CLASSES: {np.unique(labels)}")
#     print(f"DATA SHAPE {data.shape}")
#     dataset_not_null = type(data) is not None and type(labels) is not None
#     assert dataset_not_null

#     X_train, X_val, y_train, y_val = train_test_split(
#         data, labels, test_size=0.2, stratify=labels, random_state=RAND_SEED
#     )
#     X_train: np.ndarray = X_train.astype(np.float32)
#     X_val: np.ndarray = X_val.astype(np.float32)
#     y_train: np.ndarray = y_train.astype(np.int64)
#     y_val: np.ndarray = y_val.astype(np.int64)

#     train_ds = TensorDataset(from_numpy(X_train), from_numpy(y_train))
#     val_ds = TensorDataset(from_numpy(X_val), from_numpy(y_val))

#     train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
#     val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
#     INPUT_DIMS = dataset.data.shape[1]

#     return train_loader, val_loader, INPUT_DIMS


# train_loader, val_loader, INPUT_DIMS = data_init()

HIDN_DIMS = 32
N_CLASSES = 2
EPOCHS = 2
N_LAYERS = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # LAZY PASS
# lazy_input = torch.zeros(INPUT_DIMS, dtype=torch.float32, device=device)
# print(f"LAZY SHAPE: {lazy_input.shape}")
# model.to(device)
# model(lazy_input)

# print(f"MODEL: {model}\n\n")
# # params = model.parameters()
# # print(f"PARAMS: {params}")
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# criterion = nn.CrossEntropyLoss()


# for epoch in range(EPOCHS):
#     print(f"EPOCH {epoch + 1}")
#     loss_tensor = np.ndarray([BATCH_SIZE, 1])
#     for batch_x, batch_y in train_loader:
#         batch_x, batch_y = batch_x.to(device), batch_y.to(device)
#         logits = model(batch_x)
#         print(f"LOGITS {logits.shape}")
#         print(f"BATCH_Y {batch_y} {batch_y.shape}")
#         loss: Tensor = criterion(logits, batch_y)
#         if isinstance(loss, Tensor) is not True:
#             raise Exception(f"Erro ao processar batch {epoch}")

#         loss_tensor = np.append(loss_tensor, loss.numpy().item(), axis=0)
#         # print(f"LOSS: {loss.item()}")

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     for batch_x, batch_y in val_loader:
#         batch_x, batch_y = batch_x.to(device), batch_y.to(device)

#         logits = model(batch_x)
#         val_loss = criterion(logits, batch_y)
#         loss: Tensor = criterion(logits, batch_y)
#         print(f"LOSS: {val_loss.item()}")

dataset = StrokeDataset()
INPUT_DIMS = dataset.data.shape[1]
EXP_NAME = "stroke_1"
# mlflow.pytorch.autolog(checkpoint=False)
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment(EXP_NAME)


model = MLP(INPUT_DIMS, HIDN_DIMS, N_LAYERS, N_CLASSES)
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS
)

mlflow_logger = MLFlowLogger(
    experiment_name=EXP_NAME, tracking_uri="file:./mlruns"
)


trainer = L.Trainer(
    max_epochs=2,
    devices=1,
    accelerator="gpu",
    enable_autolog_hparams=True,
    logger=mlflow_logger,
)
# trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader,)


# MLflow run (wrap Lightning training)
with mlflow.start_run(run_name="stroke_1"):
    # log model hyperparams to MLflow manually
    mlflow.log_params(dict(model.hparams))

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # OPTIONAL: log best checkpoint to MLflow artifacts
    # mlflow.log_artifact(checkpoint_callback.best_model_path)

    # OPTIONAL: run test
    # trainer.test(model, datamodule=dm)
