import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor, from_numpy
from DataProcesser.data import StrokeDataset
from Models.model import MLP
import torch
import torch.nn as nn

## AUX_VARS
BATCH_SIZE = 8
WORKERS = 1
RAND_SEED = 42
shuffle = False



def data_init():

    dataset = StrokeDataset()

    data, labels = dataset.data.to_numpy(), dataset.labels.to_numpy()
    print(f"N_CLASSES: {np.unique(labels)}")
    print(f"DATA SHAPE {data.shape}")
    dataset_not_null = type(data) is not None and type(labels) is not None
    assert dataset_not_null


    X_train, X_val, y_train, y_val = train_test_split(
        data, labels, test_size=0.2, stratify=labels, random_state=RAND_SEED
    )
    X_train: np.ndarray = X_train.astype(np.float32)
    X_val: np.ndarray = X_val.astype(np.float32)
    y_train: np.ndarray = y_train.astype(np.int64)
    y_val: np.ndarray = y_val.astype(np.int64)
   
    train_ds = TensorDataset(from_numpy(X_train), from_numpy(y_train))
    val_ds = TensorDataset(from_numpy(X_val), from_numpy(y_val))

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    INPUT_DIMS = dataset.data.shape[1]

    return train_loader, val_loader, INPUT_DIMS


def train():
    train_loader, val_loader, INPUT_DIMS = data_init()

    HIDN_DIMS = 32
    N_CLASSES = 2
    EPOCHS = 2
    N_LAYERS = 5
    try:
            
        model = MLP(INPUT_DIMS, HIDN_DIMS, N_LAYERS, N_CLASSES)
        #LAZY PASS
        lazy_input = torch.zeros(INPUT_DIMS, dtype=torch.float32) 
        print(f"LAZY SHAPE: {lazy_input.shape}")
        model(lazy_input)
        print(f"MODEL: {model}\n\n")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        params = model.parameters()
        print(f"PARAMS: {params}")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()


        for epoch in range(EPOCHS):
            print(f"EPOCH {epoch + 1}")
            for (batch_x, batch_y) in train_loader:
                logits = model(batch_x)
                print(f"LOGITS {logits.shape}")
                print(f"BATCH_Y {batch_y} {batch_y.shape}")
                loss: Tensor = criterion(logits, batch_y)
                print(f"LOSS: {loss.item()}")
                if(type(loss) is not Tensor):
                    raise Exception(f"Erro ao processar batch {epoch}")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            for batch_x, batch_y in val_loader:
                logits = model(batch_x)
                val_loss = criterion(logits)
                loss: Tensor = criterion(logits, batch_y)
                print(f"LOSS: {val_loss.item()}")
    except Exception as e:
        print(f"EXCECAO AO TREINAR: {e}")