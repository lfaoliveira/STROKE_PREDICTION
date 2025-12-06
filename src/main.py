from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch import from_numpy
from DataProcesser.data import StrokeDataset

## AUX_VARS
BATCH_SIZE = 8
WORKERS = 1
RAND_SEED = 42
shuffle = False

dataset = StrokeDataset()

data, labels = dataset.data.to_numpy(), dataset.labels.to_numpy()
dataset_not_null = (type(data) is not None and type(labels) is not None) 
assert dataset_not_null

X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=RAND_SEED)

train_ds = TensorDataset(from_numpy(X_train), from_numpy(y_train))
val_ds = TensorDataset(from_numpy(X_val), from_numpy(y_val))

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

if __name__ == "__main__":
    for row, label in train_loader:
        print(row, label)
