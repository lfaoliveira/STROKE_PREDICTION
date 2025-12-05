from torch.utils.data import DataLoader

from DataProcesser.data import StrokeDataset

## AUX_VARS
BATCH_SIZE = 8
WORKERS = 4
shuffle = False

dataset = StrokeDataset()

dataloader = DataLoader(dataset,batch_size=BATCH_SIZE, num_workers=WORKERS)




for row, label in dataloader:
    print(row, label)