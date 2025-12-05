from torch.utils.data import Dataset
import pandas as pd
from  kagglehub import KaggleDatasetAdapter, dataset_download, dataset_load

LABELS_COLUMN = "stroke"

class StrokeDataset(Dataset):
    def __init__(self, data: pd.DataFrame, labels: pd.Series[int | str]) -> None:
        super().__init__()

        dataset_name = "fedesoriano/stroke-prediction-dataset"
        dataset_download(dataset_name)

        # Set the path to the file you'd like to load
        file_path = "healthcare-dataset-stroke-data.csv"
        print(f"FILE_PATH: {file_path}")

        # Load the latest version
        self.data: pd.DataFrame = dataset_load(
            KaggleDatasetAdapter.PANDAS,
            "fedesoriano/stroke-prediction-dataset",
            file_path,
        )
        assert isinstance(self.data, pd.DataFrame)
        
        print("First 5 records:", self.data.head())
        self.labels = self.data.xs(LABELS_COLUMN)

    def __getitem__(self, index: pd.Index[int]):
        return {"image": self.data.loc[pd.Index(index)].to_numpy(), "label": self.labels[index]}

    def __len__(self):
        return len(self.data)

    # funcao para preparacao de dados, caso seja necessario
    def data_prep(self) -> None:
        pass




class DataMappping:
    def __init__(self, data) -> None:
        pass