from enum import StrEnum

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
# from pandas import Series, DataFrame
import pandera.pandas as pa
from pandera.typing import DataFrame, Series
from sklearn.preprocessing import StandardScaler
from torch.types import Tensor
from kagglehub import KaggleDatasetAdapter, dataset_download, dataset_load
# import pandas

LABELS_COLUMN = "stroke"

class CATEGORICAL_COLUMNS(StrEnum):
    GENDER = "gender"
    MARRIED = "ever_married"
    WORK = "work_type"
    RESIDENCE = "Residence_type"
    SMOKE_STATUS = "smoking_status"


class MySchema(pa.DataFrameModel):
    id: Series[int]
    age: Series[float]
    gender: Series[str]
    ever_married: Series[str]
    work_type: Series[str]
    Residence_type: Series[str]
    smoking_status: Series[str]
    hypertension: Series[int]
    heart_disease: Series[int]
    avg_glucose_level: Series[float]
    bmi: Series[float] 
    stroke: Series[int]



class StrokeDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

        self.read_df()
        STR_COL = list(CATEGORICAL_COLUMNS)
        ## AQUI VAI PREPARACAO DOS DADOS
        self.data_prep(STR_COL)
        print("\n")
        self.labels = self.data.loc[:, LABELS_COLUMN]
        self.data = self.data.drop(columns=LABELS_COLUMN)

    def __getitem__(self, index: Tensor | int):
        print()
        if type(index) is int:
            return Tensor(self.data.loc[index].to_numpy()), Tensor(
                [self.labels[index]]
            ) 
        elif type(index) is Tensor:
            return self.data.loc[index.tolist()], self.labels[index.tolist()].to_numpy()
        else:
            raise Exception("ERRO AO PEGAR DADOS")

    def __len__(self):
        return len(self.data)

    def read_df(self):
        dataset_name = "fedesoriano/stroke-prediction-dataset"
        dataset_download(dataset_name)
        # Set the path to the file you'd like to load
        file_path = "healthcare-dataset-stroke-data.csv"
        print(f"FILE_PATH: {file_path}")

        # Load the latest version
        self.data: DataFrame[MySchema] = dataset_load(
            KaggleDatasetAdapter.PANDAS,
            dataset_name,
            file_path,
        )
        # tira valores nulos pra evitar problemas
        self.data = self.data.dropna()
        #valida schema
        validated = MySchema.validate(self.data)
        print(f"DF NORMAL: {validated.head()}\n")
        assert type(self.data) is pd.DataFrame
        
    # funcao para preparacao de dados, caso seja necessario
    def data_prep(self, bad_columns: list[CATEGORICAL_COLUMNS]) -> None:
        STR_COL = bad_columns

        ##itera sobre conjunto da coluna e bota numero pra cada string
        for col in STR_COL:
            self.data[col] = self.data[f"{col}"].astype("category")
            self.data[f"{col}_code"] = self.data[f"{col}"].cat.codes

        self.data = self.data.drop(columns=STR_COL)
        print(f"DF DROPADO: {self.data.head()}\n")

        ## standard scaler pra normalizar dados para media 0 e desvio padrao 1
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(self.data)
        self.data = DataFrame(
            scaled_values, columns=self.data.columns, index=self.data.index
        )