# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
import os
from kagglehub import KaggleDatasetAdapter
import torch
dataset_name = "fedesoriano/stroke-prediction-dataset"

# Set the path to the file you'd like to load
kagglehub.dataset_download(dataset_name)
file_path = "healthcare-dataset-stroke-data.csv"
print(f"FILE_PATH: {file_path}")

# Load the latest version
df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "fedesoriano/stroke-prediction-dataset",
    file_path,
)

print("First 5 records:", df.head())
