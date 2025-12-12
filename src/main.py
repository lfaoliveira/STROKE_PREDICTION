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

from types import NoneType
import mlflow
from torch.utils.data import DataLoader
from DataProcesser.data import StrokeDataset
from Models.model import MLP
import torch
import lightning as L
from lightning import seed_everything
from lightning.pytorch.loggers import MLFlowLogger


def zip_res(path_sqlite: str, path_mlflow: str, filename: str):
    import shutil

    PATH_TEMP = os.path.join(os.getcwd(), "ZIP_TEMP")
    os.makedirs(PATH_TEMP, exist_ok=True)

    shutil.copy(path_sqlite, os.path.join(PATH_TEMP, path_sqlite))
    shutil.copytree(path_mlflow, os.path.join(PATH_TEMP, path_mlflow))

    shutil.make_archive(filename.replace(".zip", ""), "zip", PATH_TEMP)
    shutil.rmtree(PATH_TEMP)
    print(f"PATH ZIPFILE: {os.path.abspath(filename)}")

def main():
    ## ---------- SETTINGS -----------
    RAND_SEED = 42
    seed_everything(RAND_SEED)

    BATCH_SIZE = 8
    # WORKERS = 1 if os.environ["AMBIENTE"] == "LOCAL" else 4
    WORKERS = 4
    EPOCHS = 1
    EXP_NAME = "stroke_1"
    RUN_NAME = "stroke_teste"
    # This MUST point to your sqlite DB file
    URL_TRACKING_MLFLOW = "sqlite:///mlflow.db"
    # ARTIFACT_LOCATION = os.path.abspath("./mlruns")
    # os.environ["MLFLOW_ARTIFACT_ROOT"] = ARTIFACT_LOCATION

    # Important: This MUST be the artifact folder inside your project
    
    ## ---------- DATASET & MODEL -----------
    dataset = StrokeDataset()
    INPUT_DIMS = dataset.data.shape[1]

    model = MLP(INPUT_DIMS, 32, 5, 2)
    model(model.example_input_array)

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS
    )

    ## ---------- MLflow LOGGER ---------
    mlflow_logger = MLFlowLogger(
        experiment_name=EXP_NAME,
        tracking_uri=URL_TRACKING_MLFLOW,
        run_name=RUN_NAME,
        artifact_location="./mlruns"
    )

    ## ---------- TRAINER -----------
    trainer = L.Trainer(
        max_epochs=EPOCHS,
        logger=mlflow_logger,
        enable_autolog_hparams=True,
        devices=1,
        accelerator="cpu",
    )

    ## ---------- TRAIN ----------
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("\nTRAINING FINISHED!\n")

    test_artifact_path = "test_artifact.txt"
    with open(test_artifact_path, "w") as f:
        f.write("This is a test artifact to check MLflow save location")
    
    mlflow.log_artifact(test_artifact_path)
    os.remove(test_artifact_path)

    ## --------- EXPORT ------------
    run = mlflow.get_run(str(mlflow_logger.run_id))
    artifact_folder = str(run.info.artifact_uri)
    print("Artifacts stored at:", artifact_folder)

    
if __name__ == "__main__":
    from mlflow.tracking import MlflowClient
    for experiment in mlflow.search_experiments():
        print(experiment.name)
        if experiment.experiment_id != 0:
            mlflow.delete_experiment(experiment.experiment_id)
    #-----------mAIN---------------#
    main()
    client = MlflowClient()
    for exp in client.search_experiments():
        print(exp.experiment_id, exp.name, exp.artifact_location)
