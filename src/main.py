from pathlib import Path
import os
from typing import Literal


if Path("/kaggle").exists():
    PATH_DATASET = Path("/kaggle/working/PROJETO_PESS_DADOS")
    PATH_CODE = PATH_DATASET / "src"
    os.chdir(PATH_CODE)
    os.environ["AMBIENTE"] = "KAGGLE"
elif Path("/content").exists():
    PATH_DATASET = Path("/content/DELETAR")
    os.environ["AMBIENTE"] = "COLAB"
else:
    PATH_CODE = Path.cwd()  # Already in src directory
    PATH_DATASET = PATH_CODE.parent  # Go up to PROJETO_PESS_DADOS
    os.environ["AMBIENTE"] = "LOCAL"
import gc
import mlflow
from Models.mlp import MLP
from Models.kan import MyKan
from lightning import seed_everything, Trainer
from lightning.pytorch.loggers import MLFlowLogger
from mlflow.pytorch import autolog
from lightning.pytorch.callbacks import EarlyStopping
from DataProcesser.datamodule import StrokeDataModule


def zip_res(path_sqlite: str, path_mlflow: Path, filename: str):
    import shutil

    path_sqlite_clean = path_sqlite.replace("sqlite:///", "")
    print(f"CWD: {Path.cwd()}\n")
    PATH_TEMP = Path.cwd() / "ZIP_TEMP"
    shutil.rmtree(PATH_TEMP, ignore_errors=True)
    PATH_TEMP.mkdir(parents=True, exist_ok=True)

    shutil.copy(path_sqlite_clean, PATH_TEMP / Path(path_sqlite_clean).name)
    shutil.copytree(path_mlflow, PATH_TEMP / path_mlflow.name)

    shutil.make_archive(filename.replace(".zip", ""), "zip", PATH_TEMP)
    shutil.rmtree(PATH_TEMP)
    print(f"PATH ZIPFILE: {Path(filename).resolve()}")


## -----------------------------COLAR NO KAGGLE------------------
def main():
    ###------SEEDS---------###
    RAND_SEED = 42
    seed_everything(RAND_SEED)
    AMBIENTE = os.environ["AMBIENTE"]
    GPU = True if AMBIENTE in ["KAGGLE", "COLAB"] else False
    ## ----------VARIAVEIS TREINO-----------
    cpus = os.cpu_count()
    WORKERS = cpus if cpus is not None else 1
    NUM_DEVICES = 1 if GPU else 1
    NUM_NODES = 1
    BATCH_SIZE = 16
    EPOCHS = 50
    PATIENCE = 20
    CHOICE: Literal["MLP", "KAN", "SVM", "XGBOOST"] = (
        "MLP"  ## ESCOLHA DE MODELO A SER USADO
    )
    #### -------- VARIAVEIS DE LOGGING ------------
    EXP_NAME = f"stroke_{CHOICE}_1"
    RUN_NAME: str | None = None  # nome da RUN: pode ser aleatório ou definido
    MLF_TRACK_URI = f"sqlite:///{PATH_CODE}/mlflow.db"

    mlflow.set_tracking_uri(MLF_TRACK_URI)
    mlflow.set_experiment(EXP_NAME)
    autolog(log_models=True, checkpoint=True, exclusive=False)

    ## ----------VARIAVEIS MODELO-----------
    HIDN_DIMS = 32
    N_CLASSES = 2
    N_LAYERS = 5

    datamodule = StrokeDataModule(BATCH_SIZE, WORKERS)

    datamodule.prepare_data()
    datamodule.setup("fit")

    INPUT_DIMS = datamodule.input_dims or -1
    assert INPUT_DIMS > 0
    if CHOICE == "MLP":
        model = MLP(INPUT_DIMS, HIDN_DIMS, N_LAYERS, N_CLASSES)
    elif CHOICE == "KAN":
        model = MyKan(INPUT_DIMS, HIDN_DIMS, N_LAYERS, N_CLASSES)
    else:
        raise ValueError("ESCOLHA DE MODELO ERRADA!")
    print(model)
    _ = model(model.example_input_array)

    # loop principal de treinamento
    with mlflow.start_run(run_name=RUN_NAME) as run:
        active_run_id = run.info.run_id

        mlflow_logger = MLFlowLogger(
            experiment_name=EXP_NAME,
            tracking_uri=MLF_TRACK_URI,
            log_model=True,
            run_id=active_run_id,
        )

        early_stopping = EarlyStopping(
            monitor="val_loss", patience=PATIENCE, mode="min"
        )

        trainer = Trainer(
            max_epochs=EPOCHS,
            devices=NUM_DEVICES,
            accelerator="gpu" if GPU else "cpu",
            num_nodes=NUM_NODES,
            logger=mlflow_logger,
            enable_checkpointing=False,
            callbacks=[early_stopping],
        )
        trainer.fit(model, datamodule=datamodule)
        mlflow.log_params(dict(model.hparams))

    NAME_RESZIP = f"resultado_kaggle_{EXP_NAME}"
    MLRUNS_FOLDER = Path.cwd() / "mlruns"
    zip_res(MLF_TRACK_URI, MLRUNS_FOLDER, NAME_RESZIP)
    print("\n", "=" * 60)
    print(f"RESULTADOS ZIPADOS {Path(NAME_RESZIP).resolve()}")
    print("=" * 60, "\n")
    return


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        raise e
    gc.collect()

    if os.environ["AMBIENTE"] == "LOCAL":
        from visualyze import see_model

        see_model(PATH_DATASET / "mlflow.db", PATH_DATASET / ".." / "mlruns")
