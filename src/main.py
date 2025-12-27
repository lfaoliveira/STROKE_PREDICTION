import gc
from pathlib import Path

import torch
import os

if Path("/kaggle").exists():
    PATH_DATASET = Path("/kaggle/working/PROJETO_PESS_DADOS/src")
    os.chdir(PATH_DATASET)
    os.environ["AMBIENTE"] = "KAGGLE"
elif Path("/content").exists():
    PATH_DATASET = Path("/content/DELETAR")
    os.environ["AMBIENTE"] = "COLAB"
else:
    PATH_DATASET = Path.cwd()
    os.environ["AMBIENTE"] = "LOCAL"

import mlflow
from DataProcesser.data import StrokeDataset
from Models.mlp import MLP
from lightning import seed_everything, Trainer
from lightning.pytorch.loggers import MLFlowLogger
from mlflow.pytorch import autolog
import optuna
from lightning.pytorch.callbacks import EarlyStopping
# from Models.optimizer import Optimizer


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
    ## ----------VARIAVEIS TREINO-----------
    # BATCH_SIZE = 8
    cpus = os.cpu_count()
    WORKERS = cpus if cpus is not None else 1
    EPOCHS = 2
    PATIENCE = 5
    TRIALS = 1
    #### -------- VARIAVEIS DE LOGGING ------------
    EXP_NAME = "stroke_1"
    RUN_NAME: str | None = None  # noma da RUN: pode ser aleatório ou definido
    if os.environ["AMBIENTE"] == "KAGGLE":
        MLF_TRACK_URI = f"sqlite:///{PATH_DATASET}/mlflow.db"
    else:
        MLF_TRACK_URI = "sqlite:///mlflow.db"
    AMBIENTE = os.environ["AMBIENTE"]
    GPU = True if AMBIENTE in ["KAGGLE", "COLAB"] else False

    mlflow.set_tracking_uri(MLF_TRACK_URI)
    mlflow.set_experiment(EXP_NAME)
    autolog(log_models=True, checkpoint=True, exclusive=False)

    ## ----------VARIAVEIS MODELO-----------
    # HIDN_DIMS = 32
    N_CLASSES = 2
    # N_LAYERS = 5

    dataset = StrokeDataset()

    INPUT_DIMS = dataset.data.shape[1]

    # loop principal de treinamento
    def objective(trial: optuna.Trial):
        # Suggest hyperparameters
        hidn_dims = trial.suggest_int("hidn_dims", 16, 128, step=16)
        n_layers = trial.suggest_int("n_layers", 2, 6)
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])

        # Recreate dataloaders with trial batch_size
        train_loader, val_loader = dataset.create_dataloaders(batch_size, WORKERS)
        hyperparameters = {
            # Taxa de aprendizado (escala logarítmica)
            "lr": trial.suggest_float("lr0", 1e-5, 1e-2, log=True),
            # Momentum do otimizador
            "beta0": trial.suggest_float("momentum", 0.900, 0.9999),
            "beta1": trial.suggest_float("momentum", 0.900, 0.9999),
            # Weight decay (regularização L2)
            "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-2),
            # Épocas de warmup
            "warmup_epochs": trial.suggest_float("warmup_epochs", 0.0, 5.0),
        }
        model = MLP(
            INPUT_DIMS, hidn_dims, n_layers, N_CLASSES, hyperparameters=hyperparameters
        )
        _ = model(model.example_input_array)

        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True) as run:
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
                devices=-1 if GPU else 1,
                accelerator="gpu" if GPU else "cpu",
                logger=mlflow_logger,
                enable_checkpointing=False,
                callbacks=[early_stopping],
            )

            trainer.fit(
                model, train_dataloaders=train_loader, val_dataloaders=val_loader
            )
            mlflow.log_params(trial.params)

            val_loss = trainer.callback_metrics["val_loss"].item()
            torch.cuda.empty_cache()
            gc.collect()

            return val_loss

    with mlflow.start_run(run_name=RUN_NAME) as parent_run:
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=TRIALS)

        # Log best parameters
        mlflow.log_params(
            {"best_" + k: v for k, v in study.best_trial.params.items()},
            run_id=parent_run.info.run_id,
        )
        mlflow.log_metric(
            "best_val_loss", study.best_trial.value or 0, run_id=parent_run.info.run_id
        )

        print("Best hyperparameters:", study.best_trial.params)
        print("Best validation loss:", study.best_trial.value)
        # torch.cuda.empty_cache()
        # return trainer.callback_metrics["val_loss"].item()

    NAME_RESZIP = f"resultado_kaggle_{EXP_NAME}"
    MLRUNS_FOLDER = PATH_DATASET / "mlruns"
    zip_res(MLF_TRACK_URI, MLRUNS_FOLDER, NAME_RESZIP)
    print("\n", "=" * 60)
    print(f"RESULTADOS ZIPADOS {Path(NAME_RESZIP).resolve()}")
    print("=" * 60, "\n")
    return


if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    main()
    if os.environ["AMBIENTE"] == "LOCAL":
        from visualyze import see_model

        see_model(PATH_DATASET / "mlflow.db", PATH_DATASET / ".." / "mlruns")
