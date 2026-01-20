import shutil
import subprocess
from pathlib import Path
import pathlib


def see_model(database: pathlib.Path, folder: pathlib.Path):
    subprocess.Popen(
        [
            "mlflow",
            "ui",
            "--backend-store-uri",
            f"sqlite:///{database}",
            "--default-artifact-root",
            folder,
            "--host",
            "127.0.0.1",
            "--port",
            "5000",
        ]
    )


if __name__ == "__main__":
    PATH_RES_ZIPADO = Path(
        "C:\\Users\\LUIS_FELIPE\\Downloads\\resultado_kaggle_stroke_1.zip"
    )
    DIR = Path(Path.cwd(), PATH_RES_ZIPADO.name.replace(".zip", ""))
    print(f"DIR: {DIR}")
    if DIR.exists():
        shutil.rmtree(DIR)
    DIR.mkdir()
    shutil.unpack_archive(PATH_RES_ZIPADO, DIR)

    print("COMECANDO SUBPROCESSO!\n")
    see_model(DIR / "mlflow.db", DIR / "mlruns")
