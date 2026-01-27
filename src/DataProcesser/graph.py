from pathlib import Path
from matplotlib import pyplot as plt
import mlflow
import pandas as pd


def plot_metrics(axes, choice: str, run_id: str, output_dir: Path):
    """
    PLots metrics into axes

    :param axes: Plot Axes
    :param choice: Model choice
    :type choice: str
    :param run_id: run_id
    :type run_id: str
    :param output_dir: output dir for plots
    :type output_dir: Path
    """
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Metric Value")
    axes[1].set_title("Training and Validation Metrics")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.suptitle(f"{choice} Experiment", y=0.98)
    plt.savefig(output_dir / f"metrics_{choice}_run_{run_id[:10]}.png", dpi=300)
    plt.show()
    plt.close()


def grab_values(
    axes, available_metrics: list, client: mlflow.MlflowClient, run_id: str
):
    """
    Iterates over available metrics, sorts them and plots into axes

    :param axes: Plot Axes
    :param available_metrics: metrics list from mlflow
    :type available_metrics: list
    :param client: MLFlow Client
    :type client: mlflow.MlflowClient
    :param run_id: Description
    :type run_id: str
    """

    loss_metrics = [m for m in available_metrics if "loss" in m.lower()]
    if not loss_metrics:
        skip = True
        return skip

    for metric_name in loss_metrics:
        metric_history = client.get_metric_history(run_id, metric_name)
        metric_history = sorted(metric_history, key=lambda m: m.step)

        if metric_history:
            steps = [m.step for m in metric_history]
            values = [m.value for m in metric_history]

            # for step, value in zip(steps, values):
            #     print(f"{metric_name}: {step}: {value}")
            axes[0].plot(steps, values, marker="o", label=metric_name)

    # Plot accuracy/f1 metrics
    for metric_name in available_metrics:
        metric_history = client.get_metric_history(run_id, metric_name)
        metric_history = sorted(metric_history, key=lambda m: m.step)

        if metric_history and any(
            x in metric_name.lower() for x in ["f1", "prec", "rec"]
        ):
            steps = [m.step for m in metric_history]
            values = [m.value for m in metric_history]

            # for step, value in zip(steps, values):
            #     print(f"{metric_name}: {step}: {value}")
            axes[1].plot(steps, values, marker="o", label=metric_name)


def train_metrics(models: list, output_dir: Path):
    import os

    # wether to plot only the best training
    plot_best = bool(os.environ.get("OPTUNA", None))
    for choice in models:
        # Get the most recent experiment
        experiment = mlflow.get_experiment_by_name(f"stroke_{choice}_1")
        if experiment:
            # Get all runs from the experiment
            
            runs = pd.DataFrame(
                mlflow.search_runs(experiment_ids=[experiment.experiment_id])
            )
            assert isinstance(runs, pd.DataFrame)
            if plot_best:
                prefix = os.environ["OPTUNA_BEST_RUN_PREFIX"]
                run_name = f"{prefix}_{choice}"
                runs = runs[runs["tags.mlflow.runName"] == run_name]
                assert len(runs) == 1

            output_dir.mkdir(exist_ok=True)

            # Plot metrics for each run
            for idx, run_id in enumerate(runs["run_id"]):
                client = mlflow.MlflowClient()
                # Get all available metrics for this run
                run = client.get_run(run_id)
                available_metrics = list(run.data.metrics.keys())
                # print(f"Available metrics for run {idx}: {available_metrics}")
                fig, axes = plt.subplots(2, 1, figsize=(10, 8))

                # Plot loss metrics
                skip = grab_values(axes, available_metrics, client, run_id)
                # skips if theres nothing to plot
                if skip:
                    continue

                plot_metrics(axes, choice, str(idx), output_dir)

            print(f"Graphs exported to: {output_dir}")
        else:
            print(f"Experiment 'stroke_{choice}_1' not found")
