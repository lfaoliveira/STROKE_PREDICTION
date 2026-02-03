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


def plot_all_runs_per_model(
    all_runs_metrics: dict[str, dict[str, dict[int, float]]],
    choice: str,
    output_dir: Path,
):
    """
    Plots train_loss and val_loss for all Optuna runs on a single graph per model

    :param all_runs_metrics: Dictionary containing metrics for all runs
                             Format: {run_id: {metric_name: {step: value}}}
    :type all_runs_metrics: dict
    :param choice: Model choice
    :type choice: str
    :param output_dir: output dir for plots
    :type output_dir: Path
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Track colors for each run
    import matplotlib.cm as cm
    import numpy as np

    num_runs = len(all_runs_metrics)
    colors = cm.get_cmap("rainbow", num_runs)(np.linspace(0, 1, num_runs))

    for idx, (run_id, metrics) in enumerate(all_runs_metrics.items()):
        color = colors[idx]
        run_label = f"Run {idx}"

        # Plot train_loss
        if "train_loss" in metrics:
            steps = list(metrics["train_loss"].keys())
            values = list(metrics["train_loss"].values())
            axes[0].plot(
                steps,
                values,
                marker="o",
                alpha=0.7,
                color=color,
                label=run_label,
                linewidth=1.5,
            )

        # Plot val_loss
        if "val_loss" in metrics:
            steps = list(metrics["val_loss"].keys())
            values = list(metrics["val_loss"].values())
            axes[1].plot(
                steps,
                values,
                marker="s",
                alpha=0.7,
                color=color,
                label=run_label,
                linewidth=1.5,
            )

    # Configure train_loss subplot
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss - All Optuna Runs")
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[0].grid(True, alpha=0.3)

    # Configure val_loss subplot
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Validation Loss - All Optuna Runs")
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle(
        f"{choice} - All Optuna Runs Comparison", y=0.99, fontsize=10, fontweight="bold"
    )
    plt.subplots_adjust(top=0.90, right=0.85)

    output_file = output_dir / f"all_runs_comparison_{choice}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved all runs comparison plot: {output_file}")
    plt.show()
    plt.close()


def grab_values(
    axes,
    available_metrics: list[str],
    client: mlflow.MlflowClient,
    run_id: str,
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

    metrics_dict: dict[str, dict[int, float]] = {}

    loss_metrics = [m for m in available_metrics if "loss" in m.lower()]

    def is_eval_metric(name: str):
        # if f1, prec or rec is in name, returns true
        return any(sufix in name.lower() for sufix in ["f_beta", "prec", "rec", "auc"])

    eval_metrics = [m for m in available_metrics if is_eval_metric(m)]

    # Always collect metrics regardless of whether loss_metrics exist
    for metric_name in available_metrics:
        metric_history = client.get_metric_history(run_id, metric_name)
        metric_history = sorted(metric_history, key=lambda m: m.step)

        if metric_history:
            metrics_dict[metric_name] = {m.step: m.value for m in metric_history}

    # Plot only if axes are provided and loss_metrics exist
    if axes is not None and loss_metrics:
        for metric_name in loss_metrics:
            if metric_name in metrics_dict:
                current_metric_dict = metrics_dict[metric_name]
                steps, values = current_metric_dict.keys(), current_metric_dict.values()
                axes[0].plot(steps, values, marker="o", label=metric_name)

        # Plot accuracy/f1 metrics
        for metric_name in eval_metrics:
            if metric_name in metrics_dict:
                current_metric_dict = metrics_dict[metric_name]
                steps, values = current_metric_dict.keys(), current_metric_dict.values()
                axes[1].plot(steps, values, marker="o", label=metric_name)

    return metrics_dict


def train_metrics(models: list, output_dir: Path):
    """
    Generate metrics and plots for trained models

    :param models: List of model names to analyze
    :param output_dir: Directory to save plots
    :return: DataFrame with model metrics comparison
    """
    import os

    is_optuna = bool(os.environ.get("OPTUNA", False))
    all_models_metrics = []
    output_dir.mkdir(exist_ok=True)

    for choice in models:
        experiment = mlflow.get_experiment_by_name(f"stroke_{choice}_1")
        if not experiment:
            print(f"Experiment 'stroke_{choice}_1' not found")
            continue

        # Get all runs from the experiment
        runs = pd.DataFrame(
            mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        )
        assert isinstance(runs, pd.DataFrame)

        model_metrics = {"model": choice}
        all_runs_metrics_dict = {}
        client = mlflow.MlflowClient()

        # Process each run
        for idx, run_id in enumerate(runs["run_id"]):
            run = client.get_run(run_id)
            available_metrics = list(run.data.metrics.keys())

            # Collect step-wise metrics for this run
            run_metrics_dict = grab_values(None, available_metrics, client, run_id)
            all_runs_metrics_dict[run_id] = run_metrics_dict

            # Calculate metric averages
            for metric_name, steps_values in run_metrics_dict.items():
                values = list(steps_values.values())
                if values:
                    model_metrics[f"{metric_name}_avg"] = sum(values) / len(values)

            # Plot individual runs only when NOT using Optuna
            if (
                not is_optuna
                and run_metrics_dict
                and any("loss" in m.lower() for m in run_metrics_dict.keys())
            ):
                print(is_optuna)
                fig, axes = plt.subplots(2, 1, figsize=(10, 8))
                grab_values(axes, available_metrics, client, run_id)
                plot_metrics(axes, choice, str(idx), output_dir)

        # Always plot combined view for all runs
        if all_runs_metrics_dict:
            plot_all_runs_per_model(all_runs_metrics_dict, choice, output_dir)

        all_models_metrics.append(model_metrics)
        print(f"Graphs exported to: {output_dir}")

    return pd.DataFrame(all_models_metrics).set_index("model")
