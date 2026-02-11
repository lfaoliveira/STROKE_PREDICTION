from pathlib import Path
from matplotlib import pyplot as plt
import mlflow
import pandas as pd
import matplotlib.cm as cm
import numpy as np



# Plotting Functions
def plot_single_run(
    metrics_dict: dict[str, dict[int, float]],
    choice: str,
    run_idx: str,
    output_dir: Path,
):
    """Plot metrics for a single run."""
    loss_metrics, eval_metrics = filter_metrics(metrics_dict)

    if not loss_metrics:
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Plot loss metrics
    for metric_name in loss_metrics:
        steps = list(metrics_dict[metric_name].keys())
        values = list(metrics_dict[metric_name].values())
        axes[0].plot(steps, values, marker="o", label=metric_name)

    # Plot eval metrics
    for metric_name in eval_metrics:
        steps = list(metrics_dict[metric_name].keys())
        values = list(metrics_dict[metric_name].values())
        axes[1].plot(steps, values, marker="o", label=metric_name)

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
    plt.savefig(output_dir / f"metrics_{choice}_run_{run_idx}.png", dpi=300)
    plt.close()


def plot_all_runs_per_model(
    all_runs_metrics: dict[str, dict[str, dict[int, float]]],
    choice: str,
    output_dir: Path,
):
    """Plot all runs comparison on a single graph."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

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

    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss - All Optuna Runs")
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[0].grid(True, alpha=0.3)

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
    plt.close()


# Main Pipeline
def train_metrics(models: list, output_dir: Path, residual_analysis=True) -> pd.DataFrame:
    pass
