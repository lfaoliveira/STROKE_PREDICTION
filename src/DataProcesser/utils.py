from pathlib import Path
from typing import Any, Literal
import mlflow
import pandas as pd

from view.graph import plot_all_runs_per_model, plot_single_run
from pipelines.result_processer import ResultsProcesser
from Models.error_model import ErrorModel


# Data Processing Functions
def _grab_values(
    available_metrics: list[str],
    client: mlflow.MlflowClient,
    run_id: str,
) -> dict[str, dict[int, float]]:
    """
    Collects metrics from MLflow without plotting.

    :param available_metrics: metrics list from mlflow
    :param client: MLFlow Client
    :param run_id: run ID
    :return: Dictionary with metrics data
    """
    metrics_dict: dict[str, dict[int, float]] = {}

    for metric_name in available_metrics:
        metric_history = client.get_metric_history(run_id, metric_name)
        metric_history = sorted(metric_history, key=lambda m: m.step)

        if metric_history:
            metrics_dict[metric_name] = {m.step: m.value for m in metric_history}

    return metrics_dict


def _calculate_metric_averages(
    metrics_dict: dict[str, dict[int, float]],
) -> dict[str, float]:
    """Calculate average values for all metrics."""
    averages = {}
    for metric_name, steps_values in metrics_dict.items():
        values = list(steps_values.values())
        if values:
            averages[f"{metric_name}_avg"] = sum(values) / len(values)
    return averages


def residual_analysis(
    client: mlflow.MlflowClient,
    best_run: Any,
    name: str,
    processer: ResultsProcesser,
):
    # Get artifacts from the run and search for test_results_{run_id}.csv
    artifacts = client.list_artifacts(best_run.run_id)
    print(f"ARTIFACT LIST: {artifacts}")
    test_results_file = f"test_results_{best_run.run_id}.csv"
    artifact_found = any(artifact.path == test_results_file for artifact in artifacts)

    if not artifact_found:
        print(f"Artifact '{test_results_file}' not found in run {best_run.run_id}")
        raise Exception("ARTIFACT NOT FOUND!")

    # Pass the loaded model to your analysis function
    df_path = mlflow.artifacts.download_artifacts(
        artifact_path=test_results_file, run_id=best_run.run_id
    )

    prediction_df = pd.read_csv(df_path)
    error_model = ErrorModel(prediction_df)
    processer.update(name, error_model, prediction_df)


def final_analysis(
    models: list,
    output_dir: Path,
    sort_metric: Literal["val_f_beta_avg", "val_f1_avg", "val_loss_avg"],
    residual=True,
) -> tuple[pd.DataFrame, ResultsProcesser]:
    """Generate metrics and plots for trained models. residual indicates wether to store basic residual analysis information for later use"""
    import os

    is_optuna = bool(os.environ.get("OPTUNA", False))
    all_models_metrics = []
    output_dir.mkdir(exist_ok=True)
    client = mlflow.MlflowClient()

    processer = ResultsProcesser()

    for choice in models:
        experiment = mlflow.get_experiment_by_name(f"stroke_{choice}_1")
        if not experiment:
            print(f"Experiment 'stroke_{choice}_1' not found")
            continue

        runs = pd.DataFrame(
            mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        )
        assert isinstance(runs, pd.DataFrame)

        model_metrics = {"model": choice}
        all_runs_metrics_dict = {}

        # Process each run
        for idx, run_id in enumerate(runs["run_id"]):
            run = client.get_run(run_id)
            available_metrics = list(run.data.metrics.keys())
            # Collect metrics
            run_metrics_dict = _grab_values(available_metrics, client, run_id)
            all_runs_metrics_dict[run_id] = run_metrics_dict
            # Calculate averages
            averages = _calculate_metric_averages(run_metrics_dict)

            # Inject averages into the runs DataFrame
            for metric_name, avg_value in averages.items():
                runs.loc[runs["run_id"] == run_id, f"metrics.{metric_name}"] = avg_value

            model_metrics.update(averages)

            # Plot individual runs only when NOT using Optuna
            if not is_optuna and run_metrics_dict:
                plot_single_run(run_metrics_dict, choice, str(idx), output_dir)

        # armazena modelos de erro e dataframe
        if residual and not runs.empty:
            ascending = "loss" in sort_metric
            best_run = runs.sort_values(
                f"metrics.{sort_metric}", ascending=ascending
            ).iloc[0]
            residual_analysis(
                client,
                best_run,
                choice,
                processer,
            )

        # Always plot combined view
        if all_runs_metrics_dict:
            plot_all_runs_per_model(all_runs_metrics_dict, choice, output_dir)

        all_models_metrics.append(model_metrics)
        print(f"Graphs exported to: {output_dir}")

    return pd.DataFrame(all_models_metrics).set_index("model"), processer
