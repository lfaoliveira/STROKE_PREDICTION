import pandas as pd

from Models.error_model import ErrorModel


class ResultsProcesser:
    analysis_models: dict[str, ErrorModel]
    results_dict: dict[str, pd.DataFrame]

    def __init__(
        self,
    ) -> None:
        self.analysis_models = {}
        self.results_dict = {}

    def update(self, name: str, model: ErrorModel, df: pd.DataFrame):
        self.analysis_models[name] = model
        self.results_dict[name] = df