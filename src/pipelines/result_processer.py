import pandas as pd

from Models.error_model import ErrorModel


class ResultsProcesser:
    """ Class that stores Error Model results and predictions dfs"""
    analysis_models: dict[str, ErrorModel]
    classifier_pred_df: dict[str, pd.DataFrame] 

    def __init__(
        self,
    ) -> None:
        self.analysis_models = {}
        self.classifier_pred_df = {}

    def update(self, name: str, model: ErrorModel, df: pd.DataFrame):
        self.analysis_models[name] = model
        self.classifier_pred_df[name] = df

    def fit_predict(self, name: str, log_predict=True):
        self.analysis_models[name].fit()
        self.analysis_models[name].predict(log_predict)
