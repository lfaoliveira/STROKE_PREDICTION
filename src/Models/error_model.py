## Model to fit on the errors of a prediction model and then be interpretable.
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from DataProcesser.dataset import ErrorModelDataset


class ErrorModel:
    def __init__(self, predictiosn_df: pd.DataFrame, **kwargs):
        super().__init__()
        # NOTE: not setting random state since PyTorch Lightning's seed_everything function already does it
        self.dataset = ErrorModelDataset(predictiosn_df)
        self.X, self.y = self.dataset.data, self.dataset.labels
        # usar parametros para overfit completo
        self.model = RandomForestClassifier(
            n_estimators=1000, max_depth=300, max_features="sqrt"
        )

    def split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

    def fit(self):
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        predictions = self.model.predict(self.X_test)
        print(classification_report(self.y_test, predictions))
