from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier

from Models.abc import ClassificationModel, HyperParameterModel, SuperKeys


class RandomForestHyperParameters(HyperParameterModel):
    """Hyperparameters for Random Forest model."""

    class Keys(SuperKeys):
        N_ESTIMATORS = "n_estimators"
        MAX_DEPTH = "max_depth"
        MIN_SAMPLES_SPLIT = "min_samples_split"
        MIN_SAMPLES_LEAF = "min_samples_leaf"
        MAX_FEATURES = "max_features"

    n_estimators: int = 100
    max_depth: int = 15
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: str = "sqrt"
    bootstrap: bool = True
    random_state: int = 42

    def suggest(
        self, values_dict: dict[str, float | int]
    ) -> dict[str, float | int | str]:
        """Apply suggested hyperparameters from values_dict."""
        return {
            self.Keys.N_ESTIMATORS: int(
                values_dict.get(self.Keys.N_ESTIMATORS, self.n_estimators)
            ),
            self.Keys.MAX_DEPTH: int(
                values_dict.get(self.Keys.MAX_DEPTH, self.max_depth)
            ),
            self.Keys.MIN_SAMPLES_SPLIT: int(
                values_dict.get(self.Keys.MIN_SAMPLES_SPLIT, self.min_samples_split)
            ),
            self.Keys.MIN_SAMPLES_LEAF: int(
                values_dict.get(self.Keys.MIN_SAMPLES_LEAF, self.min_samples_leaf)
            ),
            self.Keys.MAX_FEATURES: str(
                values_dict.get(self.Keys.MAX_FEATURES, self.max_features)
            ),
        }

    def suggest_optuna(self, trial: Any = None) -> Dict[str, Any]:
        """Suggest hyperparameters using Optuna trial."""
        if trial is None:
            return {
                self.Keys.N_ESTIMATORS: self.n_estimators,
                self.Keys.MAX_DEPTH: self.max_depth,
                self.Keys.MIN_SAMPLES_SPLIT: self.min_samples_split,
                self.Keys.MIN_SAMPLES_LEAF: self.min_samples_leaf,
                self.Keys.MAX_FEATURES: self.max_features,
            }

        return {
            self.Keys.N_ESTIMATORS: trial.suggest_int(self.Keys.N_ESTIMATORS, 50, 300),
            self.Keys.MAX_DEPTH: trial.suggest_int(self.Keys.MAX_DEPTH, 5, 30),
            self.Keys.MIN_SAMPLES_SPLIT: trial.suggest_int(
                self.Keys.MIN_SAMPLES_SPLIT, 2, 10
            ),
            self.Keys.MIN_SAMPLES_LEAF: trial.suggest_int(
                self.Keys.MIN_SAMPLES_LEAF, 1, 5
            ),
            self.Keys.MAX_FEATURES: trial.suggest_categorical(
                self.Keys.MAX_FEATURES, ["sqrt", "log2"]
            ),
        }


class RandomForestModel(ClassificationModel):
    """Random Forest classifier wrapped in PyTorch Lightning."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        recall_factor: List[float],
        **kwargs: Any,
    ) -> None:
        super().__init__(input_dim, num_classes, recall_factor, **kwargs)

        self.num_classes = num_classes
        self.input_dim = input_dim

        # Initialize Random Forest model
        hyperparams = self.hyperparams
        self.rf_model = RandomForestClassifier(
            n_estimators=hyperparams.get("n_estimators", 100),
            max_depth=hyperparams.get("max_depth", 15),
            min_samples_split=hyperparams.get("min_samples_split", 2),
            min_samples_leaf=hyperparams.get("min_samples_leaf", 1),
            max_features=hyperparams.get("max_features", "sqrt"),
            bootstrap=hyperparams.get("bootstrap", True),
            random_state=hyperparams.get("random_state", 42),
            n_jobs=-1,
        )
        self.is_fitted = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - converts tensor to numpy, predicts, converts back."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before forward pass")

        x_numpy = x.cpu().detach().numpy()
        predictions = self.rf_model.predict_proba(x_numpy)
        return torch.tensor(predictions, dtype=torch.float32, device=x.device)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step - fit Random Forest on batch."""

        data, labels = batch
        x_numpy = data.cpu().detach().numpy()
        y_numpy = labels.cpu().detach().numpy().squeeze()

        # Fit Random Forest model
        self.rf_model.fit(x_numpy, y_numpy)
        self.is_fitted = True

        # Calculate training loss
        predictions = self.rf_model.predict_proba(x_numpy)
        loss = self._calculate_loss(predictions, y_numpy)

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return torch.tensor(loss, dtype=torch.float32)

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Validation step."""
        if not self.is_fitted:
            return torch.tensor(0.0)

        data, labels = batch
        x_numpy = data.cpu().detach().numpy()
        y_numpy = labels.cpu().detach().numpy().squeeze().astype(int)

        predictions = self.rf_model.predict_proba(x_numpy)
        loss = self._calculate_loss(predictions, y_numpy)

        preds_tensor = torch.tensor(
            predictions.argmax(axis=1), dtype=torch.long, device=labels.device
        )
        self.val_metrics.update(preds_tensor, labels)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=False)

        return torch.tensor(loss, dtype=torch.float32)

    def _calculate_loss(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """Calculate cross-entropy loss."""
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        ce_loss = -np.mean(
            np.log(predictions[np.arange(len(labels)), labels.astype(int)])
        )
        return float(ce_loss)
