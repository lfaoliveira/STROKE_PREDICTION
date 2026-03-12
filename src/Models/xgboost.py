from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from xgboost import XGBClassifier

from Models.abc import ClassificationModel, HyperParameterModel, SuperKeys


class XGBoostHyperParameters(HyperParameterModel):
    """Hyperparameters for XGBoost model."""

    class Keys(SuperKeys):
        MAX_DEPTH = "max_depth"
        LEARNING_RATE = "learning_rate"
        N_ESTIMATORS = "n_estimators"
        SUBSAMPLE = "subsample"
        COLSAMPLE_BYTREE = "colsample_bytree"
        REG_ALPHA = "reg_alpha"
        REG_LAMBDA = "reg_lambda"
        MIN_CHILD_WEIGHT = "min_child_weight"

    max_depth: int = 6
    learning_rate: float = 0.1
    n_estimators: int = 100
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    min_child_weight: int = 1

    def suggest(self, values_dict: dict[str, float | int]) -> dict[str, float | int]:
        """Apply suggested hyperparameters from values_dict."""
        return {
            self.Keys.MAX_DEPTH: int(
                values_dict.get(self.Keys.MAX_DEPTH, self.max_depth)
            ),
            self.Keys.LEARNING_RATE: float(
                values_dict.get(self.Keys.LEARNING_RATE, self.learning_rate)
            ),
            self.Keys.N_ESTIMATORS: int(
                values_dict.get(self.Keys.N_ESTIMATORS, self.n_estimators)
            ),
            self.Keys.SUBSAMPLE: float(
                values_dict.get(self.Keys.SUBSAMPLE, self.subsample)
            ),
            self.Keys.COLSAMPLE_BYTREE: float(
                values_dict.get(self.Keys.COLSAMPLE_BYTREE, self.colsample_bytree)
            ),
            self.Keys.REG_ALPHA: float(
                values_dict.get(self.Keys.REG_ALPHA, self.reg_alpha)
            ),
            self.Keys.REG_LAMBDA: float(
                values_dict.get(self.Keys.REG_LAMBDA, self.reg_lambda)
            ),
            self.Keys.MIN_CHILD_WEIGHT: int(
                values_dict.get(self.Keys.MIN_CHILD_WEIGHT, self.min_child_weight)
            ),
        }

    def suggest_optuna(self, trial: Any = None) -> Dict[str, Any]:
        """Suggest hyperparameters using Optuna trial."""
        if trial is None:
            return {
                self.Keys.MAX_DEPTH: self.max_depth,
                self.Keys.LEARNING_RATE: self.learning_rate,
                self.Keys.N_ESTIMATORS: self.n_estimators,
                self.Keys.SUBSAMPLE: self.subsample,
                self.Keys.COLSAMPLE_BYTREE: self.colsample_bytree,
                self.Keys.REG_ALPHA: self.reg_alpha,
                self.Keys.REG_LAMBDA: self.reg_lambda,
                self.Keys.MIN_CHILD_WEIGHT: self.min_child_weight,
            }

        return {
            self.Keys.MAX_DEPTH: trial.suggest_int(self.Keys.MAX_DEPTH, 3, 10),
            self.Keys.LEARNING_RATE: trial.suggest_float(
                self.Keys.LEARNING_RATE, 0.01, 0.3
            ),
            self.Keys.N_ESTIMATORS: trial.suggest_int(self.Keys.N_ESTIMATORS, 50, 300),
            self.Keys.SUBSAMPLE: trial.suggest_float(self.Keys.SUBSAMPLE, 0.5, 1.0),
            self.Keys.COLSAMPLE_BYTREE: trial.suggest_float(
                self.Keys.COLSAMPLE_BYTREE, 0.5, 1.0
            ),
            self.Keys.REG_ALPHA: trial.suggest_float(self.Keys.REG_ALPHA, 0.0, 1.0),
            self.Keys.REG_LAMBDA: trial.suggest_float(self.Keys.REG_LAMBDA, 0.0, 2.0),
            self.Keys.MIN_CHILD_WEIGHT: trial.suggest_int(
                self.Keys.MIN_CHILD_WEIGHT, 1, 5
            ),
        }


class XGBoostModel(ClassificationModel):
    """XGBoost classifier wrapped in PyTorch Lightning."""

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

        # Initialize XGBoost model
        hyperparams = self.hyperparams
        self.xgb_model = XGBClassifier(  # type: ignore
            max_depth=hyperparams.get("max_depth", 6),
            learning_rate=hyperparams.get("learning_rate", 0.1),
            n_estimators=hyperparams.get("n_estimators", 100),
            subsample=hyperparams.get("subsample", 0.8),
            colsample_bytree=hyperparams.get("colsample_bytree", 0.8),
            reg_alpha=hyperparams.get("reg_alpha", 0.0),
            reg_lambda=hyperparams.get("reg_lambda", 1.0),
            min_child_weight=hyperparams.get("min_child_weight", 1),
            random_state=42,
            num_class=num_classes,
            objective="multi:softprob" if num_classes > 2 else "binary:logistic",
        )
        self.is_fitted = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - converts tensor to numpy, predicts, converts back."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before forward pass")

        x_numpy = x.cpu().detach().numpy()
        predictions = self.xgb_model.predict_proba(x_numpy)
        return torch.tensor(predictions, dtype=torch.float32, device=x.device)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step - fit XGBoost on batch."""
        data, labels = batch
        x_numpy = data.cpu().detach().numpy()
        y_numpy = labels.cpu().detach().numpy().squeeze()

        # Fit XGBoost model
        self.xgb_model.fit(x_numpy, y_numpy)
        self.is_fitted = True

        # Calculate training loss
        predictions = self.xgb_model.predict_proba(x_numpy)
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

        predictions = self.xgb_model.predict_proba(x_numpy)
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
