from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd


def probability_to_logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


@dataclass
class CalibratedStockModel:
    estimator: Any
    calibrator: Any
    threshold: float
    input_columns: Sequence[str]
    numeric_feature_columns: Sequence[str]
    refinement_round: int
    model_name: str
    horizon_days: int

    def _select_input(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [column for column in self.input_columns if column not in X.columns]
        if missing:
            raise ValueError(f"Prediction input is missing columns: {missing}")
        return X[list(self.input_columns)].copy()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        selected = self._select_input(X)
        raw_up_probability = self.estimator.predict_proba(selected)[:, 1]
        calibrated_up = self.calibrator.predict_proba(
            probability_to_logit(raw_up_probability)
        )[:, 1]
        return np.column_stack([1 - calibrated_up, calibrated_up])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)
