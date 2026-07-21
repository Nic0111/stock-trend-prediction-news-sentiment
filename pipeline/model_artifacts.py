from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

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
    threshold_constraints_passed: bool = True
    feature_reference_values: Dict[str, float] | None = None

    def _select_input(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [column for column in self.input_columns if column not in X.columns]
        if missing:
            raise ValueError(f"Prediction input is missing columns: {missing}")
        return X[list(self.input_columns)].copy()

    def _predict_estimator_proba(self, selected: pd.DataFrame) -> np.ndarray:
        """Predict while preserving fitted transformed feature names.

        Older LightGBM artifacts were fitted on a NumPy array emitted by a
        scikit-learn ColumnTransformer, while LightGBM retained synthetic
        names such as ``Column_0``. Passing the array back at inference emits
        a misleading feature-name warning. Reconstructing a DataFrame with
        the classifier's fitted names preserves the exact values and removes
        the warning without changing the saved model.
        """
        estimator = self.estimator
        named_steps = getattr(estimator, "named_steps", None)
        if not named_steps:
            return estimator.predict_proba(selected)[:, 1]

        preprocessor = named_steps.get("preprocess")
        classifier = named_steps.get("classifier")
        fitted_names = getattr(classifier, "feature_names_in_", None)
        if preprocessor is None or classifier is None or fitted_names is None:
            return estimator.predict_proba(selected)[:, 1]

        transformed = preprocessor.transform(selected)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        if isinstance(transformed, pd.DataFrame):
            transformed_frame = transformed
        else:
            names = list(fitted_names)
            if getattr(transformed, "ndim", 0) != 2 or transformed.shape[1] != len(names):
                return estimator.predict_proba(selected)[:, 1]
            transformed_frame = pd.DataFrame(transformed, columns=names, index=selected.index)
        return classifier.predict_proba(transformed_frame)[:, 1]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        selected = self._select_input(X)
        raw_up_probability = self._predict_estimator_proba(selected)
        calibrated_up = self.calibrator.predict_proba(
            probability_to_logit(raw_up_probability)
        )[:, 1]
        return np.column_stack([1 - calibrated_up, calibrated_up])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)

    def local_feature_contributions(
        self,
        X: pd.DataFrame,
        top_n: int = 6,
    ) -> List[Dict[str, float | str]]:
        """Estimate local model sensitivity by replacing one feature at a time.

        The values are probability differences relative to development medians.
        They explain model behavior, not causal effects on the future stock price.
        """
        selected = self._select_input(X)
        if len(selected) != 1:
            raise ValueError("Local contributions require exactly one prediction row")
        references = getattr(self, "feature_reference_values", None) or {}
        if not references:
            return []

        base_probability = float(self.predict_proba(selected)[0, 1])
        contributions: List[Dict[str, float | str]] = []
        for feature_name, reference_value in references.items():
            if feature_name not in selected.columns or pd.isna(reference_value):
                continue
            current_value = pd.to_numeric(selected[feature_name], errors="coerce").iloc[0]
            if pd.isna(current_value):
                continue
            perturbed = selected.copy()
            perturbed[feature_name] = float(reference_value)
            reference_probability = float(self.predict_proba(perturbed)[0, 1])
            contributions.append(
                {
                    "feature": feature_name,
                    "contribution": base_probability - reference_probability,
                    "current_value": float(current_value),
                    "reference_value": float(reference_value),
                }
            )
        return sorted(
            contributions,
            key=lambda item: abs(float(item["contribution"])),
            reverse=True,
        )[:top_n]
