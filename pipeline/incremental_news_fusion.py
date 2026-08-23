"""Anchored, exact-fallback stock/news probability fusion.

The module deliberately exposes one small interface: fit an incremental news
correction and predict from a stock probability plus point-in-time news
evidence.  The stock log-odds coefficient is fixed at one.  Consequently the
module cannot relearn or silently replace the stock model, and observations
without eligible news are mathematically identical to stock-only inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize


EPSILON = 1e-6


def _sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(value, dtype=float), -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _logit(probability: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(probability, dtype=float), EPSILON, 1.0 - EPSILON)
    return np.log(value / (1.0 - value))


@dataclass(frozen=True)
class FusionPrediction:
    """Observable result returned by :class:`IncrementalNewsFusion`."""

    probability_up: np.ndarray
    news_adjustment_log_odds: np.ndarray
    news_adjustment_probability_points: np.ndarray
    eligible_news: np.ndarray


class IncrementalNewsFusion:
    """Learn only a regularised news correction around frozen stock odds.

    Invariants
    ----------
    * ``stock_probability_up`` must be present at fit and prediction time.
    * ``news_available <= 0`` produces exactly the stock probability.
    * the learned log-odds correction is capped symmetrically.
    * preprocessing statistics are learned on training rows only.
    """

    def __init__(
        self,
        feature_names: Sequence[str],
        *,
        penalty_strength: float = 1.0,
        l1_ratio: float = 0.0,
        adjustment_cap: float = 0.5,
        random_seed: int = 42,
        balance_classes: bool = False,
    ) -> None:
        self.feature_names = tuple(str(name) for name in feature_names)
        self.penalty_strength = float(penalty_strength)
        self.l1_ratio = float(l1_ratio)
        self.adjustment_cap = float(adjustment_cap)
        self.random_seed = int(random_seed)
        self.balance_classes = bool(balance_classes)
        if not self.feature_names:
            raise ValueError("At least one incremental news feature is required")
        if self.penalty_strength < 0:
            raise ValueError("penalty_strength must be non-negative")
        if not 0.0 <= self.l1_ratio <= 1.0:
            raise ValueError("l1_ratio must be between zero and one")
        if self.adjustment_cap <= 0:
            raise ValueError("adjustment_cap must be positive")
        self.location_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.coefficients_: np.ndarray | None = None
        self.fit_diagnostics_: dict[str, Any] = {}

    @staticmethod
    def _available(frame: pd.DataFrame) -> np.ndarray:
        if "news_available" not in frame:
            return np.zeros(len(frame), dtype=bool)
        values = pd.to_numeric(frame["news_available"], errors="coerce").fillna(0.0)
        return values.to_numpy(dtype=float) > 0.0

    def _matrix(self, frame: pd.DataFrame, *, fitting: bool) -> np.ndarray:
        columns = []
        for name in self.feature_names:
            values = (
                pd.to_numeric(frame[name], errors="coerce")
                if name in frame
                else pd.Series(0.0, index=frame.index)
            )
            columns.append(values.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float))
        matrix = np.column_stack(columns)
        if fitting:
            available = self._available(frame)
            reference = matrix[available] if available.any() else matrix
            self.location_ = np.nanmedian(reference, axis=0)
            q75 = np.nanpercentile(reference, 75, axis=0)
            q25 = np.nanpercentile(reference, 25, axis=0)
            self.scale_ = np.where((q75 - q25) > 1e-9, q75 - q25, 1.0)
        if self.location_ is None or self.scale_ is None:
            raise RuntimeError("IncrementalNewsFusion has not been fitted")
        return (matrix - self.location_) / self.scale_

    def fit(self, frame: pd.DataFrame, target_up: Sequence[int]) -> "IncrementalNewsFusion":
        if "stock_probability_up" not in frame:
            raise ValueError("stock_probability_up is required")
        target = np.asarray(target_up, dtype=float)
        if len(target) != len(frame):
            raise ValueError("target_up length must match the feature frame")
        available = self._available(frame)
        if int(available.sum()) < max(10, len(self.feature_names) + 2):
            raise ValueError("Insufficient news-present observations for anchored fusion")
        matrix = self._matrix(frame, fitting=True)[available]
        target = target[available]
        if np.unique(target.astype(int)).size < 2:
            raise ValueError(
                "Anchored fusion requires both Up and Down targets among news-present rows"
            )
        stock_logit = _logit(
            pd.to_numeric(frame.loc[available, "stock_probability_up"], errors="coerce")
            .fillna(0.5)
            .to_numpy(dtype=float)
        )
        if self.balance_classes:
            class_counts = np.bincount(target.astype(int), minlength=2).astype(float)
            class_weights = np.where(
                class_counts > 0, len(target) / (2.0 * class_counts), 1.0
            )
            sample_weight = class_weights[target.astype(int)]
        else:
            # Fusion is evaluated as a probability forecast (Brier/log loss),
            # not as a class-balanced classifier.  Weighting classes here
            # would alter the base rate and manufacture an intercept shift.
            sample_weight = np.ones(len(target), dtype=float)
        l1 = self.penalty_strength * self.l1_ratio
        l2 = self.penalty_strength * (1.0 - self.l1_ratio)

        def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
            intercept = float(parameters[0])
            beta = parameters[1:]
            raw_delta = matrix @ beta
            probability = _sigmoid(stock_logit + intercept + raw_delta)
            loss = -np.average(
                target * np.log(np.clip(probability, EPSILON, 1.0))
                + (1.0 - target) * np.log(np.clip(1.0 - probability, EPSILON, 1.0)),
                weights=sample_weight,
            )
            smooth_l1 = np.sqrt(beta * beta + 1e-8)
            penalty = l2 * 0.5 * float(beta @ beta) + l1 * float(smooth_l1.sum())
            residual = (probability - target) * sample_weight / sample_weight.mean()
            coefficient_gradient = matrix.T @ residual / len(target)
            coefficient_gradient += l2 * beta + l1 * beta / smooth_l1
            intercept_gradient = np.asarray([residual.mean()], dtype=float)
            gradient = np.concatenate([intercept_gradient, coefficient_gradient])
            return float(loss + penalty), gradient

        result = minimize(
            lambda beta: objective(beta)[0],
            np.zeros(matrix.shape[1] + 1, dtype=float),
            jac=lambda parameters: objective(parameters)[1],
            method="L-BFGS-B",
            options={"maxiter": 1000, "ftol": 1e-10},
        )
        if not result.success:
            raise RuntimeError(f"Anchored fusion optimization failed: {result.message}")
        self.intercept_ = float(result.x[0])
        self.coefficients_ = np.asarray(result.x[1:], dtype=float)
        self.fit_diagnostics_ = {
            "converged": True,
            "iterations": int(result.nit),
            "objective": float(result.fun),
            "news_present_rows": int(available.sum()),
            "feature_count": len(self.feature_names),
            "coefficient_l1_norm": float(np.abs(self.coefficients_).sum()),
            "news_present_intercept": self.intercept_,
            "balance_classes": self.balance_classes,
        }
        return self

    def predict(self, frame: pd.DataFrame) -> FusionPrediction:
        if self.coefficients_ is None or self.intercept_ is None:
            raise RuntimeError("IncrementalNewsFusion has not been fitted")
        if "stock_probability_up" not in frame:
            raise ValueError("stock_probability_up is required")
        stock = (
            pd.to_numeric(frame["stock_probability_up"], errors="coerce")
            .fillna(0.5)
            .clip(EPSILON, 1.0 - EPSILON)
            .to_numpy(dtype=float)
        )
        available = self._available(frame)
        matrix = self._matrix(frame, fitting=False)
        delta = np.clip(
            self.intercept_ + matrix @ self.coefficients_,
            -self.adjustment_cap,
            self.adjustment_cap,
        )
        delta = np.where(available, delta, 0.0)
        probability = _sigmoid(_logit(stock) + delta)
        # Assign directly rather than relying on floating-point inverse-logit
        # round trips: exact fallback is an interface invariant.
        probability[~available] = stock[~available]
        return FusionPrediction(
            probability_up=probability,
            news_adjustment_log_odds=delta,
            news_adjustment_probability_points=(probability - stock) * 100.0,
            eligible_news=available,
        )

    def provenance(self) -> Mapping[str, Any]:
        if self.coefficients_ is None or self.intercept_ is None:
            raise RuntimeError("IncrementalNewsFusion has not been fitted")
        return {
            "module": "IncrementalNewsFusion",
            "formulation": "sigmoid(logit(stock_probability)+news_delta)",
            "feature_names": list(self.feature_names),
            "penalty_strength": self.penalty_strength,
            "l1_ratio": self.l1_ratio,
            "adjustment_cap": self.adjustment_cap,
            "news_present_intercept": self.intercept_,
            "balance_classes": self.balance_classes,
            "coefficients": {
                name: float(value)
                for name, value in zip(self.feature_names, self.coefficients_)
            },
            "exact_no_news_fallback": True,
            "fit_diagnostics": dict(self.fit_diagnostics_),
        }
