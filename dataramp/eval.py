"""Module for evaluating machine learning models using cross-validation and A/B testing.

Provides functionality for model evaluation, comparison, and statistical testing.
"""

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.base import ClassifierMixin, RegressorMixin, is_classifier, is_regressor
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Class for evaluating machine learning models."""

    def __init__(self, model: Union[ClassifierMixin, RegressorMixin]) -> None:
        """Initialize the ModelEvaluator.

        Args:
            model: The model to evaluate.
        """
        self.model = model

    def cross_validation_scores(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: Union[int, StratifiedKFold, KFold] = 5,
        scoring: Optional[Union[str, List[str]]] = None,
    ) -> Dict[str, List[float]]:
        """Perform cross-validation and return scores.

        Args:
            X: Features.
            y: Labels.
            cv: Number of cross-validation folds or a splitter object.
            scoring: Metric(s) to evaluate (default: accuracy for classifiers, r2 for regressors).

        Returns:
            Dictionary of scores for each metric.

        Raises:
            ValueError: If the input data is invalid.
            RuntimeError: If cross-validation fails.
        """
        if X.empty or y.empty:
            raise ValueError("Input data (X or y) is empty.")

        if scoring is None:
            if is_classifier(self.model):
                scoring = ["accuracy"]
            elif is_regressor(self.model):
                scoring = ["r2"]
            else:
                raise ValueError("Model must be a classifier or regressor.")
        elif isinstance(scoring, str):
            scoring = [scoring]

        try:
            scores = cross_validate(self.model, X, y, cv=cv, scoring=scoring)
            return {metric: scores[f"test_{metric}"].tolist() for metric in scoring}
        except Exception as e:
            logger.error(f"Error occurred during cross-validation: {e}")
            raise RuntimeError(f"Cross-validation failed: {e}") from e

    def cross_validation_report(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: Union[int, StratifiedKFold, KFold] = 5,
        target_names: Optional[List[str]] = None,
    ) -> str:
        """Generate a cross-validation report.

        Args:
            X: Features.
            y: Labels.
            cv: Number of cross-validation folds or a splitter object.
            target_names: Names of target classes (for classification).

        Returns:
            A string containing the cross-validation report.

        Raises:
            ValueError: If the input data is invalid or model is not a classifier.
            RuntimeError: If report generation fails.
        """
        if X.empty or y.empty:
            raise ValueError("Input data (X or y) is empty.")

        if not is_classifier(self.model):
            raise ValueError("Model must be a classifier for classification metrics.")

        try:
            y_pred = cross_val_predict(self.model, X, y, cv=cv)
            report = "Cross-Validation Classification Report:\n"
            report += classification_report(y, y_pred, target_names=target_names)
            return report
        except Exception as e:
            logger.error(f"Error generating cross-validation report: {e}")
            raise RuntimeError("Cross-validation report failed.") from e

    def confusion_matrix_report(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: Union[int, StratifiedKFold, KFold] = 5,
        labels: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Generate a confusion matrix report.

        Args:
            X: Features.
            y: Labels.
            cv: Number of cross-validation folds or a splitter object.
            labels: List of labels to include in the confusion matrix.

        Returns:
            The confusion matrix.

        Raises:
            ValueError: If the input data is invalid or model is not a classifier.
            RuntimeError: If report generation fails.
        """
        if X.empty or y.empty:
            raise ValueError("Input data (X or y) is empty.")

        if not is_classifier(self.model):
            raise ValueError("Model must be a classifier for classification metrics.")

        try:
            y_pred = cross_val_predict(self.model, X, y, cv=cv)
            cm = confusion_matrix(y, y_pred, labels=labels)
            return cm
        except Exception as e:
            logger.error(f"Error generating confusion matrix: {e}")
            raise RuntimeError("Confusion matrix report failed.") from e


def perform_ab_test(
    model_a: Union[ClassifierMixin, RegressorMixin],
    model_b: Union[ClassifierMixin, RegressorMixin],
    X: pd.DataFrame,
    y: pd.Series,
    cv: Union[int, StratifiedKFold, KFold] = 5,
    scoring: Optional[Union[str, List[str]]] = None,
) -> Dict[str, Dict[str, Union[float, List[float]]]]:
    """Perform A/B testing between two models.

    Args:
        model_a: The first model to compare.
        model_b: The second model to compare.
        X: Features.
        y: Labels.
        cv: Number of cross-validation folds or a splitter object.
        scoring: Metric(s) to evaluate (default: accuracy for classifiers, r2 for regressors).

    Returns:
        Dictionary containing A/B test results.

    Raises:
        ValueError: If input data is invalid or models are not the same type.
        RuntimeError: If A/B testing fails.
    """
    if X.empty or y.empty:
        raise ValueError("Input data (X or y) is empty.")

    if not (
        (is_classifier(model_a) and is_classifier(model_b))
        or (is_regressor(model_a) and is_regressor(model_b))
    ):
        raise ValueError("Both models must be classifiers or regressors.")

    evaluator_a = ModelEvaluator(model_a)
    evaluator_b = ModelEvaluator(model_b)

    try:
        if isinstance(cv, int):
            if is_classifier(model_a):
                cv_splitter = StratifiedKFold(n_splits=cv)
            else:
                cv_splitter = KFold(n_splits=cv)
        else:
            cv_splitter = cv

        scores_a = evaluator_a.cross_validation_scores(
            X, y, cv=cv_splitter, scoring=scoring
        )
        scores_b = evaluator_b.cross_validation_scores(
            X, y, cv=cv_splitter, scoring=scoring
        )

        results = {}
        for metric in scores_a:
            mean_a = np.mean(scores_a[metric])
            mean_b = np.mean(scores_b[metric])
            _, p_value = ttest_rel(scores_a[metric], scores_b[metric])

            results[metric] = {
                "mean_score_a": mean_a,
                "mean_score_b": mean_b,
                "p_value": p_value,
                "scores_a": scores_a[metric],
                "scores_b": scores_b[metric],
            }

        return results
    except Exception as e:
        logger.error(f"A/B testing error: {e}")
        raise RuntimeError("A/B testing failed.") from e
