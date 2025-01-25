import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_validate

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
        cv: int = 5,
        scoring: Optional[Union[str, List[str]]] = None,
    ) -> Dict[str, List[float]]:
        """Perform cross-validation and return scores.

        Args:
            X: Features.
            y: Labels.
            cv: Number of cross-validation folds.
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
            if isinstance(self.model, ClassifierMixin):
                scoring = "accuracy"
            elif isinstance(self.model, RegressorMixin):
                scoring = "r2"
            else:
                raise ValueError("Model must be a classifier or regressor.")

        try:
            scores = cross_validate(self.model, X, y, cv=cv, scoring=scoring)
            return {metric: scores[f"test_{metric}"].tolist() for metric in scoring}
        except Exception as e:
            logger.error(f"Error occurred during cross-validation: {e}")
            raise RuntimeError(f"Cross-validation failed: {e}")

    def cross_validation_report(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
        target_names: Optional[List[str]] = None,
    ) -> str:
        """Generate a cross-validation report.

        Args:
            X: Features.
            y: Labels.
            cv: Number of cross-validation folds.
            target_names: Names of target classes (for classification).

        Returns:
            A string containing the cross-validation report.

        Raises:
            ValueError: If the input data is invalid.
            RuntimeError: If the model is not a classifier.
        """
        if X.empty or y.empty:
            raise ValueError("Input data (X or y) is empty.")

        if not isinstance(self.model, ClassifierMixin):
            raise ValueError("Model must be a classifier for classification metrics.")

        try:
            cv_results = cross_validate(
                self.model,
                X,
                y,
                cv=cv,
                return_train_score=False,
                scoring=["accuracy", "precision", "recall", "f1"],
            )

            report = "Cross-Validation Metrics:\n"
            for metric in ["accuracy", "precision", "recall", "f1"]:
                mean_score = np.mean(cv_results[f"test_{metric}"])
                report += f"{metric.capitalize()}: {mean_score:.4f}\n"

            report += "\nClassification Report:\n"
            report += classification_report(
                y, self.model.predict(X), target_names=target_names
            )

            return report
        except Exception as e:
            logger.error(
                f"Error occurred while generating cross-validation report: {e}"
            )
            raise RuntimeError(f"Cross-validation report generation failed: {e}")

    def confusion_matrix_report(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        labels: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Generate a confusion matrix report.

        Args:
            X: Features.
            y: Labels.
            labels: List of labels to include in the confusion matrix.

        Returns:
            The confusion matrix.

        Raises:
            ValueError: If the input data is invalid.
            RuntimeError: If the model is not a classifier.
        """
        if X.empty or y.empty:
            raise ValueError("Input data (X or y) is empty.")

        if not isinstance(self.model, ClassifierMixin):
            raise ValueError("Model must be a classifier for classification metrics.")

        try:
            cm = confusion_matrix(y, self.model.predict(X), labels=labels)
            return cm
        except Exception as e:
            logger.error(
                f"Error occurred while generating confusion matrix report: {e}"
            )
            raise RuntimeError(f"Confusion matrix report generation failed: {e}")


def perform_ab_test(
    model_a: Union[ClassifierMixin, RegressorMixin],
    model_b: Union[ClassifierMixin, RegressorMixin],
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    scoring: Optional[Union[str, List[str]]] = None,
) -> Dict[str, Dict[str, Union[float, List[float]]]]:
    """Perform A/B testing between two models.

    Args:
        model_a: The first model to compare.
        model_b: The second model to compare.
        X: Features.
        y: Labels.
        cv: Number of cross-validation folds.
        scoring: Metric(s) to evaluate (default: accuracy for classifiers, r2 for regressors).

    Returns:
        Dictionary containing A/B test results.

    Raises:
        ValueError: If the input data is invalid.
        RuntimeError: If A/B testing fails.
    """
    if X.empty or y.empty:
        raise ValueError("Input data (X or y) is empty.")

    evaluator_a = ModelEvaluator(model_a)
    evaluator_b = ModelEvaluator(model_b)

    try:
        scores_a = evaluator_a.cross_validation_scores(X, y, cv=cv, scoring=scoring)
        scores_b = evaluator_b.cross_validation_scores(X, y, cv=cv, scoring=scoring)

        results = {}
        for metric in scores_a.keys():
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
        logger.error(f"Error occurred during A/B testing: {e}")
        raise RuntimeError(f"A/B testing failed: {e}")
