"""
Model Evaluation: Cross Validation, Model Reporting, A/B Testing.
"""

import logging
from typing import List, Union

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, cross_validate

logger = logging.getLogger(__name__)


class ModelEvaluator(BaseEstimator):
    """Class for evaluating ml models."""

    def __init__(self, model: ClassifierMixin) -> None:
        self.model = model

    def cross_validation_scores(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        scoring: Union[str, None] = None,
    ) -> List[float]:
        if not isinstance(self.model, ClassifierMixin):
            raise ValueError("Model must be a classifier for classification metrics.")

        try:
            scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring)
        except Exception as e:
            logger.error(f"Error occurred during cross-validation: {str(e)}")
            raise

        return scores.tolist() # return scores

    def cross_validation_report(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        target_names: Union[List[str], None] = None,
    ) -> str:
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

            train_metrics = {
                f"train_{metric}": np.mean(cv_results[f"train_{metric}"])
                for metric in ["accuracy", "precision", "recall", "f1"]
            }
            test_metrics = {
                f"test_{metric}": np.mean(cv_results[f"test_{metric}"])
                for metric in ["accuracy", "precision", "recall", "f1"]
            }

            report = "Training Metrics:\n"
            report += (
                ", ".join([f"{k}: {v:.2f}" for k, v in train_metrics.items()]) + "\n"
            )
            report += "Testing Metrics:\n"
            report += (
                ", ".join([f"{k}: {v:.2f}" for k, v in test_metrics.items()]) + "\n"
            )

            if target_names:
                report += classification_report(
                    y, self.model.predict(X), target_names=target_names
                )
            else:
                report += classification_report(y, self.model.predict(X))

        except Exception as e:
            logger.error(
                f"Error occurred while generating cross-validation report: {str(e)}"
            )
            raise

        return report

    def confusion_matrix_report(
        self, X: np.ndarray, y: np.ndarray, labels: Union[List[int], None] = None
    ) -> np.ndarray:
        """Confusion matrix report."""
        if not isinstance(self.model, ClassifierMixin):
            raise ValueError("Model must be a classifier for classification metrics.")

        try:
            cm = confusion_matrix(y, self.model.predict(X), labels=labels)
        except Exception as e:
            logger.error(
                f"Error occurred while generating confusion matrix report: {str(e)}"
            )
            raise

        return cm


def perform_ab_test(
    model_a: ClassifierMixin, model_b: ClassifierMixin, X: np.ndarray, y: np.ndarray
) -> tuple:
    """A/B testing between two models."""
    evaluator_a = ModelEvaluator(model_a)
    evaluator_b = ModelEvaluator(model_b)

    try:
        scores_a = evaluator_a.cross_validation_scores(X, y)
        scores_b = evaluator_b.cross_validation_scores(X, y)

        mean_score_a = np.mean(scores_a)
        mean_score_b = np.mean(scores_b)

    except Exception as e:
        logger.error(f"Error occurred during A/B testing: {str(e)}")
        raise

    return mean_score_a, mean_score_b
