import logging
import os
import platform
from typing import Callable, Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score
from sklearn.utils.validation import check_array, check_X_y

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def switch_plotting_backend() -> None:
    if platform.system() != "Darwin":
        plt.switch_backend("Agg")


def train_and_evaluate(
    model: BaseEstimator,
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    y_val: Optional[Union[pd.Series, np.ndarray]] = None,
    cv: int = 5,
    metrics: Optional[Dict[str, Callable]] = None,
    plot: bool = True,
    save_plot: Optional[str] = None,
) -> Dict[str, Union[Dict[str, float], np.ndarray, float, BaseEstimator]]:
    """
    Train and evaluate a machine learning model.

    Args:
        model: The classifier model to train.
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features (optional). If not provided, cross-validation is used.
        y_val: Validation labels (optional).
        cv: Number of cross-validation folds (default: 5).
        metrics: Dictionary of custom metric functions (default: accuracy, F1, precision, recall).
        plot: Whether to generate plots (default: True).
        save_plot: Path to save the ROC curve plot (optional).

    Returns:
        Dictionary containing training and evaluation results.
    """
    # Validate inputs
    X_train, y_train = check_X_y(X_train, y_train)
    if X_val is not None and y_val is not None:
        X_val = check_array(X_val)
        if X_val.shape[1] != X_train.shape[1]:
            raise ValueError("X_train and X_val must have the same number of features.")

    # Default metrics
    if metrics is None:
        metrics = {
            "Accuracy": accuracy_score,
            "F1-score": f1_score,
            "Precision": precision_score,
            "Recall": recall_score,
        }

    results = {}

    # Cross-validation
    if X_val is None or y_val is None:
        logger.info("Performing cross-validation...")
        cv_scores = {}
        for metric_name, metric_func in metrics.items():
            scores = cross_val_score(
                model, X_train, y_train, scoring=metric_func, cv=cv, n_jobs=-1
            )
            cv_scores[metric_name] = {"mean": scores.mean(), "std": scores.std()}
            logger.info(f"{metric_name}: {scores.mean():.4f} +/- {scores.std():.4f}")
        results["cross_validation_scores"] = cv_scores
    else:
        # Train the model
        logger.info("Training the model...")
        model.fit(X_train, y_train)

        # Evaluate the model
        logger.info("Evaluating the model...")
        y_pred = model.predict(X_val)
        results["classification_report"] = classification_report(
            y_val, y_pred, output_dict=True
        )
        results["confusion_matrix"] = confusion_matrix(y_val, y_pred)

        logger.info("Classification Report:")
        logger.info(classification_report(y_val, y_pred))
        logger.info("Confusion Matrix:")
        logger.info(confusion_matrix(y_val, y_pred))

        # ROC curve for binary classification
        if plot and hasattr(model, "predict_proba") and len(np.unique(y_val)) == 2:
            logger.info("Generating ROC curve...")
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
            roc_auc = roc_auc_score(y_val, y_pred_proba)

            plt.figure()
            plt.plot(
                fpr, tpr, color="darkorange", label=f"ROC curve (AUC = {roc_auc:.2f})"
            )
            plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Receiver Operating Characteristic Curve")
            plt.legend()

            if save_plot:
                plt.savefig(save_plot)
            else:
                plt.show()

            results["roc_auc"] = roc_auc

    # Return the trained model
    results["model"] = model
    return results
