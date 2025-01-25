import logging
import platform
from typing import Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score
from sklearn.utils.validation import check_array, check_X_y

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def switch_plotting_backend() -> None:
    if platform.system() != "Darwin":
        plt.switch_backend("Agg")


def train_model(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    **kwargs: Dict[str, Any],
) -> BaseEstimator:
    """Train a machine learning model.

    Args:
        model: The model to train.
        X: Training features.
        y: Training labels.
        model_name: Name of the model (for logging purposes).
        **kwargs: Additional keyword arguments to pass to the model's `fit` method.

    Returns:
        The trained model.

    Raises:
        ValueError: If the input data is invalid.
        RuntimeError: If the model training fails.
    """
    if X.empty or y.empty:
        raise ValueError("Input data (X or y) is empty.")

    try:
        logger.info(f"Training {model_name} model...")
        model.fit(X, y, **kwargs)
        logger.info(f"{model_name} model trained successfully.")
        return model
    except Exception as e:
        logger.error(f"Error training {model_name} model: {e}")
        raise RuntimeError(f"Failed to train {model_name} model: {e}")


def evaluate_model(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "classifier",
    plot: bool = True,
    save_plot: Optional[str] = None,
) -> Dict[str, Union[Dict[str, float], np.ndarray, float]]:
    """Evaluate a machine learning model.

    Args:
        model: The trained model.
        X: Validation features.
        y: Validation labels.
        model_type: Type of model ('classifier' or 'regressor').
        plot: Whether to generate plots (default: True).
        save_plot: Path to save the plot (optional).

    Returns:
        Dictionary containing evaluation results.
    """
    results = {}

    # Predict
    y_pred = model.predict(X)

    # Classification metrics
    if model_type == "classifier":
        results["classification_report"] = classification_report(
            y, y_pred, output_dict=True
        )
        results["confusion_matrix"] = confusion_matrix(y, y_pred)

        logger.info("Classification Report:")
        logger.info(classification_report(y, y_pred))
        logger.info("Confusion Matrix:")
        logger.info(confusion_matrix(y, y_pred))

        # ROC curve for binary classification
        if plot and hasattr(model, "predict_proba") and len(np.unique(y)) == 2:
            logger.info("Generating ROC curve...")
            y_pred_proba = model.predict_proba(X)[:, 1]
            fpr, tpr, _ = roc_curve(y, y_pred_proba)
            roc_auc = roc_auc_score(y, y_pred_proba)

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

    # Regression metrics
    elif model_type == "regressor":
        results["mse"] = mean_squared_error(y, y_pred)
        results["r2"] = r2_score(y, y_pred)

        logger.info(f"MSE: {results['mse']:.4f}")
        logger.info(f"R2: {results['r2']:.4f}")

        # Residual plot
        if plot:
            logger.info("Generating residual plot...")
            residuals = y - y_pred
            plt.figure()
            plt.scatter(y_pred, residuals, color="blue", alpha=0.5)
            plt.axhline(y=0, color="red", linestyle="--")
            plt.xlabel("Predicted Values")
            plt.ylabel("Residuals")
            plt.title("Residual Plot")

            if save_plot:
                plt.savefig(save_plot)
            else:
                plt.show()

    return results


def train_and_evaluate(
    model: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    cv: int = 5,
    model_type: str = "classifier",
    plot: bool = True,
    save_plot: Optional[str] = None,
    **kwargs: Dict[str, Any],
) -> Dict[str, Union[Dict[str, float], np.ndarray, float, BaseEstimator]]:
    """Train and evaluate a machine learning model.

    Args:
        model: The model to train.
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features (optional). If not provided, cross-validation is used.
        y_val: Validation labels (optional).
        cv: Number of cross-validation folds (default: 5).
        model_type: Type of model ('classifier' or 'regressor').
        plot: Whether to generate plots (default: True).
        save_plot: Path to save the plot (optional).
        **kwargs: Additional keyword arguments to pass to the model's `fit` method.

    Returns:
        Dictionary containing training and evaluation results.
    """
    # Validate inputs
    X_train, y_train = check_X_y(X_train, y_train)
    if X_val is not None and y_val is not None:
        X_val = check_array(X_val)
        if X_val.shape[1] != X_train.shape[1]:
            raise ValueError("X_train and X_val must have the same number of features.")

    results = {}

    # Cross-validation
    if X_val is None or y_val is None:
        logger.info("Performing cross-validation...")
        scorers = (
            {
                "accuracy": accuracy_score,
                "f1": f1_score,
                "precision": precision_score,
                "recall": recall_score,
            }
            if model_type == "classifier"
            else {
                "mse": mean_squared_error,
                "r2": r2_score,
            }
        )

        cv_scores = {}
        for metric_name, metric_func in scorers.items():
            scores = cross_val_score(
                model, X_train, y_train, scoring=metric_func, cv=cv, n_jobs=-1
            )
            cv_scores[metric_name] = {"mean": scores.mean(), "std": scores.std()}
            logger.info(f"{metric_name}: {scores.mean():.4f} +/- {scores.std():.4f}")
        results["cross_validation_scores"] = cv_scores
    else:
        # Train the model
        model = train_model(model, X_train, y_train, model.__class__.__name__, **kwargs)

        # Evaluate the model
        results.update(evaluate_model(model, X_val, y_val, model_type, plot, save_plot))

    # Return the trained model
    results["model"] = model
    return results
