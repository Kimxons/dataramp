import logging
from typing import Any, Dict, Union

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
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


def train_model(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    **kwargs: Dict[str, Any],
) -> BaseEstimator:
    """
    Train a machine learning model.

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
    """
    Evaluate a machine learning model.

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


def train_linear_regression(
    X: pd.DataFrame,
    y: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    plot: bool = True,
    save_plot: Optional[str] = None,
    **kwargs: Dict[str, Any],
) -> Dict[str, Union[Dict[str, float], np.ndarray, float, BaseEstimator]]:
    """
    Train and evaluate a linear regression model.

    Args:
        X: Training features.
        y: Training labels.
        X_val: Validation features (optional).
        y_val: Validation labels (optional).
        plot: Whether to generate plots (default: True).
        save_plot: Path to save the plot (optional).
        **kwargs: Additional keyword arguments to pass to the model's `fit` method.

    Returns:
        Dictionary containing training and evaluation results.
    """
    model = LinearRegression()
    model = train_model(model, X, y, "linear regression", **kwargs)

    if X_val is not None and y_val is not None:
        results = evaluate_model(
            model, X_val, y_val, model_type="regressor", plot=plot, save_plot=save_plot
        )
        results["model"] = model
        return results
    else:
        return {"model": model}


def train_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    model_type: str = "classifier",
    plot: bool = True,
    save_plot: Optional[str] = None,
    **kwargs: Dict[str, Any],
) -> Dict[str, Union[Dict[str, float], np.ndarray, float, BaseEstimator]]:
    """
    Train and evaluate a random forest model.

    Args:
        X: Training features.
        y: Training labels.
        X_val: Validation features (optional).
        y_val: Validation labels (optional).
        model_type: Type of model ('classifier' or 'regressor').
        plot: Whether to generate plots (default: True).
        save_plot: Path to save the plot (optional).
        **kwargs: Additional keyword arguments to pass to the model's `fit` method.

    Returns:
        Dictionary containing training and evaluation results.
    """
    if model_type == "classifier":
        model = RandomForestClassifier()
    elif model_type == "regressor":
        model = RandomForestRegressor()
    else:
        raise ValueError("Invalid model_type. Choose 'classifier' or 'regressor'.")

    model = train_model(model, X, y, "random forest", **kwargs)

    if X_val is not None and y_val is not None:
        results = evaluate_model(
            model, X_val, y_val, model_type=model_type, plot=plot, save_plot=save_plot
        )
        results["model"] = model
        return results
    else:
        return {"model": model}


def train_logistic_regression(
    X: pd.DataFrame,
    y: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    plot: bool = True,
    save_plot: Optional[str] = None,
    **kwargs: Dict[str, Any],
) -> Dict[str, Union[Dict[str, float], np.ndarray, float, BaseEstimator]]:
    """
    Train and evaluate a logistic regression model.

    Args:
        X: Training features.
        y: Training labels.
        X_val: Validation features (optional).
        y_val: Validation labels (optional).
        plot: Whether to generate plots (default: True).
        save_plot: Path to save the plot (optional).
        **kwargs: Additional keyword arguments to pass to the model's `fit` method.

    Returns:
        Dictionary containing training and evaluation results.
    """
    model = LogisticRegression(max_iter=1000)
    model = train_model(model, X, y, "logistic regression", **kwargs)

    if X_val is not None and y_val is not None:
        results = evaluate_model(
            model, X_val, y_val, model_type="classifier", plot=plot, save_plot=save_plot
        )
        results["model"] = model
        return results
    else:
        return {"model": model}
