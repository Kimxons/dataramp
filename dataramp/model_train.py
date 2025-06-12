import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, is_classifier, is_regressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import type_of_target

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
plt.style.use("seaborn-v0_8-whitegrid")


def _validate_input_data(X: pd.DataFrame, y: pd.Series, validate_features: bool = False) -> None:
    """Validate input data for common issues."""
    if X.empty or y.empty:
        raise ValueError("Input data (X or y) is empty.")
    if len(X) != len(y):
        raise ValueError("X and y have different number of samples.")
    if y.isna().any() or X.isna().any().any():
        raise ValueError("Input data contains NaN values.")
    if validate_features and not all(X.dtypes.apply(lambda dt: np.issubdtype(dt, np.number))):
        raise ValueError("X contains non-numeric features.")
    if len(y.unique()) == 1:
        logger.warning("Target variable has only one unique value.")


def _validate_model_type(model: BaseEstimator, expected_type: str) -> None:
    """Validate model type matches expected type."""
    if expected_type == "classifier" and not is_classifier(model):
        raise ValueError(f"Model {model.__class__.__name__} is not a classifier.")
    if expected_type == "regressor" and not is_regressor(model):
        raise ValueError(f"Model {model.__class__.__name__} is not a regressor.")


def train_model(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    validate_features: bool = False,
    **fit_kwargs: Dict[str, Any],
) -> BaseEstimator:
    """
    Train a machine learning model with comprehensive validation.
    
    Args:
        model: Initialized scikit-learn estimator
        X: Training features
        y: Training target
        model_name: Name for logging identification
        validate_features: Check for numeric features
        **fit_kwargs: Additional arguments for model.fit()
    
    Returns:
        Trained model
    """
    _validate_input_data(X, y, validate_features)
    
    try:
        logger.info(f"Training {model_name} model on {len(X)} samples...")
        model.fit(X, y, **fit_kwargs)
        logger.info(f"{model_name} trained successfully")
        return model
    except Exception as e:
        logger.exception(f"Error training {model_name}: {e}")
        raise RuntimeError(f"Training failed for {model_name}") from e


def evaluate_model(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    model_type: Optional[str] = None,
    plots: Optional[List[str]] = None,
    plot_dir: Optional[str] = None,
    sample_weight: Optional[np.ndarray] = None,
) -> Dict[str, Union[float, Dict, np.ndarray]]:
    """
    Evaluate model performance with comprehensive metrics and visualizations.
    
    Args:
        model: Trained scikit-learn model
        X: Input features
        y: True target values
        model_type: 'classifier' or 'regressor' (auto-detected if None)
        plots: List of plots to generate
        plot_dir: Directory to save plots
        sample_weight: Sample weights for evaluation
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Auto-detect model type if not provided
    if model_type is None:
        model_type = "classifier" if is_classifier(model) else "regressor"
    
    _validate_model_type(model, model_type)
    _validate_input_data(X, y)
    
    if not hasattr(model, "predict"):
        raise AttributeError("Model does not have a predict method")
    
    try:
        y_pred = model.predict(X)
    except NotFittedError:
        logger.error("Model not fitted. Call train_model first.")
        raise
    
    y_true = y.to_numpy() if isinstance(y, pd.Series) else y
    results = {}
    plots = plots or []
    
    if model_type == "classifier":
        # Classification metrics
        results["report"] = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )
        results["confusion_matrix"] = confusion_matrix(y_true, y_pred)
        logger.info("Classification Report:\n" + classification_report(y_true, y_pred))
        
        # Handle binary classification metrics
        if len(np.unique(y_true)) == 2:
            try:
                y_score = model.predict_proba(X)[:, 1]
                results["roc_auc"] = roc_auc_score(y_true, y_score)
                results["pr_auc"] = _calculate_pr_auc(y_true, y_score)
            except (AttributeError, IndexError):
                logger.warning("ROC/PR metrics unavailable for this classifier")
        
        # Generate requested plots
        if "confusion_matrix" in plots:
            _plot_confusion_matrix(results["confusion_matrix"], plot_dir)
        if "roc" in plots and "roc_auc" in results:
            _plot_roc_curve(y_true, y_score, results["roc_auc"], plot_dir)
        if "pr" in plots and "pr_auc" in results:
            _plot_pr_curve(y_true, y_score, results["pr_auc"], plot_dir)
            
    else:  # Regression
        # Regression metrics
        results["mse"] = mean_squared_error(y_true, y_pred, sample_weight=sample_weight)
        results["mae"] = mean_absolute_error(y_true, y_pred, sample_weight=sample_weight)
        results["r2"] = r2_score(y_true, y_pred, sample_weight=sample_weight)
        logger.info(f"MSE: {results['mse']:.4f}, MAE: {results['mae']:.4f}, R²: {results['r2']:.4f}")
        
        # Generate requested plots
        if "residuals" in plots:
            _plot_residuals(y_true, y_pred, plot_dir)
        if "prediction" in plots:
            _plot_pred_vs_actual(y_true, y_pred, plot_dir)
            
    return results


def _calculate_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Calculate precision-recall AUC score."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return np.trapz(precision, recall)


def _plot_confusion_matrix(cm: np.ndarray, plot_dir: Optional[str] = None) -> None:
    """Plot confusion matrix as heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    _save_plot("confusion_matrix", plot_dir)


def _plot_roc_curve(
    y_true: np.ndarray, 
    y_score: np.ndarray, 
    roc_auc: float,
    plot_dir: Optional[str] = None
) -> None:
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", label=f"ROC (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    _save_plot("roc_curve", plot_dir)


def _plot_pr_curve(
    y_true: np.ndarray, 
    y_score: np.ndarray, 
    pr_auc: float,
    plot_dir: Optional[str] = None
) -> None:
    """Plot precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    plt.figure()
    plt.plot(recall, precision, color="blue", label=f"PR (AUC = {pr_auc:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    _save_plot("pr_curve", plot_dir)


def _plot_residuals(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    plot_dir: Optional[str] = None
) -> None:
    """Plot residuals vs predicted values."""
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color="r", linestyle="--")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    _save_plot("residual_plot", plot_dir)


def _plot_pred_vs_actual(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    plot_dir: Optional[str] = None
) -> None:
    """Plot predicted vs actual values."""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], "r--")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted")
    _save_plot("pred_vs_actual", plot_dir)


def _save_plot(plot_name: str, plot_dir: Optional[str] = None) -> None:
    """Save plot to directory if specified."""
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f"{plot_name}.png"))
        plt.close()
    else:
        plt.show()


def train_and_evaluate(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    test_size: float = 0.2,
    random_state: int = 42,
    plots: Optional[List[str]] = None,
    plot_dir: Optional[str] = None,
    **kwargs: Any,
) -> Tuple[BaseEstimator, Dict[str, Any]]:
    """
    End-to-end training and evaluation workflow.
    
    Args:
        model: Initialized model
        X: Full feature set
        y: Full target set
        model_name: Identifier for logging
        test_size: Proportion for test split
        random_state: Random seed
        plots: List of plots to generate
        plot_dir: Directory to save plots
        **kwargs: Additional training parameters
    
    Returns:
        Tuple of trained model and evaluation results
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    trained_model = train_model(
        model, X_train, y_train, model_name, **kwargs
    )
    
    model_type = "classifier" if is_classifier(model) else "regressor"
    evaluation = evaluate_model(
        trained_model,
        X_test,
        y_test,
        model_type=model_type,
        plots=plots,
        plot_dir=plot_dir
    )
    
    return trained_model, evaluation


# Model-specific training functions (with standardized interface)
def train_linear_regression(
    X: pd.DataFrame,
    y: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    plots: Optional[List[str]] = None,
    plot_dir: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Train and evaluate linear regression model."""
    model = LinearRegression(**kwargs.get("model_kwargs", {}))
    return _train_and_evaluate_wrapper(
        model, X, y, X_val, y_val, "linear regression", plots, plot_dir, kwargs
    )


def train_logistic_regression(
    X: pd.DataFrame,
    y: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    plots: Optional[List[str]] = None,
    plot_dir: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Train and evaluate logistic regression model."""
    model_kwargs = kwargs.get("model_kwargs", {})
    model_kwargs.setdefault("max_iter", 1000)
    model = LogisticRegression(**model_kwargs)
    return _train_and_evaluate_wrapper(
        model, X, y, X_val, y_val, "logistic regression", plots, plot_dir, kwargs
    )


def train_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    model_type: str = "classifier",
    plots: Optional[List[str]] = None,
    plot_dir: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Train and evaluate random forest model."""
    model_kwargs = kwargs.get("model_kwargs", {})
    if model_type == "classifier":
        model = RandomForestClassifier(**model_kwargs)
    elif model_type == "regressor":
        model = RandomForestRegressor(**model_kwargs)
    else:
        raise ValueError("model_type must be 'classifier' or 'regressor'")
    
    return _train_and_evaluate_wrapper(
        model, X, y, X_val, y_val, f"random forest {model_type}", plots, plot_dir, kwargs
    )


def _train_and_evaluate_wrapper(
    model: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame],
    y_val: Optional[pd.Series],
    model_name: str,
    plots: Optional[List[str]],
    plot_dir: Optional[str],
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Wrapper for training and evaluation workflow."""
    # Train model
    trained_model = train_model(
        model, 
        X_train, 
        y_train, 
        model_name,
        **kwargs.get("fit_kwargs", {})
    )
    
    # Prepare results
    results = {"model": trained_model}
    
    # Evaluate on validation set if available
    if X_val is not None and y_val is not None:
        model_type = "classifier" if is_classifier(model) else "regressor"
        results["evaluation"] = evaluate_model(
            trained_model,
            X_val,
            y_val,
            model_type=model_type,
            plots=plots,
            plot_dir=plot_dir,
            sample_weight=kwargs.get("sample_weight")
        )
    
    return results