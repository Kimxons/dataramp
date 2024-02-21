import os
import platform
import sys
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

"""
Model traning: Algorithms, Ensemble, Parameter tuning,
    Retraining, Model management.
"""
sys.path.append(os.getcwd())


def switch_plotting_backend():
    if platform.system() != "Darwin":
        plt.switch_backend("Agg")


def train_classifier(
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    estimator: object,
    X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    y_val: Optional[Union[pd.Series, np.ndarray]] = None,
    cross_validate: bool = False,
    cv: int = 5,
) -> dict:
    """
    Train a classification estimator and calculate numerous performance metrics.

    Args:
        X_train: The feature set (X) to use in training an estimator to predict the outcome (y).
        y_train: The ground truth value for the training dataset.
        estimator: The estimator to be trained and evaluated.
        X_val: The feature set (X) to use in validating a trained estimator (optional).
        y_val: The ground truth value for the validation dataset (optional).
        cross_validate: Whether to use a cross-validation strategy.
        cv: Number of folds to use in cross-validation.

    Returns:
        dict: A dictionary containing various classification metrics.
    """
    if any(arg is None for arg in [X_train, y_train, X_val, y_val]):
        raise ValueError("Some input arguments are None.")

    result_dict = {}

    switch_plotting_backend()

    if cross_validate:
        scorers = [
            ("Accuracy", accuracy_score),
            ("F1-score", f1_score),
            ("Precision", precision_score),
            ("Recall", recall_score),
        ]

        for metric_name, scorer in scorers:
            cv_score = cross_val_score(
                estimator, X_train, y_train, scoring=scorer, cv=cv
            )
            mean_score, std_score = cv_score.mean(), cv_score.std()
            result_dict[metric_name] = {"mean": mean_score, "std": std_score}
            print(f"{metric_name}: {mean_score:.4f} +/- {std_score:.4f}")
    else:
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_val)
        classification_rep = classification_report(y_val, y_pred, output_dict=True)
        confusion_mat = confusion_matrix(y_val, y_pred)

        result_dict["classification_report"] = classification_rep
        result_dict["confusion_matrix"] = confusion_mat

        print(classification_report(y_val, y_pred))
        print(f"Confusion Matrix:\n {confusion_mat}")

        # ROC plot
        if hasattr(estimator, "predict_proba"):
            y_pred_proba = estimator.predict_proba(X_val)[:, 1]
            fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
            roc_auc = roc_auc_score(y_val, y_pred_proba)

            plt.plot(
                fpr, tpr, color="darkorange", label=f"ROC curve (AUC = {roc_auc:.2f})"
            )
            plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Receiver Operating Characteristic Curve")
            plt.legend()

            result_dict["roc_auc"] = roc_auc

    return result_dict
