import os
import platform
import sys
from typing import Callable, Dict, Optional, Union

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
    x_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    estimator: object,
    x_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    y_val: Optional[Union[pd.Series, np.ndarray]] = None,
    cross_validate: bool = False,
    cv: int = 5,
    custom_metrics: Optional[Dict[str, Callable]] = None,
    callbacks: Optional[Dict[str, Callable]] = None,
    aggregation: str = "mean",
    plot_options: Optional[Dict[str, str]] = None,
) -> dict:
    if any(arg is None for arg in [x_train, y_train, x_val, y_val]):
        raise ValueError("Some input arguments are None.")

    result_dict = {}

    switch_plotting_backend()

    if cross_validate:
        scorers = {
            "Accuracy": accuracy_score,
            "F1-score": f1_score,
            "Precision": precision_score,
            "Recall": recall_score,
        }

        if custom_metrics:
            scorers |= custom_metrics

        cv_scores = {}

        for metric_name, scorer in scorers.items():
            cv_score = cross_val_score(
                estimator, x_train, y_train, scoring=scorer, cv=cv
            )
            if aggregation == "mean":
                mean_score, std_score = cv_score.mean(), cv_score.std()
            elif aggregation == "weighted":
                mean_score, std_score = np.average(cv_score, weights=len(cv_score)), np.std(cv_score)
            else:
                raise ValueError("Invalid aggregation method. Choose 'mean' or 'weighted'.")

            cv_scores[metric_name] = {"mean": mean_score, "std": std_score}
            print(f"{metric_name}: {mean_score:.4f} +/- {std_score:.4f}")

        result_dict["cross_validation_scores"] = cv_scores
    else:
        estimator.fit(x_train, y_train)
        y_pred = estimator.predict(x_val)
        classification_rep = classification_report(y_val, y_pred, output_dict=True)
        confusion_mat = confusion_matrix(y_val, y_pred)

        result_dict["classification_report"] = classification_rep
        result_dict["confusion_matrix"] = confusion_mat

        print(classification_report(y_val, y_pred))
        print(f"Confusion Matrix:\n {confusion_mat}")

        # ROC plot
        if hasattr(estimator, "predict_proba"):
            y_pred_proba = estimator.predict_proba(x_val)[:, 1]
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

            if plot_options:
                for key, value in plot_options.items():
                    plt.set(key, value)

            plt.show()

    # Callbacks
    if callbacks:
        for callback_name, callback_func in callbacks.items():
            callback_func(result_dict)

    return result_dict
