import platform
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import cross_val_score

if platform.system() == "Darwin":
    plt.switch_backend()
else:
    plt.switch_backend("Agg")


def train_classifier(
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    X_val: Union[pd.DataFrame, np.ndarray],
    y_val: Union[pd.Series, np.ndarray],
    estimator: object,
    cross_validate: bool = False,
    cv: int = 5,
) -> None:
    """
    Train a classification estimator and calculate numerous performance metrics.

    Args:
        X_train: The feature set (X) to use in training an estimator to predict the outcome (y).
        y_train: The ground truth value for the training dataset.
        X_val: The feature set (X) to use in validating a trained estimator.
        y_val: The ground truth value for the validation dataset.
        estimator: The estimator to be trained and evaluated.
        cross_validate: Whether to use a cross-validation strategy.
        cv: Number of folds to use in cross-validation.

    Returns:
        None
    """
    if X_train is None:
        raise ValueError("X_train is None")

    if y_train is None:
        raise ValueError("y_train is None")

    if X_val is None:
        raise ValueError("X_val is None")

    if y_val is None:
        raise ValueError("y_val is None")

    if cross_validate:
        scorers = [
            ("Accuracy", accuracy_score),
            ("F1-score", f1_score),
            ("Precision", precision_score),
            ("Recall", recall_score),
        ]

        for metric_name, scorer in scorers:
            cv_score = cross_val_score(estimator, X_train, y_train, scoring=scorer, cv=cv)
            print(f"{metric_name}: {cv_score.mean():.4f} +/- {cv_score.std():.4f}")
    else:
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_val)
        print(classification_report(y_val, y_pred))
        print(f"Confusion Matrix:\n {confusion_matrix(y_val, y_pred)}")

        # ROC plot
        if hasattr(estimator, "predict_proba"):
            y_pred_proba = estimator.predict_proba(X_val)[:, 1]
            fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
            roc_auc = roc_auc_score(y_val, y_pred_proba)

            plt.plot(fpr, tpr, color="darkorange", label=f"ROC curve (AUC = {roc_auc:.2f})")
            plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Receiver Operating Characteristic Curve")


# def plot_feature_importance(estimator=None, col_name=None):
#     """
#     Plots the feature importance from a trained scikit learn estimator
#     as a bar chart.

#     Parameters:
#     -----------
#     estimator : scikit-learn estimator
#         A fitted estimator that has a `feature_importances_` attribute.
#     col_names : list of str
#         The names of the columns in the same order as the feature importances.

#     Returns:
#     --------
#     fig : matplotlib Figure
#         The figure object containing the plot.
#     ax : matplotlib Axes
#         The axes object containing the plot.
#     """

#     if not hasattr(estimator, "feature_importances_"):
#         raise ValueError("The estimator does not have a 'feature_importances_' attribute.")
#     if not isinstance(feature_names, list) or len(feature_names) != estimator.n_features_:
#         raise ValueError("The 'feature_names' argument should be a list of the same length as the number of features.")

#     feature_importances = estimator.feature_importances_
#     feature_importances_df = pd.DataFrame({"feature": feature_names, "importance": feature_importances})
#     feature_importances_df = feature_importances_df.sort_values(by="importance", ascending=False)

#     fig, ax = plt.subplots()
#     sns.barplot(x="importance", y="feature", data=feature_importances_df, ax=ax)
#     ax.set_title("Feature importance plot")

#     return fig, ax
