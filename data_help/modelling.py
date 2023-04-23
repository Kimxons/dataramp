from sklearn.model_selection import cross_val_score
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix,
                             classification_report, roc_curve, roc_auc_score)
import matplotlib.pyplot as plt


def train_classifier(
        X_train, y_train,
        X_val, y_val,
        estimator,
        cross_validate=False, cv=5):
    """
    Train a classification estimator and calculate numerous performance metrics.

    Parameters:
        X_train: {array-like, sparse matrix}, shape (n_samples, n_features)
            The feature set (X) to use in training an estimator to predict the outcome (y).
        y_train: array-like, shape (n_samples,)
            The ground truth value for the training dataset.
        X_val: {array-like, sparse matrix}, shape (n_samples, n_features)
            The feature set (X) to use in validating a trained estimator.
        y_val: array-like, shape (n_samples,)
            The ground truth value for the validation dataset.
        estimator: estimator object implementing 'fit'
            The estimator to be trained and evaluated.
        cross_validate: bool, default False
            Whether to use a cross-validation strategy.
        cv: int, default 5
            Number of folds to use in cross-validation.

    Returns:
        None
    """
    if X_train is None:
        raise ValueError(
            "X_train: Expecting a DataFrame/ numpy2d array, got 'None'")
    if y_train is None:
        raise ValueError(
            "y_train: Expecting a Series/ numpy1D array, got 'None'")
    if X_val is None:
        raise ValueError(
            "X_val: Expecting a DataFrame/ numpy array, got 'None'")
    if y_val is None:
        raise ValueError(
            "y_val: Expecting a Series/ numpy1D array, got 'None'")

    if cross_validate:
        dict_scorers = {'Accuracy': accuracy_score,
                        'F1-score': f1_score,
                        'Precision': precision_score,
                        'Recall': recall_score}

        for metric_name, scorer in dict_scorers.items():
            cv_score = cross_val_score(
                estimator, X_train, y_train, scoring=scorer, cv=cv)
            print(f'{metric_name}: {cv_score.mean():.4f} +/- {cv_score.std():.4f}')
    else:
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_val)
        print(classification_report(y_val, y_pred))
        print(f"Confusion Matrix:\n {confusion_matrix(y_val, y_pred)}")

        # ROC plot
        y_pred_proba = estimator.predict_proba(X_val)[:, 1]
        fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
        roc_auc = roc_auc_score(y_val, y_pred_proba)

        plt.plot(fpr, tpr, color='darkorange',
                 label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic Curve')
        plt.legend(loc='lower right')
        plt.show()
