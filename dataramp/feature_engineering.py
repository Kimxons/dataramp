"""This module provides classes for feature engineering and selection in machine learning pipelines.

It includes tools for handling missing values, encoding categorical variables, and selecting important features.
"""

import logging

from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO)


class FeatureSelector:
    """A class to perform feature selection using different methods."""

    def __init__(self, estimator=None):
        """Initialize the FeatureSelector with an estimator.

        Parameters:
        -----------
        estimator : object, optional
            An estimator to use for feature selection.
        """
        self.estimator = estimator

    def recursive_feature_elimination(self, X, y, n_features_to_select=None):
        """Perform Recursive Feature Elimination (RFE) for feature selection.

        Parameters:
        -----------
        X : array-like or DataFrame, shape (n_samples, n_features)
            The input samples.
        y : array-like, shape (n_samples,)
            The target values.
        n_features_to_select : int or None, optional (default=None)
            The number of features to select. If None, half of the features are selected.

        Returns:
        --------
        array-like or DataFrame
            The selected features.
        """
        logging.info("Performing Recursive Feature Elimination (RFE)")
        try:
            selector = RFE(
                estimator=self.estimator, n_features_to_select=n_features_to_select
            )
            return selector.fit_transform(X, y)
        except Exception as e:
            logging.error(f"Error in recursive_feature_elimination: {e}")
            raise

    def feature_importance_analysis(self, X, y):
        """Perform feature selection based on feature importance analysis.

        Parameters:
        -----------
        X : array-like or DataFrame, shape (n_samples, n_features)
            The input samples.
        y : array-like, shape (n_samples,)
            The target values.

        Returns:
        --------
        array-like or DataFrame
            The selected features.
        """
        logging.info("Performing feature importance analysis")
        try:
            self.estimator.fit(X, y)
            return SelectFromModel(self.estimator, prefit=True).transform(X)
        except Exception as e:
            logging.error(f"Error in feature_importance_analysis: {e}")
            raise


class FeatureEngineer:
    """A class to perform feature engineering tasks."""

    def __init__(self, categorical_features=None, numerical_features=None):
        """Initialize the FeatureEngineer with categorical and numerical features.

        Parameters:
        -----------
        categorical_features : list, optional
            The list of categorical feature names.
        numerical_features : list, optional
            The list of numerical feature names.
        """
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features

    def handle_missing_values(self, X, strategy="mean"):
        """Handle missing values in the dataset.

        Parameters:
        -----------
        X : array-like or DataFrame, shape (n_samples, n_features)
            The input samples.
        strategy : str, optional (default="mean")
            The strategy to use for imputing missing values.

        Returns:
        --------
        array-like or DataFrame
            The dataset with missing values handled.
        """
        logging.info("Handling missing values")
        try:
            imputer = SimpleImputer(strategy=strategy)
            return imputer.fit_transform(X)
        except Exception as e:
            logging.error(f"Error in handle_missing_values: {e}")
            raise

    def encode_categorical_variables(self, X):
        """Encode categorical variables using OneHotEncoder.

        Parameters:
        -----------
        X : DataFrame, shape (n_samples, n_features)
            The input samples.

        Returns:
        --------
        array-like
            The encoded categorical variables.
        """
        logging.info("Encoding categorical variables")
        try:
            encoder = OneHotEncoder(handle_unknown="ignore")
            return encoder.fit_transform(X[self.categorical_features])
        except Exception as e:
            logging.error(f"Error in encode_categorical_variables: {e}")
            raise

    def scale_numerical_features(self, X):
        """Scale numerical features using StandardScaler.

        Parameters:
        -----------
        X : DataFrame, shape (n_samples, n_features)
            The input samples.

        Returns:
        --------
        array-like
            The scaled numerical features.
        """
        logging.info("Scaling numerical features")
        try:
            scaler = StandardScaler()
            return scaler.fit_transform(X[self.numerical_features])
        except Exception as e:
            logging.error(f"Error in scale_numerical_features: {e}")
            raise

    def feature_engineering_pipeline(self):
        """Create a feature engineering pipeline for preprocessing the data.

        Returns:
        --------
        Pipeline
            The feature engineering pipeline.
        """
        logging.info("Creating feature engineering pipeline")

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        numerical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, self.numerical_features),
                ("cat", categorical_transformer, self.categorical_features),
            ]
        )

        return Pipeline(steps=[("preprocessor", preprocessor)])


# Example usage (commented out to prevent execution in this context)
# feature_engineer = FeatureEngineer(categorical_features=cat_vars, numerical_features=num_vars)
# preprocessing_pipeline = feature_engineer.feature_engineering_pipeline()
# feature_selector = FeatureSelector(estimator=RandomForestClassifier(n_estimators=100))

# full_pipeline = Pipeline([
#     ('preprocessor', preprocessing_pipeline),
#     ('feature_selector', SelectFromModel(estimator=RandomForestClassifier(n_estimators=100)))
# ])

# X_train_transformed = full_pipeline.fit_transform(X_train, y_train)
# X_test_transformed = full_pipeline.transform(X_test)
