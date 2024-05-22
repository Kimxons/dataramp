"""
Feature engineering is the process of using domain knowledge to extract features from raw data via data mining techniques.
These features can be used to improve the performance of machine learning algorithms.
Feature engineering can be considered as applied machine learning itself.
"""

from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# credits: [https://github.com/ashishpatel26/Amazing-Feature-Engineering/blob/master/A%20Short%20Guide%20for%20Feature%20Engineering%20and%20Feature%20Selection.pdf]

class FeatureSelector:
    """
    A class to perform feature selection using different methods.
    """

    def __init__(self, estimator=None):
        """
        Initialize the FeatureSelector with an estimator.

        Parameters:
        -----------
        estimator : object, optional
            An estimator to use for feature selection.
        """
        self.estimator = estimator

    def recursive_feature_elimination(self, X, y, n_features_to_select=None):
        """
        Perform Recursive Feature Elimination (RFE) for feature selection.

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
        selector = RFE(
            estimator=self.estimator, n_features_to_select=n_features_to_select
        )
        return selector.fit_transform(X, y)

    def feature_importance_analysis(self, X, y):
        self.estimator.fit(X, y)
        return SelectFromModel(self.estimator, prefit=True).transform(X)


class FeatureEngineer:
    """
    A class to perform feature engineering tasks.
    """

    def __init__(self, categorical_features=None, numerical_features=None):
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features

    def handle_missing_values(self, X):
        imputer = SimpleImputer(strategy="mean")
        return imputer.fit_transform(X)

    def encode_categorical_variables(self, X):
        encoder = OneHotEncoder()
        return encoder.fit_transform(X[self.categorical_features])

    def scale_numerical_features(self, X):
        scaler = StandardScaler()
        return scaler.fit_transform(X[self.numerical_features])

    def feature_engineering_pipeline(self):
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

# Usage
# feature_engineer = FeatureEngineer(categorical_features=cat_vars, numerical_features=num_vars)

# preprocessing_pipeline = feature_engineer.feature_engineering_pipeline()

# feature_selector = FeatureSelector(estimator=RandomForestClassifier(n_estimators=100))

# full_pipeline = Pipeline([
#     ('preprocessor', preprocessing_pipeline),
#     ('feature_selector', SelectFromModel(estimator=RandomForestClassifier(n_estimators=100)))
# ])

# X_train_transformed = full_pipeline.fit_transform(X_train, y_train)
# X_test_transformed = full_pipeline.transform(X_test)

