import logging
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

def train_linear_regression(X: pd.DataFrame, y: pd.Series) -> Any:
    try:
        model = LinearRegression()
        model.fit(X, y)
        logger.info("Linear regression model trained successfully")
        return model
    except Exception as e:
        logger.error(f"Error training linear regression model: {e}")
        raise

def train_random_forest(X: pd.DataFrame, y: pd.Series) -> Any:
    try:
        model = RandomForestRegressor()
        model.fit(X, y)
        logger.info("Random forest model trained successfully")
        return model
    except Exception as e:
        logger.error(f"Error training random forest model: {e}")
        raise

def train_logistic_regression(X: pd.DataFrame, y: pd.Series) -> Any:
    try:
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        logger.info("Logistic regression model trained successfully")
        return model
    except Exception as e:
        logger.error(f"Error training logistic regression model: {e}")
        raise