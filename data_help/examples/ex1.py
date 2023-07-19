import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# import data_help as dh

df = pd.read_csv("data/iris.csv")

print(df.head())

"""
X = df.data
y = df.target

rf = RandomForestRegressor()
rf.fit(X, y)

feature_importances = rf.feature_importances_
feature_names = boston.feature_names

fig, ax = plot_feature_importance(rf, feature_names)
"""
