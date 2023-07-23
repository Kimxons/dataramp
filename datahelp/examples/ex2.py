import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from eda import feature_summary, display_missing
from feature_engineering import drop_missing

# Data Configuration
data_file_path = "data/iris.csv"


# 2. Load and Preprocess Data
def load_dataset(file_path):
    return pd.read_csv(file_path)


def split_dataset(dataset):
    X = dataset.drop("species", axis=1)
    y = dataset["species"]
    return X, y


# Load and preprocess the dataset
try:
    df = load_dataset(data_file_path)
except FileNotFoundError:
    print("Dataset not found. Please check the file path.")
    exit(1)

# Perform EDA - Feature Summary and Display Missing Values
print("Feature Summary:")
summary_df = feature_summary(df)
print(summary_df)

print("\nMissing Values:")
missing_df = display_missing(df, plot=True)

# Drop columns with high missing values
df = drop_missing(df, threshold=50)

X, y = split_dataset(df)


# 3. Train the Model
def train_model(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return model, accuracy


# Train the model and get accuracy
trained_model, accuracy = train_model(X, y)
print(f"Model Accuracy: {accuracy:.2f}")


# 4. Visualize Feature Importance
def plot_feature_importance(importances, feature_names):
    sorted_indices = importances.argsort()[::-1]
    plt.bar(range(len(importances)), importances[sorted_indices])
    plt.xticks(range(len(importances)), feature_names[sorted_indices], rotation=90)
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.title("Feature Importances")
    plt.show()


# Get feature importances and plot
feature_importances = trained_model.feature_importances_
feature_names = X.columns
plot_feature_importance(feature_importances, feature_names)
