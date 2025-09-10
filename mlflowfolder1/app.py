# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties.
# In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import os
import warnings
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import mlflow
import logging

# --- Logging setup ---
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# --- MLflow tracking URI ---
#remote_server_uri = "https://dagshub.com/GSNandakumar77/MLFLOW_EXPERIMENT-.mlflow"
#mlflow.set_tracking_uri(remote_server_uri)

# --- Metric evaluation helper ---
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Load dataset
    csv_url = "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception("Unable to download dataset: %s", e)
        sys.exit(1)

    # Train-test split
    train, test = train_test_split(data)
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    # Parameters
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    # Start MLflow run
    with mlflow.start_run():
        # Train ElasticNet
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        # Predictions
        predicted_qualities = lr.predict(test_x)

        # Metrics
        rmse, mae, r2 = eval_metrics(test_y, predicted_qualities)

        print(f"ElasticNet model (alpha={alpha}, l1_ratio={l1_ratio}):")
        print(f"  RMSE: {rmse}")
        print(f"  MAE: {mae}")
        print(f"  R2: {r2}")

        # Log parameters and metrics
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # Save model locally
        os.makedirs("models", exist_ok=True)
        joblib.dump(lr, "models/model.pkl")

        # Log as artifact (bypasses unsupported endpoint)
        mlflow.log_artifact("models/model.pkl", artifact_path="model")
