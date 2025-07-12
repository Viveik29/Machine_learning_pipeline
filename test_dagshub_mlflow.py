import os
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
import numpy as np

# Replace these with your values
USERNAME = "viveik16693"
REPO_NAME = 'Machine_learning_pipeline'
TOKEN = '265a35ffa313931d4d42d67acb23b1b5ddc92b08'

# Set MLflow Tracking URI and authentication
mlflow.set_tracking_uri(f"https://dagshub.com/viveik16693/Machine_learning_pipeline.mlflow")
os.environ["MLFLOW_TRACKING_USERNAME"] = "viveik16693"
os.environ["MLFLOW_TRACKING_PASSWORD"] = '265a35ffa313931d4d42d67acb23b1b5ddc92b08'

# Start MLflow run
with mlflow.start_run(run_name="dagshub-mlflow-test") as run:
    print(f"Run ID: {run.info.run_id}")

    # Log a parameter and a metric
    mlflow.log_param("test_param", 123)
    mlflow.log_metric("test_metric", 0.99)

    # Train and log a dummy model
    X = np.array([[1], [2], [3], [4]])
    y = np.array([2, 4, 6, 8])
    model = LinearRegression()
    model.fit(X, y)

    mlflow.sklearn.log_model(model, "model")

print("✅ Test run completed. Check your DagsHub MLflow dashboard.")
