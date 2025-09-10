# MLFLOW EXPERIMENTS 


import dagshub
dagshub.init(repo_owner='GSNandakumar77', repo_name='MLFLOW_EXPERIMENT-', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)

  https://dagshub.com/GSNandakumar77/MLFLOW_EXPERIMENT-.mlflow 

  # Set your MLflow tracking URI
export MLFLOW_TRACKING_URI=https://dagshub.com/GSNandakumar77/MLFLOW_EXPERIMENT-.mlflow

# Set your DagsHub username
export MLFLOW_TRACKING_USERNAME=GSNandakumar77

# Set your Personal Access Token
export MLFLOW_TRACKING_PASSWORD=0438d805f8025ac359dff7d27f17165549ede5e6

# Now run your Python script
python script.py