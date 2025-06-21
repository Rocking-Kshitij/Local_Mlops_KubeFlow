from airflow import DAG

from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from datetime import datetime

with DAG(dag_id="train_model_on_k8s_mlflow_ns",
         start_date=datetime(2024, 1, 1),
         schedule_interval=None,
         catchup=False) as dag:

    train_model = KubernetesPodOperator(
        task_id="train-model",
        name="train-model",
        namespace="mlflow",
        image="mnist-train-mlflow:latest",
        is_delete_operator_pod=True,
        get_logs=True,
        image_pull_policy="Never"
)
