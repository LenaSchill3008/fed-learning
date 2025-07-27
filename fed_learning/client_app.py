# client_app.py
"""fed-learning: A Flower / sklearn app."""

import warnings
from sklearn.metrics import log_loss, precision_score, recall_score, f1_score
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from fed_learning.task import (
    get_model,
    get_model_params,
    load_data,
    set_initial_params,
    set_model_params,
    set_dataset,
)

class FlowerClient(NumPyClient):
    def __init__(self, model, X_train, X_test, y_train, y_test):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def fit(self, parameters, config):
        set_model_params(self.model, parameters)

        # Suppress convergence warnings during local training
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.X_train, self.y_train)

        # Calculate training metrics
        y_train_pred = self.model.predict(self.X_train)
        
        train_accuracy = self.model.score(self.X_train, self.y_train)
        train_precision = precision_score(self.y_train, y_train_pred, average='macro', zero_division=0)
        train_recall = recall_score(self.y_train, y_train_pred, average='macro', zero_division=0)
        train_f1 = f1_score(self.y_train, y_train_pred, average='macro', zero_division=0)

        train_metrics = {
            "Accuracy": train_accuracy,
            "Precision": train_precision,
            "Recall": train_recall,
            "F1_Score": train_f1,
        }

        return get_model_params(self.model), len(self.X_train), train_metrics

    def evaluate(self, parameters, config):
        set_model_params(self.model, parameters)

        # Calculate loss
        y_proba = self.model.predict_proba(self.X_test)
        loss = log_loss(self.y_test, y_proba, labels=self.model.classes_)

        # Calculate predictions for additional metrics
        y_pred = self.model.predict(self.X_test)

        # Calculate all metrics
        accuracy = self.model.score(self.X_test, self.y_test)
        
        # Use 'macro' average for multi-class classification
        precision = precision_score(self.y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(self.y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(self.y_test, y_pred, average='macro', zero_division=0)

        metrics = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1_Score": f1,
        }

        return loss, len(self.X_test), metrics


def client_fn(context: Context):
    # Read config
    partition_id   = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    dataset_name = context.run_config.get("dataset", "iris")
    set_dataset(dataset_name)                      # inform task.py

    # Load partitioned client data
    X_train, X_test, y_train, y_test = load_data(partition_id, num_partitions)

    # Create and initialize model
    penalty       = context.run_config["penalty"]
    local_epochs  = context.run_config["local-epochs"]
    model         = get_model(penalty, local_epochs)
    set_initial_params(model)

    return FlowerClient(model, X_train, X_test, y_train, y_test).to_client()


app = ClientApp(client_fn=client_fn)