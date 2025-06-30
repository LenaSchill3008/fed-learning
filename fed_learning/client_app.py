# client_app.py
"""fed-learning: A Flower / sklearn app."""

import warnings
from sklearn.metrics import log_loss
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

        return get_model_params(self.model), len(self.X_train), {}

    def evaluate(self, parameters, config):
        set_model_params(self.model, parameters)

        y_proba = self.model.predict_proba(self.X_test)
        loss = log_loss(self.y_test, y_proba, labels=self.model.classes_)

        accuracy = self.model.score(self.X_test, self.y_test)
        return loss, len(self.X_test), {"accuracy": accuracy}


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
