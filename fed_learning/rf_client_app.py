# fed_learning/rf_client_app.py
"""Random Forest client for federated learning - fixed implementation."""

import warnings
import numpy as np
import pickle
import base64
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, parameters_to_ndarrays, ndarrays_to_parameters

from fed_learning.task import load_data, set_dataset
from fed_learning.fed_random_forest import (
    create_rf_model,
    calculate_rf_metrics
)


class RFFlowerClient(NumPyClient):
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = None

    def get_parameters(self, config=None):
        """Return model parameters as numpy arrays."""
        if self.model is None or not hasattr(self.model, 'estimators_'):
            # Return default hyperparameters as numpy array
            return [np.array([10.0, 5.0, 2.0, 1.0])]  # n_estimators, max_depth, min_samples_split, min_samples_leaf
        
        # Return hyperparameters as numpy array
        hyperparams = [
            float(self.model.n_estimators),
            float(self.model.max_depth if self.model.max_depth is not None else 40),
            float(self.model.min_samples_split),
            float(self.model.min_samples_leaf)
        ]
        return [np.array(hyperparams)]

    def set_parameters(self, parameters):
        """Set model parameters from numpy arrays."""
        if parameters and len(parameters) > 0:
            hyperparams = parameters[0].tolist()
            # Ensure hyperparameters are within valid ranges
            hyperparams = self._validate_hyperparams(hyperparams)
            self.model = create_rf_model(hyperparams)
        else:
            self.model = create_rf_model()
    
    def _validate_hyperparams(self, hyperparams):
        """Ensure hyperparameters are within valid ranges for RandomForest."""
        if len(hyperparams) >= 4:
            # n_estimators: must be >= 1
            hyperparams[0] = max(1, int(round(hyperparams[0])))
            # max_depth: must be >= 1 or None (we use 40 as None marker)
            hyperparams[1] = max(1, int(round(hyperparams[1])))
            # min_samples_split: must be >= 2
            hyperparams[2] = max(2, int(round(hyperparams[2])))
            # min_samples_leaf: must be >= 1
            hyperparams[3] = max(1, int(round(hyperparams[3])))
        return hyperparams

    def fit(self, parameters, config):
        """Train the model and return updated parameters."""
        # Set parameters from server
        self.set_parameters(parameters)
        
        # Train the model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.X_train, self.y_train)

        # Calculate training metrics
        train_metrics = calculate_rf_metrics(self.model, self.X_train, self.y_train)
        
        # Return current hyperparameters
        current_params = self.get_parameters()
        
        return current_params, len(self.X_train), train_metrics

    def evaluate(self, parameters, config):
        """Evaluate the model and return metrics."""
        # Create a model with received parameters and train it
        if parameters and len(parameters) > 0:
            hyperparams = parameters[0].tolist()
            # Ensure hyperparameters are within valid ranges
            hyperparams = self._validate_hyperparams(hyperparams)
            eval_model = create_rf_model(hyperparams)
        else:
            eval_model = create_rf_model()
        
        # Train the evaluation model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eval_model.fit(self.X_train, self.y_train)
        
        # Calculate test metrics
        test_metrics = calculate_rf_metrics(eval_model, self.X_test, self.y_test)
        
        # Calculate loss (we'll use 1 - accuracy as a simple loss)
        loss = 1.0 - test_metrics["Accuracy"]
        
        return loss, len(self.X_test), test_metrics


def rf_client_fn(context: Context):
    # Read config
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    
    dataset_name = context.run_config.get("dataset", "iris")
    set_dataset(dataset_name)
    
    # Load partitioned client data
    X_train, X_test, y_train, y_test = load_data(partition_id, num_partitions)
    
    return RFFlowerClient(X_train, X_test, y_train, y_test).to_client()


# Create the app
rf_app = ClientApp(client_fn=rf_client_fn)