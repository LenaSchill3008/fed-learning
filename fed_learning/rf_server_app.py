# fed_learning/rf_server_app.py
"""Random Forest server for federated learning - fixed implementation."""

import numpy as np
from typing import Dict, List, Tuple, Optional
from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from fed_learning.task import set_dataset


def rf_metrics_aggregate(results: List[Tuple[int, Metrics]]) -> Dict:
    """Aggregate metrics from multiple clients using weighted averaging."""
    if not results:
        return {}
    
    total_samples = 0
    aggregated_metrics = {
        "Accuracy": 0,
        "Precision": 0,
        "Recall": 0,
        "F1_Score": 0,
    }
    
    # Extract values from the results
    for samples, metrics in results:
        for key, value in metrics.items():
            if key in aggregated_metrics:
                aggregated_metrics[key] += (value * samples)
        total_samples += samples
    
    # Compute the weighted average for each metric
    if total_samples > 0:
        for key in aggregated_metrics.keys():
            aggregated_metrics[key] = round(aggregated_metrics[key] / total_samples, 6)
    
    return aggregated_metrics


def rf_server_fn(context: Context):
    # Read config values
    num_rounds = context.run_config["num-server-rounds"]
    dataset_name = context.run_config.get("dataset", "iris")
    
    # Set the global dataset
    set_dataset(dataset_name)
    
    # Initial hyperparameters for Random Forest as numpy array
    n_estimators = context.run_config.get("n-estimators", 10)
    max_depth = context.run_config.get("max-depth", 5)
    
    initial_hyperparams = np.array([
        float(n_estimators),     # n_estimators
        float(max_depth),        # max_depth
        2.0,                     # min_samples_split
        1.0                      # min_samples_leaf
    ])
    
    # Convert to Flower parameters format
    initial_parameters = ndarrays_to_parameters([initial_hyperparams])
    
    # Use FedAvg to average hyperparameters
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=initial_parameters,
        evaluate_metrics_aggregation_fn=rf_metrics_aggregate,
        fit_metrics_aggregation_fn=rf_metrics_aggregate,
    )
    
    config = ServerConfig(num_rounds=num_rounds)
    
    return ServerAppComponents(strategy=strategy, config=config)


# Create the app
rf_app = ServerApp(server_fn=rf_server_fn)