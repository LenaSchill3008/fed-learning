# server_app.py

from typing import Dict, List, Tuple
from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from fed_learning.task import (
    get_model,
    get_model_params,
    set_initial_params,
    set_dataset,
)


def metrics_aggregate(results: List[Tuple[int, Metrics]]) -> Dict:
    """Aggregate metrics from multiple clients using weighted averaging."""
    if not results:
        return {}
    
    total_samples = 0
    # Collecting metrics
    aggregated_metrics = {
        "Accuracy": 0,
        "Precision": 0,
        "Recall": 0,
        "F1_Score": 0,
    }
    
    # Extracting values from the results
    for samples, metrics in results:
        for key, value in metrics.items():
            if key not in aggregated_metrics:
                aggregated_metrics[key] = 0
            else:
                aggregated_metrics[key] += (value * samples)
        total_samples += samples
    
    # Compute the weighted average for each metric
    for key in aggregated_metrics.keys():
        aggregated_metrics[key] = round(aggregated_metrics[key] / total_samples, 6)
    
    return aggregated_metrics


def server_fn(context: Context):
    # Read config values
    num_rounds   = context.run_config["num-server-rounds"]
    penalty      = context.run_config["penalty"]
    local_epochs = context.run_config["local-epochs"]
    dataset_name = context.run_config.get("dataset", "iris")

    # Set the global dataset
    set_dataset(dataset_name)

    # Initialize model and parameters
    model = get_model(penalty, local_epochs)
    set_initial_params(model)
    initial_parameters = ndarrays_to_parameters(get_model_params(model))

    # Define strategy with metrics aggregation
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=initial_parameters,
        evaluate_metrics_aggregation_fn=metrics_aggregate,
        fit_metrics_aggregation_fn=metrics_aggregate,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# Create the app
app = ServerApp(server_fn=server_fn)