# server_app.py

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from fed_learning.task import (
    get_model,
    get_model_params,
    set_initial_params,
    set_dataset,
)

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

    # Define strategy
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=initial_parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# Create the app
app = ServerApp(server_fn=server_fn)
