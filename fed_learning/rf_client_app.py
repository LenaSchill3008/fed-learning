import warnings
import numpy as np
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

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

        if self.model is None or not hasattr(self.model, 'estimators_'):
            # Return default hyperparameters as numpy array

            return [np.array([10.0, 5.0, 2.0, 1.0])]
        
        # Return hyperparameters as numpy array
        hyperparams = [
            float(self.model.n_estimators),
            float(self.model.max_depth if self.model.max_depth is not None else 40),
            float(self.model.min_samples_split),
            float(self.model.min_samples_leaf)
        ]
        return [np.array(hyperparams)]

    def set_parameters(self, parameters):

        if parameters and len(parameters) > 0:
            hyperparams = parameters[0].tolist()
            hyperparams = self._validate_hyperparams(hyperparams)
            self.model = create_rf_model(hyperparams)
        else:
            self.model = create_rf_model()
    

    def _validate_hyperparams(self, hyperparams):

        """
        Ensure hyperparameters are within valid ranges for RandomForest
        """

        if len(hyperparams) >= 4:
            hyperparams[0] = max(1, int(round(hyperparams[0])))  # n_estimators >= 1
            hyperparams[1] = max(1, int(round(hyperparams[1])))  # max_depth >= 1
            hyperparams[2] = max(2, int(round(hyperparams[2])))  # min_samples_split >= 2
            hyperparams[3] = max(1, int(round(hyperparams[3])))  # min_samples_leaf >= 1
        return hyperparams

    def _generate_random_hyperparams(self):

        """
        Generate random hyperparameters for Random Forest
        """

        n_estimators = np.random.randint(5, 100)  # 5 to 50 trees
        max_depth = np.random.choice([40, np.random.randint(3, 20)])  # None (40) or 3-20
        min_samples_split = np.random.randint(2, 10)  # 2 to 10
        min_samples_leaf = np.random.randint(1, 5)  # 1 to 5
        
        return [float(n_estimators), float(max_depth), float(min_samples_split), float(min_samples_leaf)]

    def _hyperparameter_search(self):

        """
        Perform random hyperparameter search and return best hyperparameters
        """

        n_trials = 50  # Fixed number of random trials
        
        # Split training data for validation (80% train, 20% validation)
        val_size = 0.2
        split_idx = int(len(self.X_train) * (1 - val_size))
        X_train_hp = self.X_train[:split_idx]
        X_val_hp = self.X_train[split_idx:]
        y_train_hp = self.y_train[:split_idx]
        y_val_hp = self.y_train[split_idx:]
        
        best_score = -1
        best_hyperparams = None
        
        # Random hyperparameter search
        for trial in range(n_trials):
            # Generate random hyperparameters
            trial_hyperparams = self._generate_random_hyperparams()
            
            # Create and train model
            trial_model = create_rf_model(trial_hyperparams)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                trial_model.fit(X_train_hp, y_train_hp)
            
            # Evaluate on validation set
            val_metrics = calculate_rf_metrics(trial_model, X_val_hp, y_val_hp)
            val_score = val_metrics["Accuracy"]
            
            # Keep track of best hyperparameters
            if val_score > best_score:
                best_score = val_score
                best_hyperparams = trial_hyperparams
        
        # Fallback to default if no good hyperparameters found
        if best_hyperparams is None:
            best_hyperparams = [10.0, 5.0, 2.0, 1.0]
        
        return best_hyperparams

    def fit(self, parameters, config):

        """
        Train the model with hyperparameter search and return updated parameters
        """

        # Perform hyperparameter search to find best hyperparameters
        best_hyperparams = self._hyperparameter_search()
        
        # Train final model with best hyperparameters on full training data
        self.model = create_rf_model(best_hyperparams)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.X_train, self.y_train)

        # Calculate training metrics
        train_metrics = calculate_rf_metrics(self.model, self.X_train, self.y_train)
        
        # Return current hyperparameters
        current_params = self.get_parameters()
        
        return current_params, len(self.X_train), train_metrics


    def evaluate(self, parameters, config):

        # Create a model with received parameters and train it
        if parameters and len(parameters) > 0:
            hyperparams = parameters[0].tolist()
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
        
        # Calculate loss (1 - accuracy as a simple loss)
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