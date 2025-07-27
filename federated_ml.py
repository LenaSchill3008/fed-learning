import pandas as pd
import sys
import warnings
from pathlib import Path
from typing import Dict

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class FederatedLearningRunner:


    def __init__(self, result_dir = "results"):
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(exist_ok=True)
        self.csv_file = self.result_dir / "results.csv"
        
        # Initialize results CSV with strategy column
        self.init_results_csv()
    

    def init_results_csv(self):

        columns = ["model", "dataset", "strategy", "loss", "accuracy", "precision", "recall", "f1_score"]
        df = pd.DataFrame(columns=columns)
        df.to_csv(self.csv_file, index=False)
    

    def run_experiment(self, model_type, dataset, strategy = "fedavg", rounds = 5):
        
        """
        Run a single federated learning experiment with specified strategy
        """

        print(f"Running {model_type} on {dataset} dataset with {strategy.upper()}")
        
        log_file = self.result_dir / f"{model_type}_{dataset}_{strategy}.log"
        print(f"    log -> {log_file}")
        
        # Use our custom federated learning implementation
        try:
            metrics = self.run_custom_federated_experiment(model_type, dataset, strategy, rounds, log_file)
            return metrics
            
        except Exception as e:
            print(f"Error running {model_type} on {dataset} with {strategy}: {e}")
            return {"loss": "ERROR", "accuracy": "ERROR", "precision": "ERROR", 
                   "recall": "ERROR", "f1_score": "ERROR"}
    
    def run_custom_federated_experiment(self, model_type, dataset, strategy, rounds, log_file):
        
        """
        Run federated learning experiment using our custom implementation with aggregation strategies
        """
        
        # Import our federated learning modules
        sys.path.append('.')


        from fed_learning.task import (
            set_dataset, set_model_type, load_data, 
            get_model, get_model_params, set_initial_params, set_model_params
        )
        from fed_learning.aggregation_strategies import get_strategy
        import numpy as np
        

        # Handle Random Forest separately
        if model_type == "random_forest":
            return self.run_random_forest_experiment(dataset, strategy, rounds, log_file)
        
        # Convert model names to our format
        model_mapping = {
            "logistic_regression": "logistic",
            "svm": "svm"
        }
        
        internal_model_type = model_mapping.get(model_type, "logistic")
        
        # Setup experiment
        set_dataset(dataset)
        set_model_type(internal_model_type)
        
        num_clients = 3
        
        # Strategy-specific parameters
        strategy_params = {}
        if strategy == "fedprox":
            strategy_params = {"mu": 0.01}
    
        
        # Create aggregation strategy
        try:
            aggregation_strategy = get_strategy(strategy, **strategy_params)
        except Exception as e:
            print(f"Failed to create strategy {strategy}: {e}")
            return {"loss": "STRATEGY_ERROR", "accuracy": "STRATEGY_ERROR", "precision": "STRATEGY_ERROR", 
                   "recall": "STRATEGY_ERROR", "f1_score": "STRATEGY_ERROR"}
        
        with open(log_file, 'w') as f:
            f.write(f"Starting {model_type} experiment on {dataset} dataset\n")
            f.write(f"Model type: {internal_model_type}, Strategy: {strategy}, Clients: {num_clients}\n")
            f.write(f"Strategy params: {strategy_params}\n")
            f.write("=" * 50 + "\n")
        
        try:
            # Create and initialize model
            if internal_model_type == "logistic":
                global_model = get_model(
                    model_type=internal_model_type,
                    penalty="l2",
                    local_epochs=20
                )
            else:  # SVM
                global_model = get_model(
                    model_type=internal_model_type,
                    kernel="rbf",
                    C=1.0,
                    gamma="scale"
                )
            
            set_initial_params(global_model)
            global_params = get_model_params(global_model)
            
            accuracies = []
            losses = []
            precisions = []
            recalls = []
            f1_scores = []
            
            # Federated learning rounds
            for round_num in range(rounds):
                with open(log_file, 'a') as f:
                    f.write(f"\n--- Round {round_num + 1}/{rounds} ---\n")
                
                client_params = []
                client_weights = []
                round_accuracies = []
                round_losses = []
                
                # Each client trains locally
                for client_id in range(num_clients):

                    # Load client data
                    X_train, X_test, y_train, y_test = load_data(client_id, num_clients)
                    
                    # Create local model
                    if internal_model_type == "logistic":
                        local_model = get_model(
                            model_type=internal_model_type,
                            penalty="l2",
                            local_epochs=10
                        )
                        
                        # Apply FedProx regularization
                        if strategy == "fedprox":
                            mu = strategy_params.get("mu", 0.01)
                            local_model.C = 1.0 / (1 + mu)
                        else:
                            local_model.C = 1.0
                            
                    else:  # SVM
                        local_model = get_model(
                            model_type=internal_model_type,
                            kernel="rbf",
                            C=1.0,
                            gamma="scale"
                        )
                    
                    # Set global parameters to local model
                    set_model_params(local_model, global_params)
                    
                    # Train local model
                    local_model.fit(X_train, y_train)
                    
                    # Evaluate local model
                    local_accuracy = local_model.score(X_test, y_test)
                    
                    # Calculate loss (using log loss approximation)
                    try:
                        if hasattr(local_model, 'predict_proba'):
                            y_pred_proba = local_model.predict_proba(X_test)

                            # Compute log loss
                            from sklearn.metrics import log_loss
                            local_loss = log_loss(y_test, y_pred_proba)
                        else:
                            # Fallback: use 1 - accuracy as proxy for loss
                            local_loss = 1.0 - local_accuracy
                    except:
                        local_loss = 1.0 - local_accuracy
                    
                    round_accuracies.append(local_accuracy)
                    round_losses.append(local_loss)
                    
                    # Get updated parameters
                    updated_params = get_model_params(local_model)
                    client_params.append(updated_params)
                    client_weights.append(len(X_train))
                    
                    with open(log_file, 'a') as f:
                        f.write(f"  Client {client_id}: accuracy = {local_accuracy:.3f}, loss = {local_loss:.3f}\n")
                
                # Apply aggregation strategy
                if internal_model_type == "logistic":
                    try:
                        # Use our aggregation strategy
                        global_params = aggregation_strategy.aggregate(
                            client_params, client_weights, global_params
                        )
                        
                        set_model_params(global_model, global_params)
                        
                        # Global evaluation
                        X_test_all, _, y_test_all, _ = load_data(0, 1)  # Use all data
                        global_accuracy = global_model.score(X_test_all, y_test_all)
                        
                        try:
                            if hasattr(global_model, 'predict_proba'):
                                y_pred_proba = global_model.predict_proba(X_test_all)
                                global_loss = log_loss(y_test_all, y_pred_proba)
                            else:
                                global_loss = 1.0 - global_accuracy
                        except:
                            global_loss = 1.0 - global_accuracy
                    
                    except Exception as e:

                        print(f"Aggregation failed: {e}")

                        # Fallback to simple averaging
                        total_weight = sum(client_weights)
                        aggregated_params = []
                        for param_idx in range(len(client_params[0])):
                            weighted_sum = np.zeros_like(client_params[0][param_idx])
                            for client_idx in range(num_clients):
                                weight = client_weights[client_idx] / total_weight
                                weighted_sum += weight * client_params[client_idx][param_idx]
                            aggregated_params.append(weighted_sum)
                        
                        global_params = aggregated_params
                        set_model_params(global_model, global_params)
                        
                        X_test_all, _, y_test_all, _ = load_data(0, 1)
                        global_accuracy = global_model.score(X_test_all, y_test_all)
                        global_loss = 1.0 - global_accuracy
                
                else:  # SVM - use average of client metrics (SVM doesn't support parameter aggregation well)
                    global_accuracy = np.mean(round_accuracies)
                    global_loss = np.mean(round_losses)
                
                # Calculate additional metrics
                try:
                    from sklearn.metrics import precision_score, recall_score, f1_score
                    if internal_model_type == "logistic":
                        X_test_all, _, y_test_all, _ = load_data(0, 1)
                        y_pred = global_model.predict(X_test_all)
                    else:
                        # For SVM, use majority vote from clients
                        X_test_all, _, y_test_all, _ = load_data(0, 1)
                        y_pred = global_model.predict(X_test_all)
                    
                    # Handle multi-class metrics
                    avg_method = 'weighted' if len(np.unique(y_test_all)) > 2 else 'binary'
                    
                    precision = precision_score(y_test_all, y_pred, average=avg_method, zero_division=0)
                    recall = recall_score(y_test_all, y_pred, average=avg_method, zero_division=0)
                    f1 = f1_score(y_test_all, y_pred, average=avg_method, zero_division=0)

                except:
                    precision = global_accuracy  # Fallback
                    recall = global_accuracy
                    f1 = global_accuracy
                
                accuracies.append(global_accuracy)
                losses.append(global_loss)
                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1)
                
                with open(log_file, 'a') as f:
                    f.write(f"  Global accuracy: {global_accuracy:.3f}\n")
                    f.write(f"  Global loss: {global_loss:.3f}\n")
                    f.write(f"  Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}\n")
                    f.write(f"  Strategy: {strategy.upper()}\n")
            
            # Return final metrics
            final_metrics = {
                "loss": f"{losses[-1]:.6f}" if losses else "NA",
                "accuracy": f"{accuracies[-1]:.6f}" if accuracies else "NA", 
                "precision": f"{precisions[-1]:.6f}" if precisions else "NA",
                "recall": f"{recalls[-1]:.6f}" if recalls else "NA",
                "f1_score": f"{f1_scores[-1]:.6f}" if f1_scores else "NA"
            }
            
            with open(log_file, 'a') as f:
                f.write(f"\nFinal Results ({strategy.upper()}):\n")
                f.write(f"Loss: {final_metrics['loss']}\n")
                f.write(f"Accuracy: {final_metrics['accuracy']}\n")
                f.write(f"Precision: {final_metrics['precision']}\n")
                f.write(f"Recall: {final_metrics['recall']}\n")
                f.write(f"F1 Score: {final_metrics['f1_score']}\n")
            
            return final_metrics
            
        except Exception as e:
            with open(log_file, 'a') as f:
                f.write(f"ERROR: {str(e)}\n")
            print(f"Error in federated experiment: {e}")
            import traceback
            traceback.print_exc()
            return {"loss": "ERROR", "accuracy": "ERROR", "precision": "ERROR", 
                   "recall": "ERROR", "f1_score": "ERROR"}
    
    def run_random_forest_experiment(self, dataset, strategy, rounds, log_file):
        
        """
        Run Random Forest federated learning experiment
        """
        

        from fed_learning.task import set_dataset, load_data
        from fed_learning.fed_random_forest import create_rf_model, calculate_rf_metrics, get_rf_hyperparams
        from fed_learning.aggregation_strategies import get_strategy
        import numpy as np

        
        set_dataset(dataset)
        num_clients = 3
        
        # Basic strategies supported for RF (hyperparameter averaging)
        if strategy not in ["fedavg", "fedprox", "fedmedian"]:
            print(f"Strategy {strategy} not supported for Random Forest, using FedAvg")
            strategy = "fedavg"
        
        with open(log_file, 'w') as f:
            f.write(f"Starting Random Forest experiment on {dataset} dataset\n")
            f.write(f"Strategy: {strategy}, Clients: {num_clients}\n")
            f.write("=" * 50 + "\n")
        
        try:
            # Initialize with default hyperparameters
            global_hyperparams = [10.0, 5.0, 2.0, 1.0]  # n_estimators, max_depth, min_samples_split, min_samples_leaf
            
            accuracies = []
            losses = []
            precisions = []
            recalls = []
            f1_scores = []
            
            # Federated learning rounds
            for round_num in range(rounds):
                with open(log_file, 'a') as f:
                    f.write(f"\n--- Round {round_num + 1}/{rounds} ---\n")
                
                client_hyperparams = []
                client_weights = []
                round_metrics = []
                
                # Each client performs hyperparameter search and training
                for client_id in range(num_clients):
                    # Load client data
                    X_train, X_test, y_train, y_test = load_data(client_id, num_clients)
                    
                    # Create model with current global hyperparameters
                    local_model = create_rf_model(global_hyperparams)
                    
                    # Train local model
                    local_model.fit(X_train, y_train)
                    
                    # Evaluate local model
                    local_metrics = calculate_rf_metrics(local_model, X_test, y_test)
                    round_metrics.append(local_metrics)
                    
                    # Get hyperparameters
                    local_hyperparams = get_rf_hyperparams(local_model)
                    client_hyperparams.append(local_hyperparams)
                    client_weights.append(len(X_train))
                    
                    with open(log_file, 'a') as f:
                        f.write(f"  Client {client_id}: accuracy = {local_metrics['Accuracy']:.3f}\n")
                
                # Aggregate hyperparameters based on strategy
                if strategy == "fedmedian":
                    # Use median aggregation for hyperparameters

                    new_hyperparams = []
                    for param_idx in range(len(client_hyperparams[0])):
                        param_values = [client_hyperparams[client_idx][param_idx] for client_idx in range(num_clients)]
                        median_value = np.median(param_values)
                        new_hyperparams.append(median_value)
                else:
                    # Use weighted average (fedavg, fedprox)
                    total_weight = sum(client_weights)
                    new_hyperparams = []
                    for param_idx in range(len(client_hyperparams[0])):
                        weighted_sum = 0
                        for client_idx in range(num_clients):
                            weight = client_weights[client_idx] / total_weight
                            weighted_sum += weight * client_hyperparams[client_idx][param_idx]
                        new_hyperparams.append(weighted_sum)
                
                global_hyperparams = new_hyperparams
                
                # Evaluate global model with averaged hyperparameters
                global_model = create_rf_model(global_hyperparams)
                
                # Train on all client data (for evaluation)
                all_X_train = []
                all_y_train = []
                all_X_test = []
                all_y_test = []
                
                for client_id in range(num_clients):
                    X_train, X_test, y_train, y_test = load_data(client_id, num_clients)
                    all_X_train.append(X_train)
                    all_X_test.append(X_test)
                    all_y_train.append(y_train)
                    all_y_test.append(y_test)
                
                # Combine all training data for global evaluation
                combined_X_train = np.vstack(all_X_train)
                combined_y_train = np.hstack(all_y_train)
                combined_X_test = np.vstack(all_X_test)
                combined_y_test = np.hstack(all_y_test)
                
                global_model.fit(combined_X_train, combined_y_train)
                global_metrics = calculate_rf_metrics(global_model, combined_X_test, combined_y_test)
                
                global_accuracy = global_metrics["Accuracy"]
                global_loss = 1.0 - global_accuracy  # Use 1 - accuracy as loss for RF
                precision = global_metrics["Precision"]
                recall = global_metrics["Recall"]
                f1 = global_metrics["F1_Score"]
                
                accuracies.append(global_accuracy)
                losses.append(global_loss)
                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1)
                
                with open(log_file, 'a') as f:
                    f.write(f"  Global accuracy: {global_accuracy:.3f}\n")
                    f.write(f"  Global loss: {global_loss:.3f}\n")
                    f.write(f"  Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}\n")
                    f.write(f"  Strategy used: {strategy.upper()}\n")
            
            # Return final metrics
            final_metrics = {
                "loss": f"{losses[-1]:.6f}" if losses else "NA",
                "accuracy": f"{accuracies[-1]:.6f}" if accuracies else "NA", 
                "precision": f"{precisions[-1]:.6f}" if precisions else "NA",
                "recall": f"{recalls[-1]:.6f}" if recalls else "NA",
                "f1_score": f"{f1_scores[-1]:.6f}" if f1_scores else "NA"
            }
            
            with open(log_file, 'a') as f:
                f.write(f"\nFinal Results ({strategy.upper()}):\n")
                f.write(f"Loss: {final_metrics['loss']}\n")
                f.write(f"Accuracy: {final_metrics['accuracy']}\n")
                f.write(f"Precision: {final_metrics['precision']}\n")
                f.write(f"Recall: {final_metrics['recall']}\n")
                f.write(f"F1 Score: {final_metrics['f1_score']}\n")
            
            return final_metrics
            
        except Exception as e:

            with open(log_file, 'a') as f:
                f.write(f"ERROR: {str(e)}\n")
            print(f"Error in Random Forest experiment: {e}")

            import traceback

            traceback.print_exc()

            return {"loss": "ERROR", "accuracy": "ERROR", "precision": "ERROR", 
                   "recall": "ERROR", "f1_score": "ERROR"}
    

    def save_results(self, model_type, dataset, strategy, metrics):

        new_row = {
            "model": model_type,
            "dataset": dataset,
            "strategy": strategy,
            **metrics
        }
        
        # Append to CSV
        df = pd.read_csv(self.csv_file)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(self.csv_file, index=False)
    
    def run_all_experiments(self):
        
        """
        Run all combinations of models, datasets, and strategies
        """

        models = ["logistic_regression", "random_forest", "svm"]
        datasets = ["iris", "adult"]
        
        # Different strategies to test
        strategies = ["fedavg", "fedprox", "fedmedian"]
        
        rounds = 5
        
        print("Starting federated learning experiments with multiple strategies")
        print("Models: Logistic Regression, Random Forest, SVM")
        print("Datasets: Iris, Adult")
        print(f"Strategies: {', '.join([s.upper() for s in strategies])}")
        print("=" * 70)
        
        total_experiments = len(models) * len(datasets) * len(strategies)
        current_experiment = 0
        
        for model in models:
            for dataset in datasets:
                for strategy in strategies:
                    current_experiment += 1
                    
                    try:
                        print(f"\n{'='*70}")
                        print(f"Experiment {current_experiment}/{total_experiments}")
                        print(f"Running {model.upper()} on {dataset.upper()} with {strategy.upper()}")
                        print(f"{'='*70}")
                        
                        metrics = self.run_experiment(model, dataset, strategy, rounds)
                        self.save_results(model, dataset, strategy, metrics)
                        
                        print(f"Completed {model} on {dataset} with {strategy}")
                        print(f"   Results: {metrics}")
                        print("-" * 50)
                        
                    except Exception as e:
                        print(f"Failed {model} on {dataset} with {strategy}: {e}")
                        error_metrics = {"loss": "ERROR", "accuracy": "ERROR", "precision": "ERROR", 
                                       "recall": "ERROR", "f1_score": "ERROR"}
                        self.save_results(model, dataset, strategy, error_metrics)
        
        print("\nAll experiments completed!")
        print(f"Results saved to: {self.csv_file}")
        
        # Display final results
        self.display_results()
        
    
    def display_results(self):

        try:
            df = pd.read_csv(self.csv_file)
            print("\nFINAL RESULTS:")
            print("=" * 100)
            print(df.to_string(index=False))
            
            # Display summary statistics by strategy
            print("\nSUMMARY BY STRATEGY:")
            print("=" * 60)
            for strategy in df['strategy'].unique():
                strategy_data = df[df['strategy'] == strategy]
                if len(strategy_data) > 0:
                    try:
                        # Convert to numeric, handling non-numeric values
                        numeric_cols = ['accuracy', 'precision', 'recall', 'f1_score']
                        for col in numeric_cols:
                            strategy_data[col] = pd.to_numeric(strategy_data[col], errors='coerce')
                        
                        avg_acc = strategy_data['accuracy'].mean()
                        if not pd.isna(avg_acc):
                            print(f"{strategy.upper():12s}: Avg Accuracy = {avg_acc:.3f}")
                        else:
                            print(f"{strategy.upper():12s}: Accuracy = ERROR/NA")
                    except:
                        print(f"{strategy.upper():12s}: Could not calculate average")
            
        except Exception as e:
            print(f"Error displaying results: {e}")


def main():

    print("\n" + "="*80)
    
    print("\nStarting experiments with multiple aggregation strategies...\n")
    
    runner = FederatedLearningRunner()
    runner.run_all_experiments()


if __name__ == "__main__":
    main()