#!/usr/bin/env python3
"""
Python runner for federated learning experiments.
Runs both Logistic Regression and Random Forest on Iris and Adult datasets.
"""

import subprocess
import pandas as pd
import re
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class FederatedLearningRunner:
    def __init__(self, result_dir: str = "results"):
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(exist_ok=True)
        self.csv_file = self.result_dir / "results.csv"
        
        # Initialize results CSV
        self.init_results_csv()
    
    def init_results_csv(self):
        """Initialize the results CSV file with headers."""
        columns = ["model", "dataset", "loss", "accuracy", "precision", "recall", "f1_score"]
        df = pd.DataFrame(columns=columns)
        df.to_csv(self.csv_file, index=False)
    
    def run_experiment(self, model_type: str, dataset: str, rounds: int = 5) -> Dict[str, str]:
        """Run a single federated learning experiment."""
        print(f"▶️  Running {model_type} on {dataset} dataset")
        
        log_file = self.result_dir / f"{model_type}_{dataset}.log"
        print(f"    log ➜ {log_file}")
        
        # Prepare command based on model type
        if model_type == "logistic_regression":
            cmd = [
                "flwr", "run", ".",
                "--run-config", 
                f'dataset="{dataset}" num-server-rounds={rounds} local-epochs=3 penalty="l2"'
            ]
        elif model_type == "random_forest":
            # Create temporary pyproject.toml for RF
            self.create_rf_config(dataset, rounds)
            cmd = [
                "flwr", "run", ".",
                "--run-config",
                f'dataset="{dataset}" num-server-rounds={rounds} n-estimators=10 max-depth=5'
            ]
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Run the command
        try:
            with open(log_file, 'w') as f:
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=300  # 5 minute timeout
                )
                f.write(result.stdout)
                print(result.stdout)  # Also print to console
            
            # Parse results from log
            metrics = self.parse_log_file(log_file, rounds)
            return metrics
            
        except subprocess.TimeoutExpired:
            print(f"❌ Timeout running {model_type} on {dataset}")
            return {"loss": "TIMEOUT", "accuracy": "TIMEOUT", "precision": "TIMEOUT", 
                   "recall": "TIMEOUT", "f1_score": "TIMEOUT"}
        except Exception as e:
            print(f"❌ Error running {model_type} on {dataset}: {e}")
            return {"loss": "ERROR", "accuracy": "ERROR", "precision": "ERROR", 
                   "recall": "ERROR", "f1_score": "ERROR"}
    
    def create_rf_config(self, dataset: str, rounds: int):
        """Create a temporary configuration for Random Forest experiments."""
        rf_config = f"""[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "fed-learning"
version = "1.0.0"
description = "Federated Random Forest with Iris and Adult datasets"
license = "Apache-2.0"
dependencies = [
    "flwr[simulation]>=1.19.0",
    "scikit-learn>=1.3.0",          
    "pandas>=2.0.0",
]

[tool.hatch.build.targets.wheel]
packages = ["fed_learning"]

[tool.flwr.app]
publisher = "lenaschill"

[tool.flwr.app.components]
serverapp = "fed_learning.rf_server_app:rf_app"
clientapp = "fed_learning.rf_client_app:rf_app"

[tool.flwr.app.config]
num-server-rounds = {rounds}
n-estimators = 10
max-depth = 5
dataset = "{dataset}"

[tool.flwr.federations]
default = "local-simulation"

[tool.flwr.federations.local-simulation]
options.num-supernodes = 10
"""
        # Backup original and write RF config
        if os.path.exists("pyproject.toml"):
            os.rename("pyproject.toml", "pyproject.toml.backup")
        
        with open("pyproject.toml", "w") as f:
            f.write(rf_config)
    
    def restore_lr_config(self):
        """Restore the original Logistic Regression configuration."""
        if os.path.exists("pyproject.toml.backup"):
            os.rename("pyproject.toml.backup", "pyproject.toml")
    
    def parse_log_file(self, log_file: Path, target_round: int) -> Dict[str, str]:
        """Parse metrics from log file for the target round."""
        try:
            with open(log_file, 'r') as f:
                content = f.read()
            
            # Extract loss
            loss_pattern = rf"round {target_round}: ([\d.]+)"
            loss_match = re.search(loss_pattern, content)
            loss = loss_match.group(1) if loss_match else "NA"
            
            # Extract evaluate metrics (test metrics)
            eval_section_match = re.search(
                r"History \(metrics, distributed, evaluate\):(.*?)(?=History|$)", 
                content, 
                re.DOTALL
            )
            
            if eval_section_match:
                eval_section = eval_section_match.group(1)
                
                # Extract metrics for target round
                metrics = {}
                for metric in ["Accuracy", "Precision", "Recall", "F1_Score"]:
                    pattern = rf"'{metric}':[^[]*\[([^]]+)\]"
                    match = re.search(pattern, eval_section)
                    if match:
                        # Find the value for target round
                        round_pattern = rf"\({target_round}, ([\d.]+)\)"
                        round_match = re.search(round_pattern, match.group(1))
                        if round_match:
                            metrics[metric.lower().replace("_", "_")] = round_match.group(1)
                        else:
                            metrics[metric.lower().replace("_", "_")] = "NA"
                    else:
                        metrics[metric.lower().replace("_", "_")] = "NA"
                
                return {
                    "loss": loss,
                    "accuracy": metrics.get("accuracy", "NA"),
                    "precision": metrics.get("precision", "NA"),
                    "recall": metrics.get("recall", "NA"),
                    "f1_score": metrics.get("f1_score", "NA")
                }
            
            return {"loss": loss, "accuracy": "NA", "precision": "NA", "recall": "NA", "f1_score": "NA"}
            
        except Exception as e:
            print(f"❌ Error parsing log file {log_file}: {e}")
            return {"loss": "ERROR", "accuracy": "ERROR", "precision": "ERROR", 
                   "recall": "ERROR", "f1_score": "ERROR"}
    
    def save_results(self, model_type: str, dataset: str, metrics: Dict[str, str]):
        """Save results to CSV file."""
        new_row = {
            "model": model_type,
            "dataset": dataset,
            **metrics
        }
        
        # Append to CSV
        df = pd.read_csv(self.csv_file)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(self.csv_file, index=False)
    
    def run_all_experiments(self):
        """Run all combinations of models and datasets."""
        models = ["logistic_regression", "random_forest"]
        datasets = ["iris", "adult"]
        rounds = 5
        
        print("🚀 Starting federated learning experiments")
        print("=" * 50)
        
        for model in models:
            for dataset in datasets:
                try:
                    metrics = self.run_experiment(model, dataset, rounds)
                    self.save_results(model, dataset, metrics)
                    print(f"✅ Completed {model} on {dataset}")
                    print(f"   Results: {metrics}")
                    print("-" * 30)
                    
                finally:
                    # Always restore config after RF experiments
                    if model == "random_forest":
                        self.restore_lr_config()
        
        print("\n🎉 All experiments completed!")
        print(f"📊 Results saved to: {self.csv_file}")
        
        # Display final results
        self.display_results()
    
    def display_results(self):
        """Display the final results table."""
        try:
            df = pd.read_csv(self.csv_file)
            print("\n📊 FINAL RESULTS:")
            print("=" * 80)
            print(df.to_string(index=False))
        except Exception as e:
            print(f"❌ Error displaying results: {e}")


def main():
    """Main function to run all experiments."""
    runner = FederatedLearningRunner()
    runner.run_all_experiments()


if __name__ == "__main__":
    main()