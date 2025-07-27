#!/usr/bin/env python3
"""
Centralized ML training for comparison with federated learning.
Trains Logistic Regression and Random Forest on full datasets.
"""

import numpy as np
import pandas as pd
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import shuffle
from pathlib import Path
from typing import Dict, Tuple


class CentralizedMLRunner:
    def __init__(self, result_dir: str = "results"):
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(exist_ok=True)
        self.csv_file = self.result_dir / "centralized_results.csv"
        
        self.datasets = {
            "iris": {
                "path": "data/iris.csv",
                "label_col": "Species"
            },
            "adult": {
                "path": "data/adult.csv", 
                "label_col": "income"
            }
        }
        
        self.init_results_csv()
    
    def init_results_csv(self):
        """Initialize the centralized results CSV file."""
        columns = ["model", "dataset", "loss", "accuracy", "precision", "recall", "f1_score"]
        df = pd.DataFrame(columns=columns)
        df.to_csv(self.csv_file, index=False)
    
    def load_and_preprocess_dataset(self, dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess dataset same way as federated version."""
        dataset_info = self.datasets[dataset_name]
        df = pd.read_csv(dataset_info["path"])
        label_col = dataset_info["label_col"]
        
        # Encode labels
        y = LabelEncoder().fit_transform(df[label_col])
        
        # One-hot categorical predictors, scale numerics
        X = pd.get_dummies(df.drop(columns=[label_col]))
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Shuffle to mix classes (same random_state as federated)
        X_final, y_final = shuffle(X_scaled.astype(np.float32), y, random_state=42)
        
        return X_final, y_final
    

    
    def train_logistic_regression(self, dataset_name: str) -> Dict[str, float]:
        """Train Logistic Regression on full dataset."""
        print(f"Training Logistic Regression on {dataset_name}")
        
        X, y = self.load_and_preprocess_dataset(dataset_name)
        
        # Same train/test split as federated (stratified 80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Use same hyperparameters as federated version
        model = LogisticRegression(
            penalty="l2",
            max_iter=5,  # local_epochs from federated config
            warm_start=True,
            multi_class="auto",
            solver="lbfgs"
        )
        
        # Train model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)
        
        # Calculate metrics
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        loss = log_loss(y_test, y_proba, labels=model.classes_)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        return {
            "loss": loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
    
    def train_random_forest(self, dataset_name: str) -> Dict[str, float]:
        """Train Random Forest on full dataset with default parameters."""
        print(f"Training Random Forest on {dataset_name}")
        
        X, y = self.load_and_preprocess_dataset(dataset_name)
        
        # Same train/test split as federated (stratified 80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Use default hyperparameters (no search)
        model = RandomForestClassifier(
            n_estimators=10,
            max_depth=5,
            min_samples_split=2,
            min_samples_leaf=1,
            bootstrap=True,
            random_state=42
        )
        
        # Train model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)
        
        # Calculate metrics
        y_pred = model.predict(X_test)
        
        # For Random Forest, use 1 - accuracy as loss (same as federated)
        accuracy = accuracy_score(y_test, y_pred)
        loss = 1.0 - accuracy
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        print(f"  Using default hyperparameters")
        
        return {
            "loss": loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
    
    def save_results(self, model_type: str, dataset: str, metrics: Dict[str, float]):
        """Save results to CSV file."""
        new_row = {
            "model": model_type,
            "dataset": dataset,
            **{k: round(v, 6) for k, v in metrics.items()}
        }
        
        # Append to CSV
        df = pd.read_csv(self.csv_file) if self.csv_file.exists() else pd.DataFrame()
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(self.csv_file, index=False)
    
    def run_all_experiments(self):
        """Run all centralized ML experiments."""
        models = ["logistic_regression", "random_forest"]
        datasets = ["iris", "adult"]
        
        print("Starting centralized ML experiments")
        print("=" * 50)
        
        for model in models:
            for dataset in datasets:
                try:
                    if model == "logistic_regression":
                        metrics = self.train_logistic_regression(dataset)
                    else:  # random_forest
                        metrics = self.train_random_forest(dataset)
                    
                    self.save_results(model, dataset, metrics)
                    print(f"Completed {model} on {dataset}")
                    print(f"  Results: {metrics}")
                    print("-" * 30)
                    
                except Exception as e:
                    print(f"Error running {model} on {dataset}: {e}")
                    error_metrics = {
                        "loss": 999.0, "accuracy": 0.0, "precision": 0.0,
                        "recall": 0.0, "f1_score": 0.0
                    }
                    self.save_results(model, dataset, error_metrics)
        
        print("\nCentralized ML experiments completed!")
        print(f"Results saved to: {self.csv_file}")
        
        # Display final results
        self.display_results()
    
    def display_results(self):
        """Display the centralized results table."""
        try:
            df = pd.read_csv(self.csv_file)
            print("\nFINAL RESULTS - Centralized ML:")
            print("=" * 80)
            print(df.to_string(index=False))
        except Exception as e:
            print(f"Error displaying results: {e}")


def main():
    """Main function to run centralized experiments."""
    runner = CentralizedMLRunner()
    runner.run_all_experiments()


if __name__ == "__main__":
    main()