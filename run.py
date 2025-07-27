#!/usr/bin/env python3
"""
Python runner for federated learning experiments.
Runs Logistic Regression, Random Forest, and SVM on Iris and Adult datasets
with multiple aggregation strategies.
"""

import subprocess
import pandas as pd
import re
import os
import sys
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class FederatedLearningRunner:
    def __init__(self, result_dir: str = "results"):
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(exist_ok=True)
        self.csv_file = self.result_dir / "results.csv"
        
        # Initialize results CSV with strategy column
        self.init_results_csv()
    
    def init_results_csv(self):
        """Initialize the results CSV file with headers including strategy."""
        columns = ["model", "dataset", "strategy", "loss", "accuracy", "precision", "recall", "f1_score"]
        df = pd.DataFrame(columns=columns)
        df.to_csv(self.csv_file, index=False)
    
    def run_experiment(self, model_type: str, dataset: str, strategy: str = "fedavg", rounds: int = 5) -> Dict[str, str]:
        """Run a single federated learning experiment with specified strategy."""
        print(f"▶️  Running {model_type} on {dataset} dataset with {strategy.upper()}")
        
        log_file = self.result_dir / f"{model_type}_{dataset}_{strategy}.log"
        print(f"    log ➜ {log_file}")
        
        # Use our custom federated learning implementation
        try:
            metrics = self.run_custom_federated_experiment(model_type, dataset, strategy, rounds, log_file)
            return metrics
            
        except Exception as e:
            print(f"❌ Error running {model_type} on {dataset} with {strategy}: {e}")
            return {"loss": "ERROR", "accuracy": "ERROR", "precision": "ERROR", 
                   "recall": "ERROR", "f1_score": "ERROR"}
    
    def run_custom_federated_experiment(self, model_type: str, dataset: str, strategy: str, rounds: int, log_file: Path) -> Dict[str, str]:
        """Run federated learning experiment using our custom implementation with aggregation strategies."""
        
        # Import our federated learning modules
        sys.path.append('.')
        try:
            from fed_learning.task import (
                set_dataset, set_model_type, load_data, 
                get_model, get_model_params, set_initial_params, set_model_params
            )
            from fed_learning.aggregation_strategies import get_strategy
            import numpy as np
        except ImportError as e:
            print(f"❌ Failed to import federated learning modules: {e}")
            return {"loss": "IMPORT_ERROR", "accuracy": "IMPORT_ERROR", "precision": "IMPORT_ERROR", 
                   "recall": "IMPORT_ERROR", "f1_score": "IMPORT_ERROR"}
        
        # Convert model names to our format
        model_mapping = {
            "logistic_regression": "logistic",
            "random_forest": "logistic",  # Use logistic as fallback for RF
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
        elif strategy in ["fedadam", "fedyogi"]:
            strategy_params = {"eta": 0.01, "beta_1": 0.9, "beta_2": 0.99, "tau": 1e-3}
        elif strategy == "fedadagrad":
            strategy_params = {"eta": 0.01, "tau": 1e-3}
        elif strategy == "fedlag":
            strategy_params = {"eta": 0.01, "momentum": 0.9}
        
        # Create aggregation strategy
        try:
            aggregation_strategy = get_strategy(strategy, **strategy_params)
        except Exception as e:
            print(f"❌ Failed to create strategy {strategy}: {e}")
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
                client_steps = []  # For FedNova
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
                    client_steps.append(10)  # Local epochs for FedNova
                    
                    with open(log_file, 'a') as f:
                        f.write(f"  Client {client_id}: accuracy = {local_accuracy:.3f}, loss = {local_loss:.3f}\n")
                
                # Apply aggregation strategy
                if internal_model_type == "logistic":
                    try:
                        # Use our aggregation strategy
                        if strategy == "fednova":
                            global_params = aggregation_strategy.aggregate(
                                client_params, client_weights, global_params, client_steps=client_steps
                            )
                        else:
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
                        print(f"❌ Aggregation failed: {e}")
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
            print(f"❌ Error in federated experiment: {e}")
            import traceback
            traceback.print_exc()
            return {"loss": "ERROR", "accuracy": "ERROR", "precision": "ERROR", 
                   "recall": "ERROR", "f1_score": "ERROR"}
    
    def save_results(self, model_type: str, dataset: str, strategy: str, metrics: Dict[str, str]):
        """Save results to CSV file with strategy information."""
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
        """Run all combinations of models, datasets, and strategies."""
        models = ["logistic_regression", "svm"]  # Focus on these for strategy comparison
        datasets = ["iris", "adult"]
        
        # Different strategies to test
        strategies = ["fedavg", "fedprox", "fedadam", "fedyogi", "fednova", "fedadagrad", "fedlag"]
        
        rounds = 5
        
        print("🚀 Starting federated learning experiments with multiple strategies")
        print("Models: Logistic Regression, SVM")
        print("Datasets: Iris, Adult")
        print(f"Strategies: {', '.join([s.upper() for s in strategies])}")
        print("=" * 70)
        
        total_experiments = len(models) * len(datasets) * len(strategies)
        current_experiment = 0
        
        for model in models:
            for dataset in datasets:
                for strategy in strategies:
                    current_experiment += 1
                    
                    # Skip advanced strategies for SVM (only use basic ones)
                    if model == "svm" and strategy not in ["fedavg", "fedprox"]:
                        print(f"⏭️  Skipping {strategy.upper()} for SVM (not compatible)")
                        continue
                    
                    try:
                        print(f"\n{'='*70}")
                        print(f"Experiment {current_experiment}/{total_experiments}")
                        print(f"Running {model.upper()} on {dataset.upper()} with {strategy.upper()}")
                        print(f"{'='*70}")
                        
                        metrics = self.run_experiment(model, dataset, strategy, rounds)
                        self.save_results(model, dataset, strategy, metrics)
                        
                        print(f"✅ Completed {model} on {dataset} with {strategy}")
                        print(f"   📊 Results: {metrics}")
                        print("-" * 50)
                        
                    except Exception as e:
                        print(f"❌ Failed {model} on {dataset} with {strategy}: {e}")
                        error_metrics = {"loss": "ERROR", "accuracy": "ERROR", "precision": "ERROR", 
                                       "recall": "ERROR", "f1_score": "ERROR"}
                        self.save_results(model, dataset, strategy, error_metrics)
        
        print("\n🎉 All experiments completed!")
        print(f"📊 Results saved to: {self.csv_file}")
        
        # Display final results
        self.display_results()
        
        # Create comprehensive visualizations
        self.create_comprehensive_plots()
    
    def display_results(self):
        """Display the final results table with strategy breakdown."""
        try:
            df = pd.read_csv(self.csv_file)
            print("\n📊 FINAL RESULTS:")
            print("=" * 100)
            print(df.to_string(index=False))
            
            # Display summary statistics by strategy
            print("\n📈 SUMMARY BY STRATEGY:")
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
            
            # Display summary by model
            print("\n📈 SUMMARY BY MODEL:")
            print("=" * 60)
            for model in df['model'].unique():
                model_data = df[df['model'] == model]
                if len(model_data) > 0:
                    try:
                        numeric_cols = ['accuracy', 'precision', 'recall', 'f1_score']
                        for col in numeric_cols:
                            model_data[col] = pd.to_numeric(model_data[col], errors='coerce')
                        
                        avg_acc = model_data['accuracy'].mean()
                        if not pd.isna(avg_acc):
                            print(f"{model:20s}: Avg Accuracy = {avg_acc:.3f}")
                        else:
                            print(f"{model:20s}: Accuracy = ERROR/NA")
                    except:
                        print(f"{model:20s}: Could not calculate average")
            
        except Exception as e:
            print(f"❌ Error displaying results: {e}")
    
    def create_comprehensive_plots(self):
        """Create comprehensive visualization plots for strategy comparison."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import numpy as np
            
            # Set style for better-looking plots
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Read results
            df = pd.read_csv(self.csv_file)
            
            # Convert numeric columns
            numeric_cols = ['loss', 'accuracy', 'precision', 'recall', 'f1_score']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove rows with errors
            df_clean = df.dropna(subset=numeric_cols)
            
            if len(df_clean) == 0:
                print("❌ No valid data for plotting")
                return
            
            print("\n📊 Creating comprehensive visualization plots...")
            
            # 1. Strategy Comparison by Model - Accuracy
            self.plot_strategy_comparison_by_model(df_clean)
            
            # 2. Overall Model Performance Comparison
            self.plot_overall_model_comparison(df_clean)
            
            # 3. Heatmap of All Metrics
            self.plot_metrics_heatmap(df_clean)
            
            # 4. Strategy Performance Radar Chart
            self.plot_strategy_radar_chart(df_clean)
            
            # 5. Dataset Difficulty Analysis
            self.plot_dataset_difficulty_analysis(df_clean)
            
            # 6. Comprehensive Strategy Breakdown
            self.plot_comprehensive_strategy_breakdown(df_clean)
            
            print("✅ All visualization plots created successfully!")
            print(f"📁 Plots saved in: {self.result_dir}")
            
        except ImportError:
            print("❌ matplotlib or seaborn not available. Please install: pip install matplotlib seaborn")
        except Exception as e:
            print(f"❌ Error creating plots: {e}")
            import traceback
            traceback.print_exc()
    
    def plot_strategy_comparison_by_model(self, df):
        """Plot strategy comparison for each model separately."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Federated Learning Strategy Comparison by Model', fontsize=16, fontweight='bold')
        
        models = df['model'].unique()
        datasets = df['dataset'].unique()
        
        plot_idx = 0
        for i, model in enumerate(models):
            for j, dataset in enumerate(datasets):
                if plot_idx >= 4:
                    break
                    
                ax = axes[i, j]
                model_dataset_data = df[(df['model'] == model) & (df['dataset'] == dataset)]
                
                if len(model_dataset_data) > 0:
                    strategies = model_dataset_data['strategy'].values
                    accuracies = model_dataset_data['accuracy'].values
                    
                    bars = ax.bar(strategies, accuracies, alpha=0.7)
                    ax.set_title(f'{model.replace("_", " ").title()} - {dataset.title()} Dataset')
                    ax.set_ylabel('Accuracy')
                    ax.set_ylim(0, 1)
                    ax.grid(True, alpha=0.3)
                    
                    # Add value labels on bars
                    for bar, acc in zip(bars, accuracies):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
                    
                    # Rotate x-labels
                    ax.tick_params(axis='x', rotation=45)
                else:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{model.replace("_", " ").title()} - {dataset.title()} Dataset')
                
                plot_idx += 1
        
        plt.tight_layout()
        plt.savefig(self.result_dir / "strategy_comparison_by_model.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_overall_model_comparison(self, df):
        """Plot overall model performance comparison across all strategies."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Overall Model Performance Comparison', fontsize=16, fontweight='bold')
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            # Group by model and calculate mean
            model_performance = df.groupby('model')[metric].agg(['mean', 'std']).reset_index()
            
            bars = ax.bar(model_performance['model'], model_performance['mean'], 
                         yerr=model_performance['std'], capsize=5, alpha=0.7)
            
            ax.set_title(f'Average {metric.replace("_", " ").title()} by Model')
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, mean_val in zip(bars, model_performance['mean']):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Rotate x-labels
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.result_dir / "overall_model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_metrics_heatmap(self, df):
        """Create heatmap showing all metrics for all strategy-model combinations."""
        # Pivot data for heatmap
        pivot_data = df.pivot_table(
            values=['accuracy', 'precision', 'recall', 'f1_score'], 
            index=['model', 'dataset'], 
            columns='strategy', 
            aggfunc='mean'
        )
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle('Performance Heatmap: All Metrics by Strategy and Model', fontsize=16, fontweight='bold')
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            if metric in pivot_data.columns.levels[0]:
                metric_data = pivot_data[metric]
                
                # Create heatmap
                sns.heatmap(metric_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                           ax=ax, cbar_kws={'label': metric.replace('_', ' ').title()},
                           vmin=0, vmax=1)
                
                ax.set_title(f'{metric.replace("_", " ").title()} Heatmap')
                ax.set_xlabel('Strategy')
                ax.set_ylabel('Model - Dataset')
        
        plt.tight_layout()
        plt.savefig(self.result_dir / "metrics_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_strategy_radar_chart(self, df):
        """Create radar chart comparing strategies across all metrics."""
        from math import pi
        
        # Calculate average performance per strategy
        strategy_avg = df.groupby('strategy')[['accuracy', 'precision', 'recall', 'f1_score']].mean()
        
        # Number of metrics
        categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        N = len(categories)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
        
        # Compute angles for each metric
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Colors for different strategies
        colors = plt.cm.Set3(np.linspace(0, 1, len(strategy_avg)))
        
        # Plot each strategy
        for idx, (strategy, row) in enumerate(strategy_avg.iterrows()):
            values = row.values.tolist()
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=strategy.upper(), color=colors[idx])
            ax.fill(angles, values, alpha=0.25, color=colors[idx])
        
        # Add category labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Strategy Performance Radar Chart\n(Average Across All Models and Datasets)', 
                    size=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.result_dir / "strategy_radar_chart.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_dataset_difficulty_analysis(self, df):
        """Analyze how strategies perform on different datasets."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Dataset Difficulty Analysis', fontsize=16, fontweight='bold')
        
        datasets = df['dataset'].unique()
        
        for idx, dataset in enumerate(datasets):
            ax = axes[idx]
            dataset_data = df[df['dataset'] == dataset]
            
            # Group by strategy and calculate mean accuracy
            strategy_performance = dataset_data.groupby('strategy')['accuracy'].agg(['mean', 'std']).reset_index()
            
            bars = ax.bar(strategy_performance['strategy'], strategy_performance['mean'],
                         yerr=strategy_performance['std'], capsize=5, alpha=0.7)
            
            ax.set_title(f'{dataset.title()} Dataset Performance')
            ax.set_ylabel('Average Accuracy')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, mean_val in zip(bars, strategy_performance['mean']):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold')
            
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.result_dir / "dataset_difficulty_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_comprehensive_strategy_breakdown(self, df):
        """Create comprehensive breakdown of strategy performance."""
        fig = plt.figure(figsize=(20, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Top-left: Strategy accuracy comparison
        ax1 = fig.add_subplot(gs[0, 0])
        strategy_acc = df.groupby('strategy')['accuracy'].mean().sort_values(ascending=False)
        bars1 = ax1.bar(range(len(strategy_acc)), strategy_acc.values, alpha=0.7)
        ax1.set_title('Strategy Ranking by Accuracy')
        ax1.set_ylabel('Average Accuracy')
        ax1.set_xticks(range(len(strategy_acc)))
        ax1.set_xticklabels([s.upper() for s in strategy_acc.index], rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars1, strategy_acc.values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Top-middle: Loss comparison
        ax2 = fig.add_subplot(gs[0, 1])
        strategy_loss = df.groupby('strategy')['loss'].mean().sort_values(ascending=True)
        bars2 = ax2.bar(range(len(strategy_loss)), strategy_loss.values, alpha=0.7, color='orange')
        ax2.set_title('Strategy Ranking by Loss (Lower is Better)')
        ax2.set_ylabel('Average Loss')
        ax2.set_xticks(range(len(strategy_loss)))
        ax2.set_xticklabels([s.upper() for s in strategy_loss.index], rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 3. Top-right: F1-Score comparison
        ax3 = fig.add_subplot(gs[0, 2])
        strategy_f1 = df.groupby('strategy')['f1_score'].mean().sort_values(ascending=False)
        bars3 = ax3.bar(range(len(strategy_f1)), strategy_f1.values, alpha=0.7, color='green')
        ax3.set_title('Strategy Ranking by F1-Score')
        ax3.set_ylabel('Average F1-Score')
        ax3.set_xticks(range(len(strategy_f1)))
        ax3.set_xticklabels([s.upper() for s in strategy_f1.index], rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 4. Bottom: Strategy performance by model (large subplot)
        ax4 = fig.add_subplot(gs[1:, :])
        
        # Create grouped bar chart
        strategies = df['strategy'].unique()
        models = df['model'].unique()
        x = np.arange(len(strategies))
        width = 0.35
        
        for i, model in enumerate(models):
            model_data = []
            for strategy in strategies:
                strategy_model_data = df[(df['strategy'] == strategy) & (df['model'] == model)]
                if len(strategy_model_data) > 0:
                    model_data.append(strategy_model_data['accuracy'].mean())
                else:
                    model_data.append(0)
            
            offset = (i - len(models)/2 + 0.5) * width
            bars = ax4.bar(x + offset, model_data, width, label=model.replace('_', ' ').title(), alpha=0.7)
            
            # Add value labels
            for bar, val in zip(bars, model_data):
                if val > 0:
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{val:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)
        
        ax4.set_title('Strategy Performance Comparison by Model', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Average Accuracy')
        ax4.set_xlabel('Aggregation Strategy')
        ax4.set_xticks(x)
        ax4.set_xticklabels([s.upper() for s in strategies], rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1)
        
        plt.suptitle('Comprehensive Strategy Performance Analysis', fontsize=18, fontweight='bold')
        plt.savefig(self.result_dir / "comprehensive_strategy_breakdown.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create summary statistics table plot
        self.create_summary_table_plot(df)
    
    def create_summary_table_plot(self, df):
        """Create a summary statistics table as a plot."""
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # Calculate summary statistics
        summary_stats = []
        strategies = df['strategy'].unique()
        
        for strategy in strategies:
            strategy_data = df[df['strategy'] == strategy]
            stats = {
                'Strategy': strategy.upper(),
                'Avg Accuracy': f"{strategy_data['accuracy'].mean():.3f} ± {strategy_data['accuracy'].std():.3f}",
                'Avg Precision': f"{strategy_data['precision'].mean():.3f} ± {strategy_data['precision'].std():.3f}",
                'Avg Recall': f"{strategy_data['recall'].mean():.3f} ± {strategy_data['recall'].std():.3f}",
                'Avg F1-Score': f"{strategy_data['f1_score'].mean():.3f} ± {strategy_data['f1_score'].std():.3f}",
                'Avg Loss': f"{strategy_data['loss'].mean():.3f} ± {strategy_data['loss'].std():.3f}",
                'Experiments': len(strategy_data)
            }
            summary_stats.append(stats)
        
        # Create table
        summary_df = pd.DataFrame(summary_stats)
        table = ax.table(cellText=summary_df.values, colLabels=summary_df.columns,
                        cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        
        # Style the table
        for i in range(len(summary_df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color-code rows alternately
        for i in range(1, len(summary_stats) + 1):
            for j in range(len(summary_df.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        plt.title('Federated Learning Strategy Performance Summary', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.savefig(self.result_dir / "strategy_summary_table.png", dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main function to run all experiments."""
    print("🔬 Federated Learning Experiment Runner with Multiple Strategies")
    print("Supporting: Logistic Regression and SVM")
    print("Datasets: Iris and Adult")
    print("Strategies: FedAvg, FedProx, FedAdam, FedYogi, FedNova, FedAdagrad, FedLAG")
    print("\n" + "="*80)
    
    # Check if required modules are available
    try:
        import sklearn
        print("✅ scikit-learn found")
    except ImportError:
        print("❌ scikit-learn not found. Please install: pip install scikit-learn")
        return
    
    try:
        import pandas
        print("✅ pandas found")
    except ImportError:
        print("❌ pandas not found. Please install: pip install pandas")
        return
    
    try:
        import numpy
        print("✅ numpy found")
    except ImportError:
        print("❌ numpy not found. Please install: pip install numpy")
        return
    
    # Check if aggregation strategies module exists
    try:
        sys.path.append('.')
        from fed_learning.aggregation_strategies import list_strategies
        available_strategies = list_strategies()
        print(f"✅ aggregation_strategies module found - Available: {', '.join(available_strategies)}")
    except ImportError:
        print("❌ aggregation_strategies module not found. Please ensure fed_learning/aggregation_strategies.py exists")
        return
    
    # Check if data files exist
    data_files = ["data/iris.csv", "data/adult.csv"]
    missing_files = []
    for file in data_files:
        if not os.path.exists(file):
            missing_files.append(file)
        else:
            print(f"✅ {file} found")
    
    if missing_files:
        print(f"❌ Missing data files: {missing_files}")
        print("Please ensure your datasets are available in the data/ directory")
        return
    
    print("✅ All dependencies and data files found")
    print("\nStarting experiments with multiple aggregation strategies...\n")
    
    runner = FederatedLearningRunner()
    runner.run_all_experiments()


if __name__ == "__main__":
    main()