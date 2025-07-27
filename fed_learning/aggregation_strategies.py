"""
Federated Learning Aggregation Strategies - UPDATED VERSION

This module implements various parameter aggregation strategies for federated learning,
including FedAvg, FedProx, FedMedian, and FedLAG.

Removed: FedAdam, FedYogi, FedAdagrad (replaced with FedMedian)
Added: FedMedian for robust aggregation against outliers
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class AggregationStrategy(ABC):
    """Abstract base class for federated learning aggregation strategies."""
    
    def __init__(self, **kwargs):
        self.round_num = 0
        self.history = []
    
    @abstractmethod
    def aggregate(self, 
                 client_params: List[List[np.ndarray]], 
                 client_weights: List[float], 
                 global_params: List[np.ndarray],
                 **kwargs) -> List[np.ndarray]:
        """Aggregate client parameters into new global parameters."""
        pass
    
    def update_round(self):
        """Update round number and history."""
        self.round_num += 1
    
    def reset(self):
        """Reset strategy state."""
        self.round_num = 0
        self.history = []


class FedAvgStrategy(AggregationStrategy):
    """
    Federated Averaging (FedAvg) Strategy
    
    Simply computes weighted average of client parameters.
    Reference: McMahan et al., "Communication-Efficient Learning of Deep Networks 
    from Decentralized Data", AISTATS 2017
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "FedAvg"
    
    def aggregate(self, 
                 client_params: List[List[np.ndarray]], 
                 client_weights: List[float], 
                 global_params: List[np.ndarray],
                 **kwargs) -> List[np.ndarray]:
        
        total_weight = sum(client_weights)
        num_params = len(client_params[0])
        
        # Weighted average of all parameters
        aggregated_params = []
        for param_idx in range(num_params):
            weighted_sum = np.zeros_like(client_params[0][param_idx])
            for client_idx, params in enumerate(client_params):
                weight = client_weights[client_idx] / total_weight
                weighted_sum += weight * params[param_idx]
            aggregated_params.append(weighted_sum)
        
        self.update_round()
        return aggregated_params


class FedProxStrategy(AggregationStrategy):
    """
    Federated Proximal (FedProx) Strategy
    
    Uses FedAvg aggregation but relies on proximal term during local training.
    The aggregation itself is identical to FedAvg.
    Reference: Li et al., "Federated Optimization in Heterogeneous Networks", MLSys 2020
    """
    
    def __init__(self, mu: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.mu = mu  # Proximal term coefficient
        self.name = "FedProx"
    
    def aggregate(self, 
                 client_params: List[List[np.ndarray]], 
                 client_weights: List[float], 
                 global_params: List[np.ndarray],
                 **kwargs) -> List[np.ndarray]:
        
        # FedProx uses same aggregation as FedAvg
        total_weight = sum(client_weights)
        num_params = len(client_params[0])
        
        aggregated_params = []
        for param_idx in range(num_params):
            weighted_sum = np.zeros_like(client_params[0][param_idx])
            for client_idx, params in enumerate(client_params):
                weight = client_weights[client_idx] / total_weight
                weighted_sum += weight * params[param_idx]
            aggregated_params.append(weighted_sum)
        
        self.update_round()
        return aggregated_params


class FedMedianStrategy(AggregationStrategy):
    """
    Federated Median (FedMedian) Strategy
    
    Uses coordinate-wise median for aggregation instead of weighted averaging.
    More robust to outliers and Byzantine failures than FedAvg.
    Reference: Yin et al., "Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates", ICML 2018
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "FedMedian"
    
    def aggregate(self, 
                 client_params: List[List[np.ndarray]], 
                 client_weights: List[float], 
                 global_params: List[np.ndarray],
                 **kwargs) -> List[np.ndarray]:
        
        num_params = len(client_params[0])
        num_clients = len(client_params)
        
        # For each parameter, compute coordinate-wise median
        aggregated_params = []
        for param_idx in range(num_params):
            # Stack all client parameters for this parameter index
            param_shape = client_params[0][param_idx].shape
            param_stack = np.zeros((num_clients,) + param_shape)
            
            for client_idx, params in enumerate(client_params):
                param_stack[client_idx] = params[param_idx]
            
            # Compute coordinate-wise median along the client axis (axis=0)
            median_param = np.median(param_stack, axis=0)
            aggregated_params.append(median_param)
        
        self.update_round()
        return aggregated_params


class FedLAGStrategy(AggregationStrategy):
    """
    Federated Learning with Gradient Tracking (FedLAG) Strategy
    
    Maintains a running estimate of the global gradient to correct for client drift.
    Includes parameter clipping for stability.
    Reference: Inspired by LAG (Local and Global) methods
    """
    
    def __init__(self, eta: float = 0.01, momentum: float = 0.9, clip_norm: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.eta = eta
        self.momentum = momentum
        self.clip_norm = clip_norm
        self.name = "FedLAG"
        
        # Gradient tracking state
        self.global_grad_estimate = None
        self.velocity = None
    
    def aggregate(self, 
                 client_params: List[List[np.ndarray]], 
                 client_weights: List[float], 
                 global_params: List[np.ndarray],
                 **kwargs) -> List[np.ndarray]:
        
        # Initialize state if first round
        if self.global_grad_estimate is None:
            self.global_grad_estimate = [np.zeros_like(param) for param in global_params]
            self.velocity = [np.zeros_like(param) for param in global_params]
        
        # Step 1: Compute weighted average
        total_weight = sum(client_weights)
        num_params = len(client_params[0])
        
        averaged_params = []
        for param_idx in range(num_params):
            weighted_sum = np.zeros_like(client_params[0][param_idx])
            for client_idx, params in enumerate(client_params):
                weight = client_weights[client_idx] / total_weight
                weighted_sum += weight * params[param_idx]
            averaged_params.append(weighted_sum)
        
        # Step 2: Update global gradient estimate and apply momentum with clipping
        new_params = []
        for i in range(num_params):
            # Current "gradient" (parameter change)
            current_grad = averaged_params[i] - global_params[i]
            
            # Update global gradient estimate with momentum
            self.global_grad_estimate[i] = (
                self.momentum * self.global_grad_estimate[i] + 
                (1 - self.momentum) * current_grad
            )
            
            # Update velocity
            self.velocity[i] = (
                self.momentum * self.velocity[i] + 
                self.eta * self.global_grad_estimate[i]
            )
            
            # Apply update with clipping
            update = self.velocity[i]
            update_norm = np.linalg.norm(update)
            if update_norm > self.clip_norm:
                update = update * (self.clip_norm / update_norm)
            
            new_param = global_params[i] + update
            new_params.append(new_param)
        
        self.update_round()
        return new_params


# Strategy registry for easy access
AGGREGATION_STRATEGIES = {
    'fedavg': FedAvgStrategy,
    'fedprox': FedProxStrategy,
    'fedmedian': FedMedianStrategy,
    'fedlag': FedLAGStrategy,
}


def get_strategy_params_for_dataset(strategy_name: str, dataset: str) -> dict:
    """
    Get dataset-specific parameters for strategies to improve stability.
    
    Args:
        strategy_name: Name of the strategy
        dataset: Name of the dataset
        
    Returns:
        Dictionary of strategy-specific parameters
    """
    
    if dataset.lower() in ["adult", "wine_quality"]:
        # More conservative parameters for complex datasets
        strategy_params = {
            "fedprox": {"mu": 0.001},  # Much smaller regularization
            "fedmedian": {},  # No specific parameters for FedMedian
            "fedlag": {"eta": 0.001, "momentum": 0.5, "clip_norm": 0.5}
        }
    else:
        # Standard parameters for simpler datasets like Iris
        strategy_params = {
            "fedprox": {"mu": 0.01},
            "fedmedian": {},  # No specific parameters for FedMedian
            "fedlag": {"eta": 0.01, "momentum": 0.9, "clip_norm": 1.0}
        }
    
    return strategy_params.get(strategy_name.lower(), {})


def get_strategy(strategy_name: str, dataset: str = None, **kwargs) -> AggregationStrategy:
    """
    Factory function to get aggregation strategy by name with dataset-specific parameters.
    
    Args:
        strategy_name: Name of the strategy (case-insensitive)
        dataset: Name of the dataset (for parameter optimization)
        **kwargs: Strategy-specific parameters (override defaults)
    
    Returns:
        AggregationStrategy instance
    
    Raises:
        ValueError: If strategy name is not recognized
    """
    strategy_name = strategy_name.lower()
    if strategy_name not in AGGREGATION_STRATEGIES:
        available = ', '.join(AGGREGATION_STRATEGIES.keys())
        raise ValueError(f"Unknown strategy '{strategy_name}'. Available: {available}")
    
    # Get dataset-specific parameters
    if dataset:
        default_params = get_strategy_params_for_dataset(strategy_name, dataset)
        # Override defaults with any provided kwargs
        default_params.update(kwargs)
        kwargs = default_params
    
    return AGGREGATION_STRATEGIES[strategy_name](**kwargs)


def list_strategies() -> List[str]:
    """Return list of available aggregation strategy names."""
    return list(AGGREGATION_STRATEGIES.keys())


# Example usage and testing
if __name__ == "__main__":
    # Example: Test different strategies
    np.random.seed(42)
    
    # Mock client parameters (3 clients, 2 parameter arrays each)
    client_params = [
        [np.random.randn(5, 3), np.random.randn(3)],  # Client 0
        [np.random.randn(5, 3), np.random.randn(3)],  # Client 1  
        [np.random.randn(5, 3), np.random.randn(3)],  # Client 2
    ]
    client_weights = [100, 150, 200]  # Data sizes
    global_params = [np.zeros((5, 3)), np.zeros(3)]  # Initial global parameters
    
    print("Testing Aggregation Strategies")
    print("=" * 40)
    
    for strategy_name in list_strategies():
        print(f"\nTesting {strategy_name.upper()}")
        
        try:
            # Test with dataset-specific parameters
            strategy = get_strategy(strategy_name, dataset="adult")
            result = strategy.aggregate(client_params, client_weights, global_params)
            
            # Print some basic stats
            param_norms = [np.linalg.norm(param) for param in result]
            print(f"   Success - Parameter norms: {[f'{norm:.3f}' for norm in param_norms]}")
            
        except Exception as e:
            print(f"   Error: {e}")