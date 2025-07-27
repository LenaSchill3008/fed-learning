"""
Federated Learning Aggregation Strategies - FIXED VERSION

This module implements various parameter aggregation strategies for federated learning,
including FedAvg, FedProx, FedAdam, FedYogi, and more advanced techniques.

Fixes included:
- Parameter clipping for adaptive methods to prevent divergence
- Dataset-specific hyperparameters for better stability
- Improved error handling and numerical stability
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


class FedAdamStrategy(AggregationStrategy):
    """
    Federated Adam (FedAdam) Strategy
    
    Applies Adam optimizer at the server level for parameter aggregation.
    Includes parameter clipping to prevent divergence on complex datasets.
    Reference: Reddi et al., "Adaptive Federated Optimization", ICLR 2021
    """
    
    def __init__(self, 
                 eta: float = 0.01,
                 beta_1: float = 0.9, 
                 beta_2: float = 0.99,
                 tau: float = 1e-3,
                 clip_norm: float = 1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.eta = eta          # Server learning rate
        self.beta_1 = beta_1    # First moment decay
        self.beta_2 = beta_2    # Second moment decay  
        self.tau = tau          # Small constant for numerical stability
        self.clip_norm = clip_norm  # Gradient clipping threshold
        self.name = "FedAdam"
        
        # Adam state variables (initialized when first called)
        self.m_t = None  # First moment estimates
        self.v_t = None  # Second moment estimates
    
    def aggregate(self, 
                 client_params: List[List[np.ndarray]], 
                 client_weights: List[float], 
                 global_params: List[np.ndarray],
                 **kwargs) -> List[np.ndarray]:
        
        # Step 1: Compute weighted average (pseudo-gradient)
        total_weight = sum(client_weights)
        num_params = len(client_params[0])
        
        # Initialize Adam state if first round
        if self.m_t is None:
            self.m_t = [np.zeros_like(param) for param in global_params]
            self.v_t = [np.full_like(param, self.tau) for param in global_params]
        
        # Compute weighted average of client parameters
        averaged_params = []
        for param_idx in range(num_params):
            weighted_sum = np.zeros_like(client_params[0][param_idx])
            for client_idx, params in enumerate(client_params):
                weight = client_weights[client_idx] / total_weight
                weighted_sum += weight * params[param_idx]
            averaged_params.append(weighted_sum)
        
        # Step 2: Compute pseudo-gradient (server drift)
        pseudo_gradients = []
        for param_idx in range(num_params):
            pseudo_grad = averaged_params[param_idx] - global_params[param_idx]
            pseudo_gradients.append(pseudo_grad)
        
        # Step 3: Apply Adam updates with clipping
        new_params = []
        for i, pseudo_grad in enumerate(pseudo_gradients):
            # Update biased first moment estimate
            self.m_t[i] = self.beta_1 * self.m_t[i] + (1 - self.beta_1) * pseudo_grad
            
            # Update biased second moment estimate
            self.v_t[i] = self.beta_2 * self.v_t[i] + (1 - self.beta_2) * np.square(pseudo_grad)
            
            # Bias correction
            m_hat = self.m_t[i] / (1 - self.beta_1 ** (self.round_num + 1))
            v_hat = self.v_t[i] / (1 - self.beta_2 ** (self.round_num + 1))
            
            # Parameter update with clipping
            update = self.eta * m_hat / (np.sqrt(v_hat) + self.tau)
            
            # Clip large updates to prevent divergence
            update_norm = np.linalg.norm(update)
            if update_norm > self.clip_norm:
                update = update * (self.clip_norm / update_norm)
            
            new_param = global_params[i] + update
            new_params.append(new_param)
        
        self.update_round()
        return new_params


class FedYogiStrategy(AggregationStrategy):
    """
    Federated Yogi (FedYogi) Strategy - FIXED VERSION
    
    Applies Yogi optimizer at the server level. Similar to FedAdam but with
    different second moment update rule. Includes parameter clipping.
    Reference: Reddi et al., "Adaptive Federated Optimization", ICLR 2021
    """
    
    def __init__(self, 
                 eta: float = 0.01,
                 beta_1: float = 0.9, 
                 beta_2: float = 0.99,
                 tau: float = 1e-3,
                 clip_norm: float = 1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.eta = eta
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.tau = tau
        self.clip_norm = clip_norm
        self.name = "FedYogi"
        
        # Yogi state variables
        self.m_t = None
        self.v_t = None
    
    def aggregate(self, 
                 client_params: List[List[np.ndarray]], 
                 client_weights: List[float], 
                 global_params: List[np.ndarray],
                 **kwargs) -> List[np.ndarray]:
        
        # Step 1: Compute weighted average
        total_weight = sum(client_weights)
        num_params = len(client_params[0])
        
        # Initialize Yogi state if first round
        if self.m_t is None:
            self.m_t = [np.zeros_like(param) for param in global_params]
            self.v_t = [np.full_like(param, self.tau) for param in global_params]
        
        # Compute weighted average of client parameters
        averaged_params = []
        for param_idx in range(num_params):
            weighted_sum = np.zeros_like(client_params[0][param_idx])
            for client_idx, params in enumerate(client_params):
                weight = client_weights[client_idx] / total_weight
                weighted_sum += weight * params[param_idx]
            averaged_params.append(weighted_sum)
        
        # Step 2: Compute pseudo-gradient
        pseudo_gradients = []
        for param_idx in range(num_params):
            pseudo_grad = averaged_params[param_idx] - global_params[param_idx]
            pseudo_gradients.append(pseudo_grad)
        
        # Step 3: Apply Yogi updates with clipping
        new_params = []
        for i, pseudo_grad in enumerate(pseudo_gradients):
            # Update biased first moment estimate
            self.m_t[i] = self.beta_1 * self.m_t[i] + (1 - self.beta_1) * pseudo_grad
            
            # Yogi-specific second moment update
            self.v_t[i] = self.v_t[i] - (1 - self.beta_2) * np.square(pseudo_grad) * np.sign(
                self.v_t[i] - np.square(pseudo_grad)
            )
            
            # Ensure v_t stays positive
            self.v_t[i] = np.maximum(self.v_t[i], self.tau)
            
            # Parameter update with clipping
            update = self.eta * self.m_t[i] / np.sqrt(self.v_t[i])
            
            # Clip large updates to prevent divergence
            update_norm = np.linalg.norm(update)
            if update_norm > self.clip_norm:
                update = update * (self.clip_norm / update_norm)
            
            new_param = global_params[i] + update
            new_params.append(new_param)
        
        self.update_round()
        return new_params


class FedAdagradStrategy(AggregationStrategy):
    """
    Federated Adagrad Strategy - IMPROVED VERSION
    
    Applies Adagrad optimizer at the server level for parameter aggregation.
    Includes parameter clipping to prevent divergence.
    """
    
    def __init__(self, eta: float = 0.01, tau: float = 1e-3, clip_norm: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.eta = eta
        self.tau = tau
        self.clip_norm = clip_norm
        self.name = "FedAdagrad"
        
        # Adagrad state
        self.G_t = None  # Accumulated squared gradients
    
    def aggregate(self, 
                 client_params: List[List[np.ndarray]], 
                 client_weights: List[float], 
                 global_params: List[np.ndarray],
                 **kwargs) -> List[np.ndarray]:
        
        # Initialize Adagrad state if first round
        if self.G_t is None:
            self.G_t = [np.zeros_like(param) for param in global_params]
        
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
        
        # Step 2: Compute pseudo-gradient and apply Adagrad with clipping
        new_params = []
        for i in range(num_params):
            pseudo_grad = averaged_params[i] - global_params[i]
            
            # Accumulate squared gradients
            self.G_t[i] += np.square(pseudo_grad)
            
            # Adagrad update with clipping
            update = self.eta * pseudo_grad / (np.sqrt(self.G_t[i]) + self.tau)
            
            # Clip large updates to prevent divergence
            update_norm = np.linalg.norm(update)
            if update_norm > self.clip_norm:
                update = update * (self.clip_norm / update_norm)
            
            new_param = global_params[i] + update
            new_params.append(new_param)
        
        self.update_round()
        return new_params


class FedLAGStrategy(AggregationStrategy):
    """
    Federated Learning with Gradient Tracking (FedLAG) Strategy - IMPROVED VERSION
    
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
    'fedadam': FedAdamStrategy,
    'fedyogi': FedYogiStrategy,
    'fedadagrad': FedAdagradStrategy,
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
            "fedadam": {"eta": 0.001, "beta_1": 0.9, "beta_2": 0.999, "tau": 1e-4, "clip_norm": 0.5},
            "fedyogi": {"eta": 0.001, "beta_1": 0.9, "beta_2": 0.999, "tau": 1e-4, "clip_norm": 0.5},
            "fedadagrad": {"eta": 0.001, "tau": 1e-4, "clip_norm": 0.5},
            "fedlag": {"eta": 0.001, "momentum": 0.5, "clip_norm": 0.5}
        }
    else:
        # Standard parameters for simpler datasets like Iris
        strategy_params = {
            "fedprox": {"mu": 0.01},
            "fedadam": {"eta": 0.01, "beta_1": 0.9, "beta_2": 0.99, "tau": 1e-3, "clip_norm": 1.0},
            "fedyogi": {"eta": 0.01, "beta_1": 0.9, "beta_2": 0.99, "tau": 1e-3, "clip_norm": 1.0},
            "fedadagrad": {"eta": 0.01, "tau": 1e-3, "clip_norm": 1.0},
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
    
    print("Testing Fixed Aggregation Strategies")
    print("=" * 40)
    
    for strategy_name in list_strategies():
        print(f"\n🔄 Testing {strategy_name.upper()}")
        
        try:
            # Test with dataset-specific parameters
            strategy = get_strategy(strategy_name, dataset="adult")
            result = strategy.aggregate(client_params, client_weights, global_params)
            
            # Print some basic stats
            param_norms = [np.linalg.norm(param) for param in result]
            print(f"   ✅ Success - Parameter norms: {[f'{norm:.3f}' for norm in param_norms]}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    