import numpy as np
from typing import List
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class AggregationStrategy(ABC):
    
    """
    Abstract base class for federated learning aggregation strategies
    """
    
    def __init__(self, **kwargs):
        self.round_num = 0
        self.history = []
    
    @abstractmethod
    def aggregate(self, 
                 client_params: List[List[np.ndarray]], 
                 client_weights: List[float], 
                 global_params: List[np.ndarray],
                 **kwargs) -> List[np.ndarray]:
        pass
    
    def update_round(self):
        """Update round number and history"""
        self.round_num += 1
    
    def reset(self):
        """Reset strategy state"""
        self.round_num = 0
        self.history = []


class FedAvgStrategy(AggregationStrategy):
    
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
            
            # Compute coordinate-wise median along the client axis
            median_param = np.median(param_stack, axis=0)
            aggregated_params.append(median_param)
        
        self.update_round()
        return aggregated_params


# Strategy registry for easy access
AGGREGATION_STRATEGIES = {
    'fedavg': FedAvgStrategy,
    'fedprox': FedProxStrategy,
    'fedmedian': FedMedianStrategy
}


def get_strategy_params_for_dataset(strategy_name: str, dataset: str) -> dict:
    
    """
    Get dataset-specific parameters for strategies to improve stability.
    """
    
    if dataset.lower() in ["adult"]:
        # More conservative parameters for complex datasets
        strategy_params = {
            "fedprox": {"mu": 0.001},  # Much smaller regularization
            "fedmedian": {},
        }
    else:
        # Standard parameters for simpler datasets like Iris
        strategy_params = {
            "fedprox": {"mu": 0.01},
            "fedmedian": {},  # No specific parameters for FedMedian
        }
    
    return strategy_params.get(strategy_name.lower(), {})


def get_strategy(strategy_name, dataset = None, **kwargs):

    """
    get aggregation strategy by name with dataset-specific parameters.
    """

    strategy_name = strategy_name.lower()
    if strategy_name not in AGGREGATION_STRATEGIES:
        available = ', '.join(AGGREGATION_STRATEGIES.keys())
        raise ValueError(f"Unknown strategy '{strategy_name}'. Available: {available}")
    
    # Get dataset-specific parameters
    if dataset:
        default_params = get_strategy_params_for_dataset(strategy_name, dataset)
        # Override defaults with provided kwargs
        default_params.update(kwargs)
        kwargs = default_params
    
    return AGGREGATION_STRATEGIES[strategy_name](**kwargs)


def list_strategies():

    return list(AGGREGATION_STRATEGIES.keys())