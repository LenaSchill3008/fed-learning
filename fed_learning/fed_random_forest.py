# fed_learning/fed_random_forest.py
"""Simplified Federated Random Forest based on Hongwei-Z approach"""

import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def get_rf_hyperparams(model: RandomForestClassifier) -> List[float]:
    """Extract hyperparameters from a trained Random Forest model."""
    return [
        float(model.n_estimators),
        float(model.max_depth if model.max_depth is not None else 40),
        float(model.min_samples_split),
        float(model.min_samples_leaf)
    ]


def set_rf_hyperparams(hyperparams: List[float]) -> Dict[str, Any]:
    """Convert hyperparameters list to RandomForest parameters dict."""
    # Ensure hyperparameters are within valid ranges
    n_estimators = max(1, int(round(hyperparams[0])))
    max_depth = max(1, int(round(hyperparams[1]))) if hyperparams[1] != 40 else None
    min_samples_split = max(2, int(round(hyperparams[2])))
    min_samples_leaf = max(1, int(round(hyperparams[3])))
    
    return {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'bootstrap': True,
        'random_state': 42
    }


def create_rf_model(hyperparams: List[float] = None) -> RandomForestClassifier:
    """Create a Random Forest model with given or default hyperparameters."""
    if hyperparams is None:
        # Default hyperparameters
        return RandomForestClassifier(
            n_estimators=10,
            max_depth=5,
            min_samples_split=2,
            min_samples_leaf=1,
            bootstrap=True,
            random_state=42
        )
    else:
        params = set_rf_hyperparams(hyperparams)
        return RandomForestClassifier(**params)


def calculate_rf_metrics(model: RandomForestClassifier, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Calculate all metrics for random forest model."""
    if not hasattr(model, 'classes_'):
        return {"Accuracy": 0.0, "Precision": 0.0, "Recall": 0.0, "F1_Score": 0.0}
    
    y_pred = model.predict(X)
    
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='macro', zero_division=0)
    recall = recall_score(y, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y, y_pred, average='macro', zero_division=0)
    
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1
    }