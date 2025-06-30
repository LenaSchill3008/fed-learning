# task.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split 
from sklearn.utils import shuffle
from typing import Tuple

# Globals and configuration
DATASET_NAME = "iris"          
_dataset_cache = {}              

CSV_PATHS = {
    "iris":  "data/iris.csv",
    "adult": "data/adult.csv",
}
LABEL_COLUMNS = {
    "iris":  "Species",
    "adult": "income",
}


# Public helper so server/client can switch dataset
def set_dataset(name: str) -> None:
    """Select which dataset to use globally (`iris` or `adult`)."""
    global DATASET_NAME
    DATASET_NAME = name


# load full dataset into the cache (runs only once per dataset)
def _ensure_dataset_loaded() -> None:
    """Populate `_dataset_cache[DATASET_NAME]` if it isn't cached yet."""
    if DATASET_NAME in _dataset_cache:
        return                                  # already loaded

    df = pd.read_csv(CSV_PATHS[DATASET_NAME])
    label_col = LABEL_COLUMNS[DATASET_NAME]

    # Encode labels 
    y = LabelEncoder().fit_transform(df[label_col])

    # One-hot any categorical predictors, scale numerics
    X = pd.get_dummies(df.drop(columns=[label_col]))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Shuffle once to mix classes before partitioning
    X_final, y_final = shuffle(X_scaled.astype(np.float32), y, random_state=42)
    _dataset_cache[DATASET_NAME] = (X_final, y_final)


def load_data(partition_id: int, num_partitions: int) -> Tuple[np.ndarray, ...]:
    """Return stratified train/test split for the given client partition (IID)."""
    _ensure_dataset_loaded()
    X_all, y_all = _dataset_cache[DATASET_NAME]

    # IID partitioning 
    part_size = len(X_all) // num_partitions
    start     = partition_id * part_size
    end       = len(X_all) if partition_id == num_partitions - 1 else (partition_id + 1) * part_size
    X_part, y_part = X_all[start:end], y_all[start:end]

    # Stratified 80/20 split so every client keeps the same class ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X_part,
        y_part,
        test_size=0.2,
        stratify=y_part,
        random_state=42,
    )

    return X_train, X_test, y_train, y_test


def get_model(penalty: str, local_epochs: int) -> LogisticRegression:
    return LogisticRegression(
        penalty=penalty,
        max_iter=local_epochs,
        warm_start=True,
        multi_class="auto",
        solver="lbfgs",
    )

def get_model_params(model):
    return [model.coef_, model.intercept_] if model.fit_intercept else [model.coef_]

def set_model_params(model, params):
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model

def set_initial_params(model):
    """Initialise coef_ / intercept_ so the server can send round-0 weights."""
    _ensure_dataset_loaded()                     # <-- NEW: guarantees cache is ready
    X_all, y_all = _dataset_cache[DATASET_NAME]

    n_classes  = len(np.unique(y_all))
    n_features = X_all.shape[1]

    model.classes_ = np.arange(n_classes)
    model.coef_    = np.zeros((n_classes, n_features), dtype=np.float32)
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,), dtype=np.float32)
