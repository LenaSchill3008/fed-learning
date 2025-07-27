import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split 
from sklearn.utils import shuffle
from typing import Tuple, Union

# Globals and configuration
DATASET_NAME = "iris"          
_dataset_cache = {}              
MODEL_TYPE = "logistic"  

CSV_PATHS = {
    "iris":  "data/iris.csv",
    "adult": "data/adult.csv",
}

LABEL_COLUMNS = {
    "iris":  "Species",
    "adult": "income",
}

# helper so server/client can switch dataset
def set_dataset(name: str) -> None:
    global DATASET_NAME
    DATASET_NAME = name

def set_model_type(model_type: str) -> None:
    global MODEL_TYPE
    MODEL_TYPE = model_type

# load full dataset 
def _ensure_dataset_loaded() -> None:

    if DATASET_NAME in _dataset_cache:
        return                                 

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


def get_model(model_type: str = None, penalty: str = "l2", local_epochs: int = 100, **kwargs) -> Union[LogisticRegression, SVC]:
    """Get model based on type (logistic or svm)."""
    if model_type is None:
        model_type = MODEL_TYPE
    
    if model_type == "logistic":

        return LogisticRegression(
            penalty=penalty,
            max_iter=max(local_epochs, 200),  # Ensure enough iterations
            warm_start=True,
            multi_class="auto",
            solver="lbfgs",
            random_state=42
        )
    
    elif model_type == "svm":
        # For SVM, we use probability=True to get probability estimates

        return SVC(
            kernel=kwargs.get("kernel", "rbf"),
            C=kwargs.get("C", 1.0),
            gamma=kwargs.get("gamma", "scale"),
            probability=True,  # Enable probability estimates
            random_state=42,
            max_iter=1000,  # Add max_iter for SVM
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def get_model_params(model) -> list:

    if isinstance(model, LogisticRegression):
        return [model.coef_, model.intercept_] if model.fit_intercept else [model.coef_]
    
    elif isinstance(model, SVC):
        # For SVM, we extract support vectors, dual coefficients, and intercept

        if hasattr(model, 'support_vectors_'):

            return [
                model.support_vectors_,
                model.dual_coef_,
                model.intercept_,
                model.support_.astype(np.float32),  # indices of support vectors
                model.n_support_.astype(np.float32)  # number of support vectors per class
            ]
        
        else:
            # Model not trained yet, return dummy parameters
            _ensure_dataset_loaded()
            X_all, y_all = _dataset_cache[DATASET_NAME]
            n_features = X_all.shape[1]
            n_classes = len(np.unique(y_all))

            return [
                np.zeros((1, n_features), dtype=np.float32),  # dummy support vectors
                np.zeros((n_classes-1, 1), dtype=np.float32),  # dummy dual coef
                np.zeros(n_classes, dtype=np.float32),  # dummy intercept
                np.zeros(1, dtype=np.float32),  # dummy support indices
                np.ones(n_classes, dtype=np.float32)  # dummy n_support
            ]
    else:
        raise ValueError(f"Unknown model type: {type(model)}")


def set_model_params(model, params):

    
    if isinstance(model, LogisticRegression):

        if len(params) >= 1:
            model.coef_ = params[0].copy()  # Use copy to ensure independence
        if model.fit_intercept and len(params) > 1:
            model.intercept_ = params[1].copy()
        
        # Ensure model is in fitted state
        if not hasattr(model, 'classes_'):
            _ensure_dataset_loaded()
            X_all, y_all = _dataset_cache[DATASET_NAME]
            model.classes_ = np.unique(y_all)
            
    elif isinstance(model, SVC):
        # For SVM in federated learning, we have limitations
        # SVM doesn't support direct parameter setting like linear models
        # In practice, you'd need more sophisticated approaches for SVM FL
        # For now, we'll skip parameter setting for SVM
        pass

    return model


def set_initial_params(model):

    _ensure_dataset_loaded()      
    X_all, y_all = _dataset_cache[DATASET_NAME]

    n_classes  = len(np.unique(y_all))
    n_features = X_all.shape[1]

    if isinstance(model, LogisticRegression):
        model.classes_ = np.arange(n_classes)
        model.coef_    = np.zeros((n_classes, n_features), dtype=np.float32)
        if model.fit_intercept:
            model.intercept_ = np.zeros((n_classes,), dtype=np.float32)

    elif isinstance(model, SVC):
        # For SVM, we initialize dummy parameters
        model.classes_ = np.arange(n_classes)
        # SVM initialization is more complex and model-dependent
        # We'll let the model initialize itself during first training
        pass


def get_logistic_model(penalty: str, local_epochs: int) -> LogisticRegression:

    return get_model("logistic", penalty=penalty, local_epochs=local_epochs)