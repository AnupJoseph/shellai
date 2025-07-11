# Internal
from dataclasses import dataclass
from typing import Dict, Any, Callable, Optional

# External
import mlflow.xgboost
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge


# Project Specific Configuration


@dataclass
class ModelConfig:
    """Configuration class for model and its parameters"""

    model_class: Any
    param_space: Dict[str, Callable]
    fixed_params: Dict[str, Any]
    model_name: str
    mlflow_log_model_func: Optional[Callable] = None


# Model configuration factory functions
def create_xgboost_config():
    """Create XGBoost model configuration"""
    return ModelConfig(
        model_class=xgb.XGBRegressor,
        param_space={
            "n_estimators": lambda trial: trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": lambda trial: trial.suggest_int("max_depth", 3, 10),
            "learning_rate": lambda trial: trial.suggest_float(
                "learning_rate", 0.01, 0.3
            ),
            "subsample": lambda trial: trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": lambda trial: trial.suggest_float(
                "colsample_bytree", 0.6, 1.0
            ),
            "reg_alpha": lambda trial: trial.suggest_float("reg_alpha", 0, 10),
            "reg_lambda": lambda trial: trial.suggest_float("reg_lambda", 0, 10),
        },
        fixed_params={"random_state": 42, "n_jobs": -1},
        model_name="XGBoost",
        # mlflow_log_model_func=mlflow.xgboost.log_model,
    )


def create_random_forest_config():
    """Create Random Forest model configuration"""
    return ModelConfig(
        model_class=RandomForestRegressor,
        param_space={
            "n_estimators": lambda trial: trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": lambda trial: trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": lambda trial: trial.suggest_int(
                "min_samples_split", 2, 10
            ),
            "min_samples_leaf": lambda trial: trial.suggest_int(
                "min_samples_leaf", 1, 5
            ),
            "max_features": lambda trial: trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None]
            ),
        },
        fixed_params={"random_state": 42, "n_jobs": -1},
        model_name="RandomForest",
    )


def create_ridge_config():
    """Create Ridge regression model configuration"""
    return ModelConfig(
        model_class=Ridge,
        param_space={
            "alpha": lambda trial: trial.suggest_float("alpha", 0.1, 100.0, log=True),
            "solver": lambda trial: trial.suggest_categorical(
                "solver",
                ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
            ),
        },
        fixed_params={"random_state": 42},
        model_name="Ridge",
    )


model_configs = {
    "xgboost": create_xgboost_config(),
    "random_forest": create_random_forest_config(),
    "ridge": create_ridge_config(),
}
