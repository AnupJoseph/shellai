# Internal
from typing import List
import warnings
import yaml


# External
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error

import optuna
import mlflow

from optuna.integration import MLflowCallback

# Project
from models import ModelConfig, model_configs


# Project Specific Configuration
warnings.filterwarnings("ignore")


class Experiment:
    def __init__(
        self,
        csv_file_path: str,
        target_columns: List[str],
        model_config: ModelConfig,
        test_size: float = 0.2,
        val_size: float = 0.2,
        random_state: int = 42,
    ):
        """
        Initialize the configurable experiment setup

        Parameters:
        - csv_file_path: Path to the CSV file
        - target_columns: List of target column names for multi-output regression
        - model_config: ModelConfig object containing model class and parameter space
        - test_size: Proportion of data for testing
        - val_size: Proportion of remaining data for validation
        - random_state: Random seed for reproducibility
        """
        self.csv_file_path = csv_file_path
        self.target_columns = target_columns
        self.model_config = model_config
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_params = None
        self.experiment_name = f"{model_config.model_name}_Experiment"

    def load_and_preprocess_data(self):
        """Load CSV data and perform preprocessing"""
        print(f"Loading and preprocessing data from {self.csv_file_path}...")

        # Load data
        self.data = pd.read_csv(self.csv_file_path)
        print(f"Data shape: {self.data.shape}")
        print(f"Columns: {list(self.data.columns)}")

        # Validate target columns exist
        missing_targets = [
            col for col in self.target_columns if col not in self.data.columns
        ]
        if missing_targets:
            raise ValueError(f"Target columns not found in data: {missing_targets}")

        # Check for missing values
        missing_values = self.data.isnull().sum()
        if missing_values.any():
            print(f"Missing values found:\n{missing_values[missing_values > 0]}")
            # Handle missing values (simple strategy - you can modify this)
            self.data = self.data.fillna(self.data.mean(numeric_only=True))

        # Separate features and targets
        self.X = self.data.drop(columns=self.target_columns)
        self.y = self.data[self.target_columns]

        # Select only numeric columns for features
        numeric_columns = self.X.select_dtypes(include=[np.number]).columns
        self.X = self.X[numeric_columns]

        print(f"Features shape: {self.X.shape}")
        print(f"Targets shape: {self.y.shape}")
        print(f"Target columns: {self.target_columns}")

        # Create train/validation/test splits
        self.create_splits()

    def create_splits(self):
        """Create train, validation, and test splits"""
        print("Creating train/validation/test splits...")

        # First split: train+val vs test
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )

        # Second split: train vs val
        val_size_adjusted = self.val_size / (1 - self.test_size)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=self.random_state
        )

        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        print(f"Train set: {self.X_train.shape[0]} samples")
        print(f"Validation set: {self.X_val.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")

    def calculate_mape(self, y_true, y_pred):
        """Calculate Mean Absolute Percentage Error"""
        mape_scores = []
        for i in range(y_true.shape[1]):
            mape = mean_absolute_percentage_error(y_true.iloc[:, i], y_pred[:, i])
            mape_scores.append(mape)
        return np.mean(mape_scores)

    def create_objective(self, trial):
        """Objective function for Optuna optimization"""
        # Sample parameters from the parameter space
        params = {}
        for param_name, param_sampler in self.model_config.param_space.items():
            params[param_name] = param_sampler(trial)

        # Add fixed parameters
        params.update(self.model_config.fixed_params)

        # Create and train model
        base_model = self.model_config.model_class(**params)
        multi_output_model = MultiOutputRegressor(base_model)
        multi_output_model.fit(self.X_train_scaled, self.y_train)

        # Make predictions on validation set
        y_pred = multi_output_model.predict(self.X_val_scaled)

        # Calculate MAPE
        mape = self.calculate_mape(self.y_val, y_pred)

        return mape

    def optimize_hyperparameters(self, n_trials: int = 100):
        """Optimize hyperparameters using Optuna"""
        print(
            f"Starting hyperparameter optimization for {self.model_config.model_name} with {n_trials} trials..."
        )

        # Set up MLflow experiment
        mlflow.set_experiment(self.experiment_name)

        # Create Optuna study
        study = optuna.create_study(direction="minimize")

        # Add MLflow callback
        mlflowc = MLflowCallback(
            tracking_uri=mlflow.get_tracking_uri(), metric_name="mape"
        )

        # Optimize
        study.optimize(self.create_objective, n_trials=n_trials, callbacks=[mlflowc])

        self.best_params = study.best_params
        self.best_params.update(self.model_config.fixed_params)
        self.best_mape = study.best_value

        print(f"Best MAPE: {self.best_mape:.4f}")
        print(f"Best parameters: {self.best_params}")

        return study

    def train_best_model(self):
        """Train the best model with optimized parameters on the training data"""
        print("Training best model...")

        # Create best model
        base_model = self.model_config.model_class(**self.best_params)
        self.best_model = MultiOutputRegressor(base_model)

        # Train on full training data
        self.best_model.fit(self.X_train_scaled, self.y_train)

        # Make predictions
        self.train_pred = self.best_model.predict(self.X_train_scaled)
        self.val_pred = self.best_model.predict(self.X_val_scaled)
        self.test_pred = self.best_model.predict(self.X_test_scaled)

        # Calculate MAPE for all sets
        self.train_mape = self.calculate_mape(self.y_train, self.train_pred)
        self.val_mape = self.calculate_mape(self.y_val, self.val_pred)
        self.test_mape = self.calculate_mape(self.y_test, self.test_pred)

        # Calculate individual MAPE scores for each target
        self.individual_mape = {}
        for i, target in enumerate(self.target_columns):
            self.individual_mape[target] = mean_absolute_percentage_error(
                self.y_test.iloc[:, i], self.test_pred[:, i]
            )

        print(f"Train MAPE: {self.train_mape:.4f}")
        print(f"Validation MAPE: {self.val_mape:.4f}")
        print(f"Test MAPE: {self.test_mape:.4f}")
        print("Individual target MAPE:")
        for target, mape in self.individual_mape.items():
            print(f"  {target}: {mape:.4f}")

    def train_final_model(self):
        """Train the final model on the entire dataset using the best parameters."""
        print("Training final model on the entire dataset...")

        # Create the best model
        base_model = self.model_config.model_class(**self.best_params)
        self.final_model = MultiOutputRegressor(base_model)

        # Scale the entire dataset
        X_scaled = self.scaler.fit_transform(self.X)

        # Train on the entire scaled dataset
        self.final_model.fit(X_scaled, self.y)
        print("Final model training complete.")

    def log_to_mlflow(self):
        """Log the best model and metrics to MLflow"""
        print("Logging to MLflow...")

        with mlflow.start_run(
            run_name=f"Best_{self.model_config.model_name}_MultiOutput"
        ):
            # Log the final model if trained on the entire dataset
            if hasattr(self, "final_model"):
                mlflow.sklearn.log_model(self.final_model, "final_multi_output_model")
                print("Final model logged to MLflow.")
            else:
                print("Final model not trained, skipping MLflow logging.")
            # Log parameters
            mlflow.log_params(self.best_params)
            mlflow.log_param("model_type", self.model_config.model_name)
            mlflow.log_param("n_features", self.X.shape[1])
            mlflow.log_param("n_targets", len(self.target_columns))
            mlflow.log_param("train_size", self.X_train.shape[0])
            mlflow.log_param("val_size", self.X_val.shape[0])
            mlflow.log_param("test_size", self.X_test.shape[0])

            # Log metrics
            mlflow.log_metric("train_mape", self.train_mape)
            mlflow.log_metric("val_mape", self.val_mape)
            mlflow.log_metric("test_mape", self.test_mape)

            # Log individual target MAPE
            for target, mape in self.individual_mape.items():
                mlflow.log_metric(f"test_mape_{target}", mape)

            # Log model
            if self.model_config.mlflow_log_model_func:
                self.model_config.mlflow_log_model_func(
                    self.best_model, "multi_output_model"
                )
            else:
                mlflow.sklearn.log_model(self.best_model, "multi_output_model")

            # Log visualizations
            self.plot_feature_importance()
            mlflow.log_artifact("feature_importance.png")

            self.plot_results()
            mlflow.log_artifact("results_comparison.png")

            self.plot_mape_comparison()
            mlflow.log_artifact("mape_comparison.png")

            print("MLflow logging completed!")

    def plot_feature_importance(self):
        """Plot feature importance for each output"""
        n_outputs = len(self.target_columns)
        fig, axes = plt.subplots(1, n_outputs, figsize=(6 * n_outputs, 6))

        if n_outputs == 1:
            axes = [axes]

        for i, (ax, target) in enumerate(zip(axes, self.target_columns)):
            # Get feature importance (handling different model types)
            try:
                if hasattr(self.best_model.estimators_[i], "feature_importances_"):
                    importance = self.best_model.estimators_[i].feature_importances_
                elif hasattr(self.best_model.estimators_[i], "coef_"):
                    importance = np.abs(self.best_model.estimators_[i].coef_)
                else:
                    print(
                        f"Feature importance not available for {self.model_config.model_name}"
                    )
                    continue

                feature_names = self.X.columns

                # Create importance dataframe
                importance_df = pd.DataFrame(
                    {"feature": feature_names, "importance": importance}
                ).sort_values("importance", ascending=True)

                # Plot top 20 features
                top_features = importance_df.tail(20)
                ax.barh(top_features["feature"], top_features["importance"])
                ax.set_title(f"Feature Importance - {target}")
                ax.set_xlabel("Importance")

            except Exception as e:
                print(f"Error plotting feature importance: {e}")
                ax.text(
                    0.5,
                    0.5,
                    f"Feature importance\nnot available for\n{self.model_config.model_name}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(f"Feature Importance - {target}")

        plt.tight_layout()
        plt.savefig("feature_importance.png", dpi=300, bbox_inches="tight")
        # plt.show()

    def plot_results(self):
        """Plot training results and predictions"""
        n_outputs = len(self.target_columns)
        fig, axes = plt.subplots(2, n_outputs, figsize=(6 * n_outputs, 12))

        if n_outputs == 1:
            axes = axes.reshape(-1, 1)

        for i, target in enumerate(self.target_columns):
            # Plot 1: Actual vs Predicted
            ax1 = axes[0, i]
            ax1.scatter(self.y_test.iloc[:, i], self.test_pred[:, i], alpha=0.6)
            ax1.plot(
                [self.y_test.iloc[:, i].min(), self.y_test.iloc[:, i].max()],
                [self.y_test.iloc[:, i].min(), self.y_test.iloc[:, i].max()],
                "r--",
                lw=2,
            )
            ax1.set_xlabel(f"Actual {target}")
            ax1.set_ylabel(f"Predicted {target}")
            ax1.set_title(
                f"Actual vs Predicted - {target}\nMAPE: {self.individual_mape[target]:.4f}"
            )

            # Add R² score
            from sklearn.metrics import r2_score

            r2 = r2_score(self.y_test.iloc[:, i], self.test_pred[:, i])
            ax1.text(
                0.05,
                0.95,
                f"R²: {r2:.3f}",
                transform=ax1.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

            # Plot 2: Residuals
            ax2 = axes[1, i]
            residuals = self.y_test.iloc[:, i] - self.test_pred[:, i]
            ax2.scatter(self.test_pred[:, i], residuals, alpha=0.6)
            ax2.axhline(y=0, color="r", linestyle="--")
            ax2.set_xlabel(f"Predicted {target}")
            ax2.set_ylabel(f"Residuals {target}")
            ax2.set_title(f"Residuals Plot - {target}")

        plt.tight_layout()
        plt.savefig("results_comparison.png", dpi=300, bbox_inches="tight")
        # plt.show()

    def plot_mape_comparison(self):
        """Plot MAPE comparison across train/val/test sets and individual targets"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Overall MAPE comparison
        mape_data = {
            "Set": ["Train", "Validation", "Test"],
            "MAPE": [self.train_mape, self.val_mape, self.test_mape],
        }

        bars1 = ax1.bar(
            mape_data["Set"],
            mape_data["MAPE"],
            color=["skyblue", "lightcoral", "lightgreen"],
        )
        ax1.set_ylabel("Mean Absolute Percentage Error")
        ax1.set_title("Overall MAPE Comparison")
        ax1.set_ylim(0, max(mape_data["MAPE"]) * 1.1)

        # Add value labels on bars
        for bar, value in zip(bars1, mape_data["MAPE"]):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.4f}",
                ha="center",
                va="bottom",
            )

        # Individual target MAPE
        targets = list(self.individual_mape.keys())
        mape_values = list(self.individual_mape.values())

        bars2 = ax2.bar(targets, mape_values, color="lightblue")
        ax2.set_ylabel("Mean Absolute Percentage Error")
        ax2.set_title("MAPE by Target Variable")
        ax2.set_ylim(0, max(mape_values) * 1.1)

        # Rotate x-axis labels if needed
        if len(max(targets, key=len)) > 8:
            ax2.tick_params(axis="x", rotation=45)

        # Add value labels on bars
        for bar, value in zip(bars2, mape_values):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.4f}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plt.savefig("mape_comparison.png", dpi=300, bbox_inches="tight")
        # plt.show()

    def run_experiment(self, n_trials: int = 100):
        """Run the complete experiment pipeline"""
        print(
            f"=== Starting {self.model_config.model_name} Multi-Output Regression Experiment ==="
        )

        # Load and preprocess data
        self.load_and_preprocess_data()

        # Optimize hyperparameters
        study = self.optimize_hyperparameters(n_trials=n_trials)

        # Train best model
        self.train_best_model()

        # Train final model on entire dataset
        # self.train_final_model()

        # Log to MLflow
        self.log_to_mlflow()

        print(f"\n=== Experiment Complete ===")
        print(f"Best Test MAPE: {self.test_mape:.4f}")

        return {
            "best_params": self.best_params,
            "train_mape": self.train_mape,
            "val_mape": self.val_mape,
            "test_mape": self.test_mape,
            "individual_mape": self.individual_mape,
            "model": self.best_model,
            "study": study,
        }

    def predict_and_save_submission(self):
        """
        Predict on test.csv using the best model and save the submission file.
        """
        print("Predicting on test.csv and saving submission...")

        # Load test data
        try:
            test_data = pd.read_csv("./data/test.csv")
        except FileNotFoundError:
            print(
                "Error: test.csv not found. Please ensure it is in the ./data/ directory."
            )
            return

        # Preprocess test data (handle missing values and select numeric columns)
        numeric_columns = test_data.select_dtypes(include=[np.number]).columns
        # numeric_columns.delete(0)
        X_test_submission = test_data[numeric_columns]
        X_test_submission.drop(columns=["ID"], inplace=True)
        X_test_submission = X_test_submission.fillna(
            X_test_submission.mean()
        )  # Impute missing values if any

        # Scale features
        X_test_submission_scaled = self.scaler.transform(X_test_submission)

        # Predict using the best model
        predictions = self.best_model.predict(X_test_submission_scaled)

        # Create submission dataframe
        submission_df = pd.DataFrame(
            predictions, columns=self.target_columns
        )  # Use target columns
        submission_df.insert(0, "ID", test_data["ID"])  # Add ID column

        # Save submission file
        submission_df.to_csv("submission.csv", index=False)
        print(
            f"Submission file 'submission.csv' created with predictions for {len(self.target_columns)} targets."
        )


# Example usage
if __name__ == "__main__":
    # Initialize MLflow
    mlflow.set_tracking_uri("mlruns")  # Local MLflow tracking

    # Example configuration - modify these according to your data
    CSV_FILE_PATH = "./data/train.csv"  # Replace with your CSV file path
    TARGET_COLUMNS = [
        f"BlendProperty{i+1}" for i in range(10)
    ]  # Replace with your target column names

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    data_config = config["data_config"]
    model_config_params = config["model_config"]
    mlflow_config_params = config["mlflow_config"]

    # Initialize MLflow tracking URI
    mlflow.set_tracking_uri(mlflow_config_params["tracking_uri"])

    # Select the model configuration
    model_name = model_config_params["model_name"]
    model_config = model_configs[model_name]

    # Create and run experiment
    experiment = Experiment(
        csv_file_path=data_config["csv_file_path"],
        target_columns=TARGET_COLUMNS,
        model_config=model_config,
        test_size=0.2,
        val_size=0.2,
        random_state=42,
    )

    # Run the complete experiment
    results = experiment.run_experiment(
        n_trials=model_config_params["n_trials"]
    )  # Adjust n_trials as needed

    print(f"\nFinal Results for {model_config.model_name}:")
    print(f"Best Parameters: {results['best_params']}")
    print(f"Test MAPE: {results['test_mape']:.4f}")
    print(f"Individual Target MAPE: {results['individual_mape']}")

    # Generate and save submission file
    experiment.predict_and_save_submission()
