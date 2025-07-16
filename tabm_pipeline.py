import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from mambular.models import MambularClassifier, MambularRegressor
from mambular.preprocessing import Preprocessor
import warnings
warnings.filterwarnings('ignore')

class MambularTrainingPipeline:
    """
    A comprehensive training pipeline for Mambular models supporting both 
    classification and regression tasks.
    """
    
    def __init__(self, task_type='classification', config=None):
        """
        Initialize the training pipeline.
        
        Args:
            task_type (str): 'classification' or 'regression'
            config (dict): Configuration parameters for the model
        """
        self.task_type = task_type
        self.config = config or self._get_default_config()
        self.model = None
        self.preprocessor = None
        self.scaler = None
        self.label_encoder = None
        self.training_history = []
        
    def _get_default_config(self):
        """Get default configuration for Mambular model."""
        return {
            'd_model': 64,
            'n_layers': 8,
            'expand_factor': 2,
            'bias': False,
            'conv_bias': True,
            'dropout': 0.1,
            'lr': 1e-3,
            'lr_patience': 10,
            'weight_decay': 1e-6,
            'batch_size': 512,
            'epochs': 100,
            'early_stopping_patience': 15,
            'validation_split': 0.2,
            'random_state': 42
        }
    
    def load_data(self, data_path=None, X=None, y=None):
        """
        Load data from file or arrays.
        
        Args:
            data_path (str): Path to CSV file
            X (array-like): Feature matrix
            y (array-like): Target vector
        """
        if data_path:
            df = pd.read_csv(data_path)
            self.X = df.drop(columns=['target']).values
            self.y = df['target'].values
        else:
            self.X = X
            self.y = y
            
        print(f"Data loaded: {self.X.shape[0]} samples, {self.X.shape[1]} features")
        return self
    
    def preprocess_data(self):
        """Preprocess the data for training."""
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, 
            test_size=self.config['validation_split'], 
            random_state=self.config['random_state']
        )
        
        # Initialize and fit preprocessor
        self.preprocessor = Preprocessor()
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        # Handle target encoding for classification
        if self.task_type == 'classification':
            self.label_encoder = LabelEncoder()
            y_train = self.label_encoder.fit_transform(y_train)
            y_test = self.label_encoder.transform(y_test)
            self.num_classes = len(self.label_encoder.classes_)
        
        # Store processed data
        self.X_train = X_train_processed
        self.X_test = X_test_processed
        self.y_train = y_train
        self.y_test = y_test
        
        print(f"Data preprocessed:")
        print(f"  Training set: {self.X_train.shape}")
        print(f"  Test set: {self.X_test.shape}")
        
        return self
    
    def build_model(self):
        """Build the Mambular model."""
        if self.task_type == 'classification':
            self.model = MambularClassifier(
                d_model=self.config['d_model'],
                n_layers=self.config['n_layers'],
                expand_factor=self.config['expand_factor'],
                bias=self.config['bias'],
                conv_bias=self.config['conv_bias'],
                dropout=self.config['dropout'],
                lr=self.config['lr'],
                lr_patience=self.config['lr_patience'],
                weight_decay=self.config['weight_decay'],
                batch_size=self.config['batch_size']
            )
        else:
            self.model = MambularRegressor(
                d_model=self.config['d_model'],
                n_layers=self.config['n_layers'],
                expand_factor=self.config['expand_factor'],
                bias=self.config['bias'],
                conv_bias=self.config['conv_bias'],
                dropout=self.config['dropout'],
                lr=self.config['lr'],
                lr_patience=self.config['lr_patience'],
                weight_decay=self.config['weight_decay'],
                batch_size=self.config['batch_size']
            )
        
        print(f"Mambular {self.task_type} model built successfully")
        return self
    
    def train(self):
        """Train the model."""
        print("Starting training...")
        
        # Fit the model
        self.model.fit(
            self.X_train, 
            self.y_train,
            epochs=self.config['epochs'],
            validation_split=0.2,
            early_stopping_patience=self.config['early_stopping_patience'],
            verbose=1
        )
        
        print("Training completed!")
        return self
    
    def evaluate(self):
        """Evaluate the model on test data."""
        print("Evaluating model...")
        
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        
        if self.task_type == 'classification':
            # Classification metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            report = classification_report(self.y_test, y_pred)
            
            print(f"Accuracy: {accuracy:.4f}")
            print("\nClassification Report:")
            print(report)
            
            self.metrics = {
                'accuracy': accuracy,
                'classification_report': report
            }
        else:
            # Regression metrics
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            
            print(f"MSE: {mse:.4f}")
            print(f"RMSE: {rmse:.4f}")
            
            self.metrics = {
                'mse': mse,
                'rmse': rmse
            }
        
        return self
    
    def save_model(self, path):
        """Save the trained model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'preprocessor': self.preprocessor,
            'label_encoder': self.label_encoder,
            'config': self.config,
            'task_type': self.task_type
        }, path)
        print(f"Model saved to {path}")
        return self
    
    def load_model(self, path):
        """Load a trained model."""
        checkpoint = torch.load(path)
        self.config = checkpoint['config']
        self.task_type = checkpoint['task_type']
        self.preprocessor = checkpoint['preprocessor']
        self.label_encoder = checkpoint['label_encoder']
        
        self.build_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {path}")
        return self
    
    def predict(self, X_new):
        """Make predictions on new data."""
        if self.preprocessor is None:
            raise ValueError("Model must be trained or loaded before making predictions")
        
        X_processed = self.preprocessor.transform(X_new)
        predictions = self.model.predict(X_processed)
        
        if self.task_type == 'classification' and self.label_encoder:
            predictions = self.label_encoder.inverse_transform(predictions)
        
        return predictions

# Example usage and demonstration
def demo_classification():
    """Demonstration with a classification task."""
    print("=== Classification Demo ===")
    
    # Generate sample data
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=1000, 
        n_features=20, 
        n_informative=15, 
        n_redundant=5, 
        n_classes=3,
        random_state=42
    )
    
    # Create and run pipeline
    pipeline = MambularTrainingPipeline(task_type='classification')
    pipeline.load_data(X=X, y=y)
    pipeline.preprocess_data()
    pipeline.build_model()
    pipeline.train()
    pipeline.evaluate()
    
    # Save model
    pipeline.save_model('mambular_classifier.pth')
    
    return pipeline

def demo_regression():
    """Demonstration with a regression task."""
    print("\n=== Regression Demo ===")
    
    # Generate sample data
    from sklearn.datasets import make_regression
    X, y = make_regression(
        n_samples=1000, 
        n_features=20, 
        n_informative=15, 
        noise=0.1,
        random_state=42
    )
    
    # Create and run pipeline
    pipeline = MambularTrainingPipeline(task_type='regression')
    pipeline.load_data(X=X, y=y)
    pipeline.preprocess_data()
    pipeline.build_model()
    pipeline.train()
    pipeline.evaluate()
    
    # Save model
    pipeline.save_model('mambular_regressor.pth')
    
    return pipeline

def advanced_config_example():
    """Example with advanced configuration."""
    print("\n=== Advanced Configuration Example ===")
    
    # Custom configuration
    custom_config = {
        'd_model': 128,
        'n_layers': 12,
        'expand_factor': 4,
        'bias': True,
        'conv_bias': True,
        'dropout': 0.2,
        'lr': 5e-4,
        'lr_patience': 5,
        'weight_decay': 1e-5,
        'batch_size': 256,
        'epochs': 200,
        'early_stopping_patience': 20,
        'validation_split': 0.25,
        'random_state': 42
    }
    
    # Generate sample data
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=2000, 
        n_features=50, 
        n_informative=30, 
        n_redundant=10, 
        n_classes=5,
        random_state=42
    )
    
    # Create pipeline with custom config
    pipeline = MambularTrainingPipeline(
        task_type='classification', 
        config=custom_config
    )
    
    pipeline.load_data(X=X, y=y)
    pipeline.preprocess_data()
    pipeline.build_model()
    pipeline.train()
    pipeline.evaluate()
    
    return pipeline

if __name__ == "__main__":
    # Run demonstrations
    print("Mambular Training Pipeline Demo")
    print("=" * 50)
    
    # Classification demo
    clf_pipeline = demo_classification()
    
    # Regression demo
    reg_pipeline = demo_regression()
    
    # Advanced configuration demo
    advanced_pipeline = advanced_config_example()
    
    print("\nDemo completed! Check the saved models.")