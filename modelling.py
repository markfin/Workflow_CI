
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import os
import sys
from datetime import datetime

def load_and_prepare_data():
    """Load preprocessed diabetes dataset"""
    # Coba beberapa lokasi untuk dataset
    possible_paths = [
        'preprocessed_diabetes.csv',
        '../preprocessed_diabetes.csv',
        '../../preprocessing/preprocessed_diabetes.csv',
        '/content/drive/MyDrive/Colab Notebooks/Demo9/preprocessing/preprocessed_diabetes.csv'
    ]
    
    df = None
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Loading data from: {path}")
            df = pd.read_csv(path)
            break
    
    if df is None:
        # Jika tidak ditemukan, download dan preprocess data mentah
        print("Preprocessed data not found. Downloading raw data...")
        import requests
        url = 'https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv'
        response = requests.get(url)
        
        raw_path = 'diabetes_raw.csv'
        with open(raw_path, 'wb') as f:
            f.write(response.content)
        
        # Simple preprocessing
        df = pd.read_csv(raw_path)
        # Replace 0 with NaN for certain columns
        cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)
        # Impute with median
        for col in cols_with_zero:
            df[col] = df[col].fillna(df[col].median())
    
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    return df

def train_models(X_train, X_test, y_train, y_test):
    """Train multiple models and log with MLflow"""
    
    models = {
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for model_name, model in models.items():
        with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            print(f"\nTraining {model_name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            train_accuracy = accuracy_score(y_train, y_pred_train)
            test_accuracy = accuracy_score(y_test, y_pred_test)
            precision = precision_score(y_test, y_pred_test, average='binary')
            recall = recall_score(y_test, y_pred_test, average='binary')
            f1 = f1_score(y_test, y_pred_test, average='binary')
            
            # Log parameters
            if model_name == 'logistic_regression':
                mlflow.log_params({
                    'solver': model.solver,
                    'max_iter': model.max_iter,
                    'C': model.C
                })
            else:
                mlflow.log_params({
                    'n_estimators': model.n_estimators,
                    'max_depth': model.max_depth,
                    'min_samples_split': model.min_samples_split
                })
            
            # Log metrics
            mlflow.log_metrics({
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })
            
            # Log model
            mlflow.sklearn.log_model(model, model_name)
            
            # Log dataset info
            mlflow.log_params({
                'train_size': len(X_train),
                'test_size': len(X_test),
                'n_features': X_train.shape[1]
            })
            
            # Store results
            results[model_name] = {
                'test_accuracy': test_accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'run_id': mlflow.active_run().info.run_id
            }
            
            print(f"{model_name} - Test Accuracy: {test_accuracy:.4f}")
            print(f"Run ID: {mlflow.active_run().info.run_id}")
    
    return results

if __name__ == "__main__":
    print("=" * 50)
    print("MLflow Model Training Pipeline")
    print("=" * 50)
    
    # Set MLflow tracking URI to local directory
    mlflow.set_tracking_uri("file:./mlruns")
    print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
    
    # Load data
    df = load_and_prepare_data()
    
    # Prepare features and target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nData split:")
    print(f"Train set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Target distribution - Train: {y_train.value_counts().to_dict()}")
    print(f"Target distribution - Test: {y_test.value_counts().to_dict()}")
    
    # Train models
    results = train_models(X_train, X_test, y_train, y_test)
    
    # Print summary
    print("\n" + "=" * 50)
    print("Training Summary")
    print("=" * 50)
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Test Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")
        print(f"  Run ID: {metrics['run_id']}")
    
    print("\n" + "=" * 50)
    print("MLflow runs have been saved to ./mlruns")
    print("To view the MLflow UI, run: mlflow ui")
    print("=" * 50)
