"""
Command Line Interface for LIFT Multi-Label Classifier

This module provides command-line tools for training, prediction, and evaluation
using the LIFT algorithm.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import train_test_split

from .classifier import LIFTClassifier


def load_data(data_path, target_cols=None):
    """
    Load data from CSV file.
    
    Parameters
    ----------
    data_path : str
        Path to the CSV file containing data.
    target_cols : list, optional
        List of column names for target variables. If None, assumes the last
        columns are targets based on binary values.
    
    Returns
    -------
    X : ndarray
        Feature matrix.
    y : ndarray
        Target matrix.
    feature_names : list
        Names of feature columns.
    target_names : list
        Names of target columns.
    """
    df = pd.read_csv(data_path)
    
    if target_cols is None:
        # Auto-detect binary columns as targets
        binary_cols = []
        for col in df.columns:
            if set(df[col].unique()).issubset({0, 1, np.nan}):
                binary_cols.append(col)
        
        if not binary_cols:
            raise ValueError("No binary columns found. Please specify target_cols.")
        
        target_cols = binary_cols[-5:]  # Take last 5 binary columns as default
        print(f"Auto-detected target columns: {target_cols}")
    
    feature_cols = [col for col in df.columns if col not in target_cols]
    
    X = df[feature_cols].values
    y = df[target_cols].values
    
    return X, y, feature_cols, target_cols


def train_main():
    """Main function for training CLI."""
    parser = argparse.ArgumentParser(description='Train LIFT classifier')
    parser.add_argument('data_path', help='Path to training data CSV file')
    parser.add_argument('--model-path', default='lift_model.pkl',
                       help='Path to save trained model')
    parser.add_argument('--target-cols', nargs='+',
                       help='Names of target columns')
    parser.add_argument('--k', type=int, default=3,
                       help='Number of clusters per label')
    parser.add_argument('--auto-tune', action='store_true',
                       help='Enable Bayesian hyperparameter optimization')
    parser.add_argument('--n-iter', type=int, default=10,
                       help='Number of optimization iterations')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Fraction of data to use for testing')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data_path}...")
    X, y, feature_names, target_names = load_data(args.data_path, args.target_cols)
    print(f"Data shape: {X.shape}, Labels shape: {y.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )
    
    # Train model
    print("Training LIFT classifier...")
    clf = LIFTClassifier(
        k=args.k,
        auto_tune=args.auto_tune,
        n_iter=args.n_iter,
        random_state=args.random_state
    )
    
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Test F1 (macro): {f1_macro:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    
    if args.auto_tune:
        print(f"Best parameters: {clf.best_params_}")
        print(f"Best CV score: {clf.best_score_:.4f}")
    
    # Save model
    model_data = {
        'classifier': clf,
        'feature_names': feature_names,
        'target_names': target_names,
        'metadata': {
            'k': args.k,
            'auto_tune': args.auto_tune,
            'n_iter': args.n_iter,
            'test_f1_macro': f1_macro,
            'test_accuracy': accuracy
        }
    }
    
    with open(args.model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved to {args.model_path}")


def predict_main():
    """Main function for prediction CLI."""
    parser = argparse.ArgumentParser(description='Make predictions with LIFT classifier')
    parser.add_argument('model_path', help='Path to trained model file')
    parser.add_argument('data_path', help='Path to data CSV file for prediction')
    parser.add_argument('--output-path', default='predictions.csv',
                       help='Path to save predictions')
    parser.add_argument('--probabilities', action='store_true',
                       help='Output probabilities instead of binary predictions')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    with open(args.model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    clf = model_data['classifier']
    feature_names = model_data['feature_names']
    target_names = model_data['target_names']
    
    # Load data
    print(f"Loading data from {args.data_path}...")
    df = pd.read_csv(args.data_path)
    
    # Extract features
    try:
        X = df[feature_names].values
    except KeyError as e:
        print(f"Error: Missing feature columns in data: {e}")
        sys.exit(1)
    
    # Make predictions
    print("Making predictions...")
    if args.probabilities:
        predictions = clf.predict_proba(X)
        pred_df = pd.DataFrame(predictions, columns=[f"{name}_prob" for name in target_names])
    else:
        predictions = clf.predict(X)
        pred_df = pd.DataFrame(predictions, columns=target_names)
    
    # Add original data
    result_df = pd.concat([df, pred_df], axis=1)
    
    # Save predictions
    result_df.to_csv(args.output_path, index=False)
    print(f"Predictions saved to {args.output_path}")


def evaluate_main():
    """Main function for evaluation CLI."""
    parser = argparse.ArgumentParser(description='Evaluate LIFT classifier')
    parser.add_argument('model_path', help='Path to trained model file')
    parser.add_argument('data_path', help='Path to test data CSV file')
    parser.add_argument('--target-cols', nargs='+',
                       help='Names of target columns')
    parser.add_argument('--output-path', default='evaluation_report.json',
                       help='Path to save evaluation report')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    with open(args.model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    clf = model_data['classifier']
    feature_names = model_data['feature_names']
    target_names = model_data['target_names']
    
    # Load data
    print(f"Loading data from {args.data_path}...")
    X, y, _, _ = load_data(args.data_path, args.target_cols or target_names)
    
    # Make predictions
    print("Evaluating model...")
    y_pred = clf.predict(X)
    y_proba = clf.predict_proba(X)
    
    # Calculate metrics
    f1_macro = f1_score(y, y_pred, average='macro')
    f1_micro = f1_score(y, y_pred, average='micro')
    f1_weighted = f1_score(y, y_pred, average='weighted')
    accuracy = accuracy_score(y, y_pred)
    
    # Per-label metrics
    f1_per_label = f1_score(y, y_pred, average=None)
    f1_per_label = np.atleast_1d(f1_per_label)  # Ensure it's always an array
    
    # Create evaluation report
    report = {
        'overall_metrics': {
            'f1_macro': float(f1_macro),
            'f1_micro': float(f1_micro),
            'f1_weighted': float(f1_weighted),
            'accuracy': float(accuracy)
        },
        'per_label_f1': {
            target_names[i]: float(f1_per_label[i])
            for i in range(len(target_names))
        },
        'model_metadata': model_data.get('metadata', {})
    }
    
    # Print summary
    print(f"F1 Score (macro): {f1_macro:.4f}")
    print(f"F1 Score (micro): {f1_micro:.4f}")
    print(f"F1 Score (weighted): {f1_weighted:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    
    print("\\nPer-label F1 scores:")
    for name, score in report['per_label_f1'].items():
        print(f"  {name}: {score:.4f}")
    
    # Save report
    with open(args.output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\\nEvaluation report saved to {args.output_path}")


if __name__ == '__main__':
    # This allows running individual functions directly
    if len(sys.argv) > 1 and sys.argv[1] in ['train', 'predict', 'evaluate']:
        function_name = sys.argv[1] + '_main'
        sys.argv = sys.argv[1:]  # Remove the function name from args
        globals()[function_name]()
    else:
        print("Usage: python -m lift_ml.cli [train|predict|evaluate] ...")
        sys.exit(1)
