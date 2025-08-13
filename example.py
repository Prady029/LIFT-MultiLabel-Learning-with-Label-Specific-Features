#!/usr/bin/env python3
"""
Example script demonstrating the LIFT multi-label classifier usage.

This script shows how to:
1. Generate synthetic multi-label data
2. Train a LIFT classifier
3. Evaluate its performance
4. Use Bayesian optimization for hyperparameter tuning
"""

import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from skopt.space import Integer

from lift_ml import LIFTClassifier, LIFTTransformer


def main():
    print("🚀 LIFT Multi-Label Classifier Example")
    print("=" * 50)
    
    # Generate synthetic multi-label dataset
    print("📊 Generating synthetic multi-label dataset...")
    X, y = make_multilabel_classification(
        n_samples=1000,
        n_features=20,
        n_classes=5,
        n_labels=2,
        random_state=42,
        return_indicator=True
    )[:2]
    
    print(f"Dataset shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Label frequency: {y.mean(axis=0)}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Example 1: Basic LIFT Classifier
    print("\\n🔧 Example 1: Basic LIFT Classifier")
    print("-" * 40)
    
    clf_basic = LIFTClassifier(k=3, random_state=42)
    clf_basic.fit(X_train, y_train)
    
    y_pred_basic = clf_basic.predict(X_test)
    f1_basic = f1_score(y_test, y_pred_basic, average='macro')
    
    print(f"F1 Score (macro): {f1_basic:.4f}")
    
    # Example 2: LIFT with Bayesian Optimization
    print("\\n⚡ Example 2: LIFT with Bayesian Optimization")
    print("-" * 50)
    
    # Define search space
    search_space = {
        'lift__k': Integer(1, 8),
    }
    
    clf_tuned = LIFTClassifier(
        auto_tune=True,
        tune_params=search_space,
        n_iter=10,  # Reduced for demo purposes
        cv=3,
        random_state=42
    )
    
    print("🔍 Performing Bayesian optimization...")
    clf_tuned.fit(X_train, y_train)
    
    y_pred_tuned = clf_tuned.predict(X_test)
    f1_tuned = f1_score(y_test, y_pred_tuned, average='macro')
    
    print(f"Best parameters: {clf_tuned.best_params_}")
    print(f"Best CV score: {clf_tuned.best_score_:.4f}")
    print(f"Test F1 Score (macro): {f1_tuned:.4f}")
    
    # Example 3: Using LIFT Transformer directly
    print("\\n🔄 Example 3: Using LIFT Transformer directly")
    print("-" * 50)
    
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.ensemble import RandomForestClassifier
    
    # Transform features
    transformer = LIFTTransformer(k=3, random_state=42)
    X_train_transformed = transformer.fit_transform(X_train, y_train)
    X_test_transformed = transformer.transform(X_test)
    
    print(f"Original features: {X_train.shape[1]}")
    print(f"Transformed features: {X_train_transformed.shape[1]}")
    
    # Use with Random Forest
    rf_clf = MultiOutputClassifier(
        RandomForestClassifier(n_estimators=50, random_state=42)
    )
    rf_clf.fit(X_train_transformed, y_train)
    
    y_pred_rf = rf_clf.predict(X_test_transformed)
    f1_rf = f1_score(y_test, y_pred_rf, average='macro')
    
    print(f"Random Forest + LIFT F1 Score: {f1_rf:.4f}")
    
    # Example 4: Probability predictions
    print("\\n📊 Example 4: Probability Predictions")
    print("-" * 40)
    
    y_proba = clf_basic.predict_proba(X_test[:5])  # First 5 samples
    
    print("Sample probability predictions:")
    for i, (true_labels, pred_proba) in enumerate(zip(y_test[:5], y_proba)):
        print(f"Sample {i+1}:")
        print(f"  True labels: {true_labels}")
        print(f"  Predicted probabilities: {pred_proba}")
        print(f"  Predicted labels: {(pred_proba > 0.5).astype(int)}")
        print()
    
    # Summary
    print("\\n📈 Summary")
    print("=" * 30)
    print(f"Basic LIFT:                {f1_basic:.4f}")
    print(f"Optimized LIFT:            {f1_tuned:.4f}")
    print(f"Random Forest + LIFT:      {f1_rf:.4f}")
    print(f"Improvement from tuning:   {f1_tuned - f1_basic:+.4f}")
    
    print("\\n✅ Example completed successfully!")


if __name__ == "__main__":
    main()
