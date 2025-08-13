import argparse
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from .lift import LIFTClassifier


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate LIFT multi-label classifier.")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument("--target-cols", type=str, nargs="+", required=True,
                        help="List of column names that contain labels")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test size fraction")
    parser.add_argument("--k", type=int, default=3, help="Number of clusters in LIFT per class")
    parser.add_argument("--auto-tune", action="store_true", help="Enable Bayesian hyperparameter optimization")
    parser.add_argument("--n-iter", type=int, default=10, help="Number of iterations for Bayesian search")
    parser.add_argument("--model-out", type=str, default="lift_model.pkl", help="Path to save trained model")

    args = parser.parse_args()

    # Load CSV
    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)

    # Separate features and labels
    Y = df[args.target_cols].values
    X = df.drop(columns=args.target_cols).values

    # Split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=args.test_size, random_state=42)

    # Init model
    clf = LIFTClassifier(
        k=args.k,
        auto_tune=args.auto_tune,
        n_iter=args.n_iter
    )

    # Train
    print("Training model...")
    clf.fit(X_train, Y_train)

    # Evaluate
    print("Evaluating...")
    pred = clf.predict(X_test)
    f1 = f1_score(Y_test, pred, average='macro')
    print(f"Macro F1-score: {f1:.4f}")

    # Save model
    joblib.dump(clf, args.model_out)
    print(f"Model saved to {args.model_out}")