import argparse
import pandas as pd
import joblib

def main():
    parser = argparse.ArgumentParser(description="Use a trained LIFT model to predict on a dataset.")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model .pkl file")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV dataset for prediction")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Path to save predictions CSV")
    parser.add_argument("--drop-cols", type=str, nargs="+", help="Columns to drop before prediction (e.g. IDs)")
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model}...")
    clf = joblib.load(args.model)

    # Load dataset
    print(f"Loading dataset from {args.data}...")
    df = pd.read_csv(args.data)

    if args.drop_cols:
        df_features = df.drop(columns=args.drop_cols)
    else:
        df_features = df

    # Predict
    print("Predicting...")
    predictions = clf.predict(df_features.values)

    # Save
    pred_df = pd.DataFrame(predictions, columns=[f"label_{i}" for i in range(predictions.shape[1])])
    pred_df.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")