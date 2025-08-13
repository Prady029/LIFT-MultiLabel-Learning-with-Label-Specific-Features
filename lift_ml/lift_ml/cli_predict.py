import argparse
import pandas as pd
import joblib
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Use a trained LIFT model to predict on a dataset.")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model .pkl file")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV dataset for prediction")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Path to save predictions CSV")
    parser.add_argument("--drop-cols", type=str, nargs="+", help="Columns to drop before prediction (e.g. IDs)")
    parser.add_argument("--proba", action="store_true",
                        help="If set, output class probabilities instead of hard labels")
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

    if args.proba:
        print("Predicting probabilities...")
        probas_all = clf.predict_proba(df_features.values)
        # predict_proba for multi-output classifier returns a list of arrays, one per label
        # Convert to a 2D array by stacking probability of positive class
        probas = np.column_stack([p[:, 1] if p.shape[1] > 1 else p[:, 0] for p in probas_all])
        out_df = pd.DataFrame(probas, columns=[f"label_{i}_proba" for i in range(probas.shape[1])])
    else:
        print("Predicting labels...")
        predictions = clf.predict(df_features.values)
        out_df = pd.DataFrame(predictions, columns=[f"label_{i}" for i in range(predictions.shape[1])])

    out_df.to_csv(args.output, index=False)
    print(f"Output saved to {args.output}")