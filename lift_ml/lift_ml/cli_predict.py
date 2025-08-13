import argparse
import pandas as pd
import joblib
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Use a trained LIFT model to predict on a dataset.")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model .pkl file")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV dataset for prediction")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Path to save predictions CSV")
    parser.add_argument("--drop-cols", type=str, nargs="+", help="Columns to drop before prediction (e.g., IDs from training)")
    parser.add_argument("--proba", action="store_true",
                        help="If set, output only class probabilities instead of hard labels")
    parser.add_argument("--both", action="store_true",
                        help="If set, output both labels AND probabilities in the same file")
    parser.add_argument("--include-input", action="store_true",
                        help="If set, include the original input columns in the output CSV")
    parser.add_argument("--id-col", type=str,
                        help="If set, include ONLY this column from the input for identifying rows (e.g. sample_id)")
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model}...")
    clf = joblib.load(args.model)

    # Load dataset
    print(f"Loading dataset from {args.data}...")
    df = pd.read_csv(args.data)

    # Handle features for prediction
    if args.drop_cols:
        df_features = df.drop(columns=args.drop_cols)
    else:
        df_features = df

    # Predictions
    predictions = clf.predict(df_features.values)
    preds_df = pd.DataFrame(predictions, columns=[f"label_{i}" for i in range(predictions.shape[1])])

    # Probabilities
    probas_all = clf.predict_proba(df_features.values)
    probas = np.column_stack([p[:, 1] if p.shape[1] > 1 else p[:, 0] for p in probas_all])
    probas_df = pd.DataFrame(probas, columns=[f"label_{i}_proba" for i in range(probas.shape[1])])

    # Decide prediction type
    if args.both:
        out_df = pd.concat([preds_df, probas_df], axis=1)
    elif args.proba:
        out_df = probas_df
    else:
        out_df = preds_df

    # Add original data or ID column if requested
    if args.include_input and args.id_col:
        print("⚠️ Both --include-input and --id-col provided. Using only --id-col for clarity.")
        args.include_input = False

    if args.include_input:
        out_df = pd.concat([df, out_df], axis=1)
    elif args.id_col:
        if args.id_col not in df.columns:
            raise ValueError(f"ID column '{args.id_col}' not found in dataset.")
        out_df.insert(0, args.id_col, df[args.id_col])

    # Save output
    out_df.to_csv(args.output, index=False)
    print(f"Output saved to {args.output}")