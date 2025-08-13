import argparse
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

def find_best_thresholds(y_true, probas, metric='f1', steps=101):
    """Find best threshold per label for given metric."""
    best_thresholds = []
    metrics_map = {
        'f1': f1_score,
        'precision': precision_score,
        'recall': recall_score
    }
    metric_func = metrics_map[metric]

    for label_idx in range(y_true.shape[1]):
        best_thr, best_score = 0.5, -1
        for thr in np.linspace(0, 1, steps):
            preds = (probas[:, label_idx] >= thr).astype(int)
            try:
                score = metric_func(y_true[:, label_idx], preds, zero_division=0)
            except Exception:
                score = 0
            if score > best_score:
                best_score = score
                best_thr = thr
        best_thresholds.append(best_thr)
    return np.array(best_thresholds)

def main():
    parser = argparse.ArgumentParser(description="Use a trained LIFT model to predict on a dataset.")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model .pkl file")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV dataset for prediction")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Path to save predictions CSV")
    parser.add_argument("--drop-cols", type=str, nargs="+", help="Columns to drop before prediction")
    parser.add_argument("--proba", action="store_true", help="Output only probabilities")
    parser.add_argument("--both", action="store_true", help="Output both labels and probabilities")
    parser.add_argument("--include-input", action="store_true", help="Include original features in output")
    parser.add_argument("--id-col", type=str, help="Include only a specific ID column")
    parser.add_argument("--threshold", type=float, help="Custom fixed threshold for all labels")
    parser.add_argument("--val-data", type=str, help="Path to validation CSV for auto metric tuning")
    parser.add_argument("--val-target-cols", type=str, nargs="+", help="Target columns in validation CSV")
    parser.add_argument("--optimize-metric", type=str,
                        choices=["f1", "precision", "recall"],
                        help="Automatically find per-label thresholds using this metric (needs --val-data)")
    args = parser.parse_args()

    # Load model
    clf = joblib.load(args.model)

    # Load main dataset
    df = pd.read_csv(args.data)
    if args.drop_cols:
        df_features = df.drop(columns=args.drop_cols)
    else:
        df_features = df

    # Predict probabilities
    probas_all = clf.predict_proba(df_features.values)
    probas = np.column_stack([p[:, 1] if p.shape[1] > 1 else p[:, 0] for p in probas_all])
    probas_df = pd.DataFrame(probas, columns=[f"label_{i}_proba" for i in range(probas.shape[1])])

    # Determine thresholds
    if args.optimize_metric:
        if not args.val_data or not args.val_target_cols:
            raise ValueError("Must provide --val-data and --val-target-cols for optimize-metric mode.")
        val_df = pd.read_csv(args.val_data)
        y_val = val_df[args.val_target_cols].values
        X_val = val_df.drop(columns=args.val_target_cols).values
        val_probas_all = clf.predict_proba(X_val)
        val_probas = np.column_stack([p[:, 1] if p.shape[1] > 1 else p[:, 0] for p in val_probas_all])
        thresholds_per_label = find_best_thresholds(y_val, val_probas, metric=args.optimize_metric)
        print(f"Optimized thresholds per label: {thresholds_per_label}")
        predictions = (probas >= thresholds_per_label).astype(int)
    elif args.threshold is not None:
        predictions = (probas >= args.threshold).astype(int)
    else:
        predictions = clf.predict(df_features.values)

    preds_df = pd.DataFrame(predictions, columns=[f"label_{i}" for i in range(predictions.shape[1])])

    # Decide output type
    if args.both:
        out_df = pd.concat([preds_df, probas_df], axis=1)
    elif args.proba:
        out_df = probas_df
    else:
        out_df = preds_df

    # Include original data or ID column
    if args.include_input and args.id_col:
        print("⚠️ Both --include-input and --id-col provided. Using only --id-col.")
        args.include_input = False
    if args.include_input:
        out_df = pd.concat([df, out_df], axis=1)
    elif args.id_col:
        if args.id_col not in df.columns:
            raise ValueError(f"ID column '{args.id_col}' not found.")
        out_df.insert(0, args.id_col, df[args.id_col])

    # Save
    out_df.to_csv(args.output, index=False)
    print(f"Output saved to {args.output}")