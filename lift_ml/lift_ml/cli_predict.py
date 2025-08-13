import argparse
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

def find_best_thresholds(y_true, probas, metric='f1', steps=101, per_label=True):
    """Find best threshold per label or global for given metric."""
    metrics_map = {
        'f1': f1_score,
        'precision': precision_score,
        'recall': recall_score
    }
    metric_func = metrics_map[metric]

    if per_label:
        # Per-label thresholds
        best_thresholds = []
        for label_idx in range(y_true.shape[1]):
            best_thr, best_score = 0.5, -1
            for thr in np.linspace(0, 1, steps):
                preds = (probas[:, label_idx] >= thr).astype(int)
                score = metric_func(y_true[:, label_idx], preds, zero_division=0)
                if score > best_score:
                    best_score, best_thr = score, thr
            best_thresholds.append(best_thr)
        return np.array(best_thresholds)
    else:
        # One global threshold for all labels
        best_thr, best_score = 0.5, -1
        for thr in np.linspace(0, 1, steps):
            preds = (probas >= thr).astype(int)
            score = metric_func(y_true, preds, average='macro', zero_division=0)
            if score > best_score:
                best_score, best_thr = score, thr
        return best_thr

def main():
    parser = argparse.ArgumentParser(description="Use a trained LIFT model to predict on a dataset.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, default="predictions.csv")
    parser.add_argument("--drop-cols", type=str, nargs="+")
    parser.add_argument("--proba", action="store_true")
    parser.add_argument("--both", action="store_true")
    parser.add_argument("--include-input", action="store_true")
    parser.add_argument("--id-col", type=str)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--val-data", type=str)
    parser.add_argument("--val-target-cols", type=str, nargs="+")
    parser.add_argument("--optimize-metric", type=str,
                        choices=["f1", "precision", "recall"])
    parser.add_argument("--optimize-global", action="store_true",
                        help="If set with --optimize-metric, finds one global threshold for all labels")
    args = parser.parse_args()

    # Load model
    clf = joblib.load(args.model)

    # Load main prediction dataset
    df = pd.read_csv(args.data)
    df_features = df.drop(columns=args.drop_cols) if args.drop_cols else df

    # Predict probabilities
    probas_all = clf.predict_proba(df_features.values)
    probas = np.column_stack([p[:, 1] if p.shape[1] > 1 else p[:, 0] for p in probas_all])
    probas_df = pd.DataFrame(probas, columns=[f"label_{i}_proba" for i in range(probas.shape[1])])

    # Determine thresholds
    if args.optimize_metric:
        if not args.val_data or not args.val_target_cols:
            raise ValueError("Must provide --val-data and --val-target-cols with optimize-metric.")
        val_df = pd.read_csv(args.val_data)
        y_val = val_df[args.val_target_cols].values
        X_val = val_df.drop(columns=args.val_target_cols).values

        val_probas_all = clf.predict_proba(X_val)
        val_probas = np.column_stack([p[:, 1] if p.shape[1] > 1 else p[:, 0] for p in val_probas_all])

        if args.optimize_global:
            thr = find_best_thresholds(y_val, val_probas, metric=args.optimize_metric, per_label=False)
            print(f"Optimized global threshold: {thr:.3f}")
            predictions = (probas >= thr).astype(int)
        else:
            thresholds_per_label = find_best_thresholds(y_val, val_probas, metric=args.optimize_metric, per_label=True)
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

    # Include original input or ID column
    if args.include_input and args.id_col:
        print("⚠️ Both --include-input and --id-col provided. Using only --id-col.")
        args.include_input = False
    if args.include_input:
        out_df = pd.concat([df, out_df], axis=1)
    elif args.id_col:
        if args.id_col not in df.columns:
            raise ValueError(f"ID column '{args.id_col}' not found.")
        out_df.insert(0, args.id_col, df[args.id_col])

    out_df.to_csv(args.output, index=False)
    print(f"Output saved to {args.output}")