import argparse
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from .cli_predict import find_best_thresholds, parse_hybrid_thresholds_by_name

def main():
    parser = argparse.ArgumentParser(description="Report optimal thresholds for a trained LIFT model.")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--val-data", type=str, required=True, help="Validation dataset CSV path")
    parser.add_argument("--val-target-cols", type=str, nargs="+", required=True, help="Label column names")
    parser.add_argument("--optimize-metric", type=str, required=True,
                        choices=["f1", "precision", "recall"], help="Metric to optimize")
    parser.add_argument("--optimize-global", action="store_true", help="Report one global threshold for all labels")
    parser.add_argument("--hybrid-thresholds", type=str,
                        help="Optional fixed per-label overrides by NAME, e.g. 'spam:0.8,important:0.3'")
    parser.add_argument("--output", type=str, help="Optional CSV file to save thresholds")
    args = parser.parse_args()

    # Load model
    clf = joblib.load(args.model)

    # Load validation data
    val_df = pd.read_csv(args.val_data)
    y_val = val_df[args.val_target_cols].values
    X_val = val_df.drop(columns=args.val_target_cols).values
    n_labels = y_val.shape[1]
    label_names = args.val_target_cols

    # Predict validation probabilities
    val_probas_all = clf.predict_proba(X_val)
    val_probas = np.column_stack([
        p[:, 1] if p.shape[1] > 1 else p[:, 0] for p in val_probas_all
    ])

    # Optimize thresholds
    if args.optimize_global:
        global_thr = find_best_thresholds(y_val, val_probas,
                                          metric=args.optimize_metric,
                                          per_label=False)
        thresholds = np.full(n_labels, global_thr)
    else:
        thresholds = find_best_thresholds(y_val, val_probas,
                                          metric=args.optimize_metric,
                                          per_label=True)

    # Apply name-based hybrid overrides
    if args.hybrid_thresholds:
        override_map = parse_hybrid_thresholds_by_name(args.hybrid_thresholds)
        for lbl_name, thr in override_map.items():
            if lbl_name not in label_names:
                raise ValueError(f"Label '{lbl_name}' not found in validation target columns {label_names}")
            idx = label_names.index(lbl_name)
            # Ensure thresholds is an array before assignment
            if isinstance(thresholds, float) or isinstance(thresholds, np.float64):
                thresholds = np.full(n_labels, thresholds)
            thresholds[idx] = thr

    # Prepare output dataframe
    report_df = pd.DataFrame({
        "label": label_names,
        "threshold": thresholds
    })

    # Print report
    print("\n📊 LIFT Threshold Report")
    print("-" * 40)
    for lbl, thr in zip(label_names, thresholds):
        print(f"{lbl}: {thr:.3f}")
    print("-" * 40)
    if args.optimize_global:
        print(f"✅ Global threshold: {thresholds[0]:.3f} (applied to all labels)")
    print()

    # Save to CSV if requested
    if args.output:
        report_df.to_csv(args.output, index=False)
        print(f"✅ Thresholds saved to {args.output}")