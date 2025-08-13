import argparse
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

def parse_hybrid_thresholds_by_name(arg_string):
    """Parse 'label_name:thr,...' into dict {str: float}."""
    mapping = {}
    for pair in arg_string.split(","):
        if ":" in pair:
            name, thr = pair.split(":")
            mapping[name.strip()] = float(thr.strip())
    return mapping

def find_best_threshold_fixed(y_true, probas, metric='f1', steps=101):
    metrics_map = {
        'f1': f1_score,
        'precision': precision_score,
        'recall': recall_score
    }
    func = metrics_map[metric]
    best_thr, best_score = 0.5, -1
    for thr in np.linspace(0, 1, steps):
        preds = (probas >= thr).astype(int)
        score = func(y_true, preds, average='macro', zero_division=0)
        if score > best_score:
            best_score, best_thr = score, thr
    return best_thr

def find_best_thresholds(y_true, probas, metric='f1', steps=101, per_label=True):
    metrics_map = {
        'f1': f1_score,
        'precision': precision_score,
        'recall': recall_score
    }
    func = metrics_map[metric]
    if per_label:
        thresholds = []
        for i in range(y_true.shape[1]):
            best_thr, best_score = 0.5, -1
            for thr in np.linspace(0, 1, steps):
                preds = (probas[:, i] >= thr).astype(int)
                score = func(y_true[:, i], preds, zero_division=0)
                if score > best_score:
                    best_score, best_thr = score, thr
            thresholds.append(best_thr)
        return np.array(thresholds)
    else:
        return find_best_threshold_fixed(y_true, probas, metric, steps)

def main():
    parser = argparse.ArgumentParser(description="Predict using LIFT model with flexible thresholding.")
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
    parser.add_argument("--optimize-metric", type=str, choices=["f1", "precision", "recall"])
    parser.add_argument("--optimize-global", action="store_true")
    parser.add_argument("--hybrid-thresholds", type=str,
                        help="Custom per-label overrides by NAME, format 'labelA:0.8,labelB:0.3'")
    args = parser.parse_args()

    # Load model
    clf = joblib.load(args.model)

    # Load main dataset for prediction
    df = pd.read_csv(args.data)
    df_features = df.drop(columns=args.drop-cols) if args.drop_cols else df

    # Predict probabilities
    probas_all = clf.predict_proba(df_features.values)
    probas = np.column_stack([p[:, 1] if p.shape[1] > 1 else p[:, 0] for p in probas_all])
    n_labels = probas.shape[1]

    # Default label names if no target cols given (label_0, label_1,...)
    label_names = [f"label_{i}" for i in range(n_labels)]
    probas_df = pd.DataFrame(probas, columns=[f"{name}_proba" for name in label_names])

    thresholds = None

    # Optimization
    if args.optimize_metric:
        if not args.val_data or not args.val_target_cols:
            raise ValueError("Must provide --val-data and --val-target-cols for optimize-metric.")
        label_names = args.val_target_cols  # Now we have actual names!
        val_df = pd.read_csv(args.val_data)
        y_val = val_df[args.val_target_cols].values
        X_val = val_df.drop(columns=args.val_target_cols).values
        val_probas_all = clf.predict_proba(X_val)
        val_probas = np.column_stack([p[:, 1] if p.shape[1] > 1 else p[:, 0] for p in val_probas_all])

        if args.optimize_global:
            global_thr = find_best_thresholds(y_val, val_probas, metric=args.optimize_metric, per_label=False)
            thresholds = np.full(n_labels, global_thr)
        else:
            thresholds = find_best_thresholds(y_val, val_probas, metric=args.optimize_metric, per_label=True)

    elif args.threshold is not None:
        thresholds = np.full(n_labels, args.threshold)

    # Apply hybrid overrides by label name
    if args.hybrid_thresholds:
        override_map = parse_hybrid_thresholds_by_name(args.hybrid_thresholds)
        if thresholds is None:
            thresholds = np.full(n_labels, 0.5)
        for lbl_name, thr in override_map.items():
            if lbl_name not in label_names:
                raise ValueError(f"Label name '{lbl_name}' not found among: {label_names}")
            idx = label_names.index(lbl_name)
            thresholds[idx] = thr
        print(f"Hybrid thresholds applied: {{name:thr for name,thr in zip(label_names, thresholds)}}")

    # Generate predictions
    if thresholds is not None:
        predictions = (probas >= thresholds).astype(int)
    else:
        predictions = clf.predict(df_features.values)

    preds_df = pd.DataFrame(predictions, columns=label_names)

    # Decide output
    if args.both:
        out_df = pd.concat([preds_df, probas_df], axis=1)
    elif args.proba:
        out_df = probas_df
    else:
        out_df = preds_df

    # Include original data or ID col
    if args.include_input and args.id_col:
        print("⚠️ Both --include-input and --id-col given. Using only --id-col.")
        args.include_input = False
    if args.include_input:
        out_df = pd.concat([df, out_df], axis=1)
    elif args.id_col:
        if args.id_col not in df.columns:
            raise ValueError(f"ID column '{args.id_col}' not found in dataset.")
        out_df.insert(0, args.id_col, df[args.id_col])

    out_df.to_csv(args.output, index=False)
    print(f"✅ Output saved to {args.output}")