# LIFT algorithm for multilabel classification

The LIFT algorithm (Learning with Label-specific Features) is a multi-label classification method that constructs and uses **label-specific features** to improve classification for each label individually. Instead of using the same feature set for all labels, LIFT performs the following steps:[^1_1][^1_2][^1_3]

- For each label, LIFT divides training data into positive (instances with the label) and negative (instances without the label) sets.[^1_2][^1_3][^1_1]
- It applies clustering (typically k-means) to both groups, selecting representative centroids for each label's positive and negative clusters.[^1_3][^1_2]
- Each instance is then represented by its distances to these centroids, yielding new, label-specific features for each label.[^1_2]
- Separate binary classifiers are trained for each label using these transformed features, enabling more discriminative power closer to each label's unique characteristics.

This approach differs from traditional multi-label classifiers that use a shared feature space for all labels, potentially overlooking the specific information relevant to each. LIFT has been validated as more effective than several standard methods on benchmark datasets.[^1_3][^1_1]

In summary, LIFT enhances multi-label classification by performing feature construction tailored to each label which enables the classifier to focus on the most relevant information for each label’s presence or absence.

### How It Works:

1. **Clustering per label** – For each label, split positive and negative samples, run KMeans separately.
2. **Distance transformation** – Replace original features with distances to each centroid.
3. **Train binary classifiers per label** – Each classifier uses the label-specific transformed space.
4. **Evaluate** – We use macro-averaged F1 here.

**Note:**

- In **real LIFT**, when transforming the test set, you should use the **cluster centers learned from the training set** rather than re-clustering test data. In my quick example, I re-clustered for the test just to keep the function call simple.
- You can store the centers from training and apply them to compute distances for the test set.

Once installed locally with:

```bash
pip install -e .
```

Now you can use it anywhere:

```python
from lift_ml import LIFTClassifier
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split

X, Y = make_multilabel_classification(n_samples=500, n_features=20, n_classes=5, random_state=42)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

clf = LIFTClassifier(auto_tune=True, n_iter=15)
clf.fit(X_train, Y_train)
print("Macro F1:", clf.score(X_test, Y_test))
```

You can also run:

```bash
lift-train --data dataset.csv \
           --target-cols label1 label2 label3 \
           --k 3 \
           --auto-tune \
           --n-iter 15 \
           --test-size 0.2 \
           --model-out my_lift_model.pkl
```

This CLI will:

1. Load your CSV file
2. Split into features \& labels
3. Train the LIFT model (with or without Bayesian tuning)
4. Print macro-F1 score
5. Save the trained model to a `.pkl` file

**Training**

```bash
lift-train --data dataset.csv \
           --target-cols label1 label2 label3 \
           --k 3 --auto-tune --n-iter 15 \
           --model-out lift_model.pkl
```


**Prediction**

```bash
lift-predict --model lift_model.pkl \
             --data new_data.csv \
             --drop-cols id \
             --output preds.csv
```

### Examples:

**1. Labels + Original Inputs**

```bash
lift-predict --model lift_model.pkl --data new_data.csv --include-input
```

**2. Probabilities + Original Inputs**

```bash
lift-predict --model lift_model.pkl --data new_data.csv --proba --include-input
```

**3. Labels + Probabilities + Original Inputs**

```bash
lift-predict --model lift_model.pkl --data new_data.csv --both --include-input
```

**4. Just predictions + row IDs**

```bash
lift-predict --model lift_model.pkl --data new_data.csv --id-col sample_id --output preds.csv
```

```
sample_id,label_0,label_1,...
```


**4.Labels + probabilities + row IDs**

```bash
lift-predict --model lift_model.pkl --data new_data.csv --both --id-col sample_id --output preds_probas.csv
```

```
sample_id,label_0,label_1,...,label_0_proba,label_1_proba,...
```


**5.Full dataset with predictions**

```bash
lift-predict --model lift_model.pkl --data new_data.csv --include-input --both --output full_preds.csv
```
**1. Set a 0.7 threshold for binarization**

```bash
lift-predict --model lift_model.pkl --data new_data.csv --threshold 0.7 --output preds_70.csv
```

**2. Output both probabilities and thresholded labels**

```bash
lift-predict --model lift_model.pkl --data new_data.csv --both --threshold 0.7 --output preds_and_probas_70.csv
```

**3. Probabilities only, but threshold stored in separate labels**

```bash
lift-predict --model lift_model.pkl --data new_data.csv --proba --threshold 0.4
```

*(In this case, only probabilities appear, but labels are binarized internally for `--both` mode.)*

***

With this, your CLI now supports **custom decision thresholds** for any probability output scenario, which is essential when optimizing for recall or precision in multi-label classification.

### 1. Automatic F1-based threshold optimization

```bash
lift-predict \
  --model lift_model.pkl \
  --data new_data.csv \
  --val-data val_data.csv \
  --val-target-cols label1 label2 label3 \
  --optimize-metric f1 \
  --both \
  --output preds_auto_f1.csv
```

- Uses validation set to find best per-label thresholds that maximize **F1**.


### 2. Optimize for recall but only output hard labels

```bash
lift-predict \
  --model lift_model.pkl \
  --data new_data.csv \
  --val-data val_data.csv \
  --val-target-cols label1 label2 \
  --optimize-metric recall \
  --output preds_max_recall.csv
```


### 3. Fixed threshold for comparison

```bash
lift-predict --model lift_model.pkl --data new_data.csv --threshold 0.7
```
- If you run with:

```bash
--optimize-metric f1 --optimize-global
```

It will:

1. Load `--val-data` and `--val-target-cols`
2. Try thresholds 0.00 → 1.00 in small steps
3. Pick **ONE single threshold** for all labels that maximizes macro‑averaged F1, or the metric you choose.

- If you **omit** `--optimize-global`, it defaults to **per‑label** thresholds.

### **One global threshold for all labels (F1 metric)**

```bash
lift-predict \
  --model lift_model.pkl \
  --data new_data.csv \
  --val-data val_data.csv \
  --val-target-cols label1 label2 label3 \
  --optimize-metric f1 \
  --optimize-global \
  --both \
  --output preds_global_f1.csv
```


### **Compare with per-label optimization**

```bash
lift-predict \
  --model lift_model.pkl \
  --data new_data.csv \
  --val-data val_data.csv \
  --val-target-cols label1 label2 label3 \
  --optimize-metric f1 \
  --both \
  --output preds_per_label_f1.csv
```

### 1️⃣ Hybrid with fixed global

Global = 0.6, Label 0 override to 0.8, Label 3 override to 0.3

```bash
lift-predict \
  --model lift_model.pkl \
  --data new_data.csv \
  --threshold 0.6 \
  --hybrid-thresholds "0:0.8,3:0.3" \
  --both \
  --output preds_hybrid.csv
```


***

### 2️⃣ Hybrid with optimized global (F1) + overrides

Global optimized on validation set, but loosen label 2

```bash
lift-predict \
  --model lift_model.pkl \
  --data new_data.csv \
  --val-data val_data.csv \
  --val-target-cols label1 label2 label3 \
  --optimize-metric f1 \
  --optimize-global \
  --hybrid-thresholds "2:0.4" \
  --both \
  --output preds_hybrid_opt.csv
```


***

### 3️⃣ Hybrid with per-label auto-opt, plus a hard-coded override

Per-label auto-opt from validation, but set label 1 fixed at 0.9

```bash
lift-predict \
  --model lift_model.pkl \
  --data new_data.csv \
  --val-data val_data.csv \
  --val-target-cols label1 label2 label3 \
  --optimize-metric recall \
  --hybrid-thresholds "1:0.9" \
  --output preds_hybrid_perlabel.csv
```
This gives **max flexibility**:

- Global threshold for consistency
- Fine-tuned per-label overrides if needed
- Supports both **fixed** and **auto-optimized** setups

### 1️⃣ Fixed global + specific label overrides

```bash
lift-predict \
  --model lift_model.pkl \
  --data new_data.csv \
  --threshold 0.6 \
  --hybrid-thresholds "spam:0.8,important:0.3" \
  --both \
  --output preds_hybrid_names.csv
```


***

### 2️⃣ Optimized global + name-based overrides

```bash
lift-predict \
  --model lift_model.pkl \
  --data new_data.csv \
  --val-data val_data.csv \
  --val-target-cols spam important urgent \
  --optimize-metric f1 \
  --optimize-global \
  --hybrid-thresholds "urgent:0.4" \
  --both \
  --output preds_hybrid_opt_names.csv
```


***

### 3️⃣ Per-label auto-opt + one name override

```bash
lift-predict \
  --model lift_model.pkl \
  --data new_data.csv \
  --val-data val_data.csv \
  --val-target-cols spam important urgent \
  --optimize-metric recall \
  --hybrid-thresholds "important:0.9" \
  --both \
  --output preds_hybrid_perlabel_names.csv
```

Now we can specify **threshold overrides in human-friendly label names**, matching wer CSV column headers.
It fully supports:

- Fixed + override
- Auto-optimized (global or per-label) + override
- Mixing with `--include-input` and `--id-col`