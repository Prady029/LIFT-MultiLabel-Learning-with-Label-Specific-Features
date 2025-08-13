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

***
