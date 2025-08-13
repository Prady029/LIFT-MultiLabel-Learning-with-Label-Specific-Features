import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


def lift_transform(X, Y, k=3):
    """
    LIFT feature transformation.
    Args:
        X: Original feature matrix (n_samples, n_features)
        Y: Binary label matrix (n_samples, n_labels)
        k: No. of clusters for positive and negative examples per label
    Returns:
        transformed_X: Transformed feature representation (n_samples, n_labels*2*k)
        cluster_centers: List of positive/negative cluster centers for each label
    """
    n_samples, n_features = X.shape
    n_labels = Y.shape[1]
    
    transformed_features = []
    cluster_info = []

    for label_idx in range(n_labels):
        pos_idx = Y[:, label_idx] == 1
        neg_idx = Y[:, label_idx] == 0

        pos_data, neg_data = X[pos_idx], X[neg_idx]
        
        # Positive and negative clustering
        pos_kmeans = KMeans(n_clusters=min(k, len(pos_data)), random_state=0).fit(pos_data)
        neg_kmeans = KMeans(n_clusters=min(k, len(neg_data)), random_state=0).fit(neg_data)

        cluster_centers = {
            'pos': pos_kmeans.cluster_centers_,
            'neg': neg_kmeans.cluster_centers_
        }
        cluster_info.append(cluster_centers)

        # Distance to positive and negative centers
        pos_distances = np.linalg.norm(X[:, None, :] - cluster_centers['pos'][None, :, :], axis=2)
        neg_distances = np.linalg.norm(X[:, None, :] - cluster_centers['neg'][None, :, :], axis=2)

        transformed_features.append(pos_distances)
        transformed_features.append(neg_distances)

    transformed_X = np.hstack(transformed_features)
    return transformed_X, cluster_info


# Example usage:
if __name__ == "__main__":
    # Generate a sample multi-label dataset
    X, Y = make_multilabel_classification(n_samples=500, n_features=20, n_classes=5,
                                          n_labels=2, random_state=42)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Apply LIFT transformation
    X_train_lift, cluster_info = lift_transform(X_train, Y_train, k=3)
    X_test_lift, _ = lift_transform(X_test, Y_test, k=3)  # In practice, use train's centroids

    # Train one classifier per label
    classifiers = []
    preds = []
    for label_idx in range(Y_train.shape[1]):
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train_lift, Y_train[:, label_idx])
        classifiers.append(clf)
        preds.append(clf.predict(X_test_lift))

    preds = np.array(preds).T

    # Evaluate with macro F1-score
    print("Macro F1:", f1_score(Y_test, preds, average='macro'))