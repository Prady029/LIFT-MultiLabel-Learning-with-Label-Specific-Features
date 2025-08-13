import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_multilabel_classification
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


class LIFTTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, k=3, random_state=0):
        """
        LIFT transformer for multi-label classification.
        Args:
            k (int): Number of clusters per positive/negative set for each label.
            random_state (int): Random seed for reproducibility.
        """
        self.k = k
        self.random_state = random_state
        self.cluster_info_ = None

    def fit(self, X, Y):
        """
        Learn label-specific cluster centers from the training data.
        """
        n_labels = Y.shape[1]
        self.cluster_info_ = []

        for label_idx in range(n_labels):
            pos_idx = Y[:, label_idx] == 1
            neg_idx = Y[:, label_idx] == 0

            pos_data, neg_data = X[pos_idx], X[neg_idx]

            pos_k = min(self.k, len(pos_data))
            neg_k = min(self.k, len(neg_data))

            pos_centers = KMeans(n_clusters=pos_k, random_state=self.random_state).fit(pos_data).cluster_centers_
            neg_centers = KMeans(n_clusters=neg_k, random_state=self.random_state).fit(neg_data).cluster_centers_

            self.cluster_info_.append({'pos': pos_centers, 'neg': neg_centers})

        return self

    def transform(self, X):
        """
        Transform features into LIFT distance-based features using stored cluster centers.
        """
        if self.cluster_info_ is None:
            raise ValueError("Must fit LIFTTransformer before calling transform()")

        transformed_features = []
        for centers in self.cluster_info_:
            pos_dist = np.linalg.norm(X[:, None, :] - centers['pos'][None, :, :], axis=2)
            neg_dist = np.linalg.norm(X[:, None, :] - centers['neg'][None, :, :], axis=2)
            transformed_features.append(pos_dist)
            transformed_features.append(neg_dist)

        return np.hstack(transformed_features)


# ==== Example Usage ====
if __name__ == "__main__":
    # Create synthetic dataset
    X, Y, _, _ = make_multilabel_classification(n_samples=500, n_features=20, n_classes=5,
                                                n_labels=2, random_state=42)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Build pipeline
    pipeline = Pipeline([
        ("lift", LIFTTransformer(k=3, random_state=42)),
        ("clf", MultiOutputClassifier(LogisticRegression(max_iter=1000)))
    ])

    # Train pipeline
    pipeline.fit(X_train, Y_train)

    # Predict
    preds = pipeline.predict(X_test)

    # Evaluate
    print("Macro F1-score:", f1_score(Y_test, preds, average='macro'))