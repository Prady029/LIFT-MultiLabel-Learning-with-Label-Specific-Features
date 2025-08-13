"""
LIFT Transformer: Learning with Label-Specific Features

This module implements the LIFTTransformer, which creates label-specific features
by clustering positive and negative samples separately for each label and transforming
the feature space based on distances to cluster centroids.
"""

from typing import Optional, List, Dict, Any
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class LIFTTransformer(BaseEstimator, TransformerMixin):
    """
    LIFT (Learning with Label-Specific Features) Transformer for multi-label classification.
    
    This transformer creates label-specific features by:
    1. Clustering positive and negative samples separately for each label
    2. Transforming features into distances to cluster centroids
    3. Concatenating distance features across all labels
    
    Parameters
    ----------
    k : int, default=3
        Number of clusters per positive/negative set for each label.
    random_state : int, default=None
        Random seed for reproducibility.
    
    Attributes
    ----------
    cluster_info_ : list of dict
        Cluster centers for each label, containing 'pos' and 'neg' centroids.
    n_features_in_ : int
        Number of features in the input data.
    n_labels_ : int
        Number of labels in the target data.
    """
    
    def __init__(self, k: int = 3, random_state: Optional[int] = None):
        self.k = k
        self.random_state = random_state
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LIFTTransformer":
        """
        Learn label-specific cluster centers from the training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples, n_labels)
            Binary label matrix where each column represents a label.
        
        Returns
        -------
        self : LIFTTransformer
            Fitted transformer.
        """
        X, y = check_X_y(X, y, multi_output=True, accept_sparse=False)
        
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        self.n_features_in_ = X.shape[1]
        self.n_labels_ = y.shape[1]
        self.cluster_info_: List[Dict[str, np.ndarray]] = []
        
        for label_idx in range(self.n_labels_):
            pos_idx = y[:, label_idx] == 1
            neg_idx = y[:, label_idx] == 0
            
            pos_data = X[pos_idx]
            neg_data = X[neg_idx]
            
            # Ensure we don't request more clusters than samples
            pos_k = min(self.k, len(pos_data)) if len(pos_data) > 0 else 1
            neg_k = min(self.k, len(neg_data)) if len(neg_data) > 0 else 1
            
            # Handle edge cases with no positive or negative samples
            if len(pos_data) == 0:
                pos_centers = np.zeros((1, self.n_features_in_))
            else:
                kmeans_pos = KMeans(
                    n_clusters=pos_k, 
                    random_state=self.random_state,
                    n_init=10
                )
                pos_centers = kmeans_pos.fit(pos_data).cluster_centers_
            
            if len(neg_data) == 0:
                neg_centers = np.zeros((1, self.n_features_in_))
            else:
                kmeans_neg = KMeans(
                    n_clusters=neg_k, 
                    random_state=self.random_state,
                    n_init=10
                )
                neg_centers = kmeans_neg.fit(neg_data).cluster_centers_
            
            self.cluster_info_.append({
                'pos': pos_centers, 
                'neg': neg_centers
            })
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the input data using the learned cluster centers.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input feature matrix to transform.
        
        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_transformed_features)
            Transformed feature matrix containing distances to cluster centroids.
        """
        check_is_fitted(self, 'cluster_info_')
        X = check_array(X, accept_sparse=False)
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but transformer was fitted with "
                f"{self.n_features_in_} features."
            )
        
        transformed_features = []
        
        for centers in self.cluster_info_:
            # Calculate distances to positive centroids
            pos_centers = centers['pos']
            pos_distances = np.linalg.norm(
                X[:, None, :] - pos_centers[None, :, :], axis=2
            )
            
            # Calculate distances to negative centroids
            neg_centers = centers['neg']
            neg_distances = np.linalg.norm(
                X[:, None, :] - neg_centers[None, :, :], axis=2
            )
            
            transformed_features.append(pos_distances)
            transformed_features.append(neg_distances)
        
        return np.hstack(transformed_features)
    
    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> np.ndarray:
        """
        Get output feature names for transformation.
        
        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input feature names. If None, generic names will be used.
        
        Returns
        -------
        feature_names_out : ndarray of str
            Output feature names.
        """
        check_is_fitted(self, 'cluster_info_')
        
        feature_names = []
        for label_idx, centers in enumerate(self.cluster_info_):
            n_pos_clusters = centers['pos'].shape[0]
            n_neg_clusters = centers['neg'].shape[0]
            
            # Positive cluster distance features
            for cluster_idx in range(n_pos_clusters):
                feature_names.append(f"label_{label_idx}_pos_cluster_{cluster_idx}_dist")
            
            # Negative cluster distance features
            for cluster_idx in range(n_neg_clusters):
                feature_names.append(f"label_{label_idx}_neg_cluster_{cluster_idx}_dist")
        
        return np.array(feature_names)
