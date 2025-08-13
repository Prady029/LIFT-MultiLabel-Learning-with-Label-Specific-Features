"""
LIFT Classifier: Complete multi-label classifier using LIFT transformation

This module implements the LIFTClassifier, which combines the LIFTTransformer
with multi-output classification and optional Bayesian hyperparameter optimization.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from .transformer import LIFTTransformer

try:
    from skopt import BayesSearchCV
    from skopt.space import Integer
    HAS_SKOPT = True
except ImportError:
    BayesSearchCV = None
    Integer = None
    HAS_SKOPT = False


class LIFTClassifier(BaseEstimator, ClassifierMixin):
    """
    LIFT (Learning with Label-Specific Features) Classifier for multi-label classification.
    
    This classifier combines LIFT feature transformation with multi-output classification.
    It can optionally use Bayesian optimization for hyperparameter tuning.
    
    Parameters
    ----------
    k : int, default=3
        Number of clusters per positive/negative set for each label.
    base_estimator : estimator object, default=None
        Base estimator to use for each label. If None, LogisticRegression is used.
    random_state : int, default=None
        Random seed for reproducibility.
    auto_tune : bool, default=False
        Whether to use Bayesian optimization for hyperparameter tuning.
    tune_params : dict, default=None
        Parameter search space for Bayesian optimization.
    n_iter : int, default=10
        Number of iterations for Bayesian optimization.
    cv : int, default=3
        Number of cross-validation folds for hyperparameter tuning.
    scoring : str, default='f1_macro'
        Scoring metric for hyperparameter tuning.
    """
    
    def __init__(
        self,
        k=3,
        base_estimator=None,
        random_state=None,
        auto_tune=False,
        tune_params=None,
        n_iter=10,
        cv=3,
        scoring='f1_macro'
    ):
        self.k = k
        self.base_estimator = base_estimator
        self.random_state = random_state
        self.auto_tune = auto_tune
        self.tune_params = tune_params
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
    
    def fit(self, X, y):
        """
        Fit the LIFT classifier to the training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples, n_labels)
            Binary label matrix where each column represents a label.
        
        Returns
        -------
        self : LIFTClassifier
            Fitted classifier.
        """
        X, y = check_X_y(X, y, multi_output=True, accept_sparse=False)
        
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        self.n_features_in_ = X.shape[1]
        self.classes_ = [np.array([0, 1]) for _ in range(y.shape[1])]
        
        # Create the base estimator
        if self.base_estimator is None:
            base_est = LogisticRegression(max_iter=1000, random_state=self.random_state)
        else:
            base_est = clone(self.base_estimator)
        
        # Create the pipeline
        lift_transformer = LIFTTransformer(k=self.k, random_state=self.random_state)
        classifier = MultiOutputClassifier(base_est)
        
        pipeline = Pipeline([
            ("lift", lift_transformer),
            ("classifier", classifier)
        ])
        
        # Apply Bayesian optimization if requested
        if self.auto_tune:
            if not HAS_SKOPT:
                raise ImportError(
                    "Bayesian optimization requires scikit-optimize. "
                    "Install it with: pip install scikit-optimize"
                )
            
            if self.tune_params is None:
                from skopt.space import Integer as _Integer
                self.tune_params = {'lift__k': _Integer(1, 10)}
            
            from skopt import BayesSearchCV as _BayesSearchCV
            search = _BayesSearchCV(
                estimator=pipeline,
                search_spaces=self.tune_params,
                n_iter=self.n_iter,
                cv=self.cv,
                scoring=self.scoring,
                random_state=self.random_state,
                refit=True,
                n_jobs=1
            )
            self.model_ = search.fit(X, y)
        else:
            self.model_ = pipeline.fit(X, y)
        
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        """
        check_is_fitted(self, 'model_')
        X = check_array(X, accept_sparse=False)
        
        if hasattr(self, 'n_features_in_') and X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but classifier was fitted with "
                f"{self.n_features_in_} features."
            )
        
        return self.model_.predict(X)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
        """
        check_is_fitted(self, 'model_')
        X = check_array(X, accept_sparse=False)
        
        if hasattr(self, 'n_features_in_') and X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but classifier was fitted with "
                f"{self.n_features_in_} features."
            )
        
        # Get the actual estimator
        if self.auto_tune:
            estimator = getattr(self.model_, 'best_estimator_', self.model_)
        else:
            estimator = self.model_
        
        # Get probabilities for each label
        proba_outputs = estimator.predict_proba(X)
        
        # Extract positive class probabilities for each label
        probas = []
        for label_proba in proba_outputs:
            if label_proba.shape[1] == 2:  # Binary classification
                probas.append(label_proba[:, 1])  # Probability of positive class
            else:
                probas.append(label_proba[:, 0])  # Single class case
        
        return np.column_stack(probas)
    
    def score(self, X, y, sample_weight=None):
        """
        Return the mean macro-averaged F1 score on the given test data and labels.
        """
        check_is_fitted(self, 'model_')
        y_pred = self.predict(X)
        
        # Convert to numpy array for consistency
        y = np.asarray(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        score = f1_score(y, y_pred, average='macro', sample_weight=sample_weight)
        return float(score)
    
    @property
    def best_params_(self):
        """Get the best parameters found during hyperparameter optimization."""
        if not self.auto_tune:
            return {}
        
        check_is_fitted(self, 'model_')
        return getattr(self.model_, 'best_params_', {})
    
    @property
    def best_score_(self):
        """Get the best score achieved during hyperparameter optimization."""
        if not self.auto_tune:
            return float('nan')
        
        check_is_fitted(self, 'model_')
        score = getattr(self.model_, 'best_score_', float('nan'))
        return float(score)
