import numpy as np
import pytest
from sklearn.datasets import make_multilabel_classification

# Import using the full path since we're in src/ layout
import sys
sys.path.insert(0, 'src')
from lift_ml import LIFTTransformer, LIFTClassifier


class TestLIFTTransformer:
    """Test cases for LIFTTransformer."""
    
    def test_fit(self):
        """Test that transformer can be fitted."""
        X = np.random.rand(100, 10)
        Y = np.random.randint(0, 2, size=(100, 5))
        transformer = LIFTTransformer(k=3, random_state=42)
        transformer.fit(X, Y)
        assert transformer.cluster_info_ is not None
        assert len(transformer.cluster_info_) == Y.shape[1]
        assert hasattr(transformer, 'n_features_in_')
        assert hasattr(transformer, 'n_labels_')

    def test_transform(self):
        """Test that transformer can transform data."""
        X = np.random.rand(100, 10)
        Y = np.random.randint(0, 2, size=(100, 5))
        transformer = LIFTTransformer(k=3, random_state=42)
        transformer.fit(X, Y)
        transformed_X = transformer.transform(X)
        assert transformed_X.shape[0] == X.shape[0]
        # Each label contributes 2*k features (pos and neg clusters)
        expected_features = sum(
            info['pos'].shape[0] + info['neg'].shape[0] 
            for info in transformer.cluster_info_
        )
        assert transformed_X.shape[1] == expected_features

    def test_fit_transform(self):
        """Test fit_transform method."""
        X = np.random.rand(100, 10)
        Y = np.random.randint(0, 2, size=(100, 5))
        transformer = LIFTTransformer(k=3, random_state=42)
        transformed_X = transformer.fit_transform(X, Y)
        assert transformed_X.shape[0] == X.shape[0]

    def test_single_label(self):
        """Test with single label."""
        X = np.random.rand(50, 5)
        Y = np.random.randint(0, 2, size=(50, 1))
        transformer = LIFTTransformer(k=2, random_state=42)
        transformer.fit(X, Y)
        transformed_X = transformer.transform(X)
        assert transformed_X.shape[0] == X.shape[0]
        assert transformer.n_labels_ == 1

    def test_edge_cases(self):
        """Test edge cases like few samples."""
        # Test with very few samples
        X = np.random.rand(5, 3)
        Y = np.array([[1], [0], [1], [0], [1]])
        transformer = LIFTTransformer(k=3, random_state=42)
        transformer.fit(X, Y)
        transformed_X = transformer.transform(X)
        assert transformed_X.shape[0] == X.shape[0]

    def test_get_feature_names_out(self):
        """Test feature name generation."""
        X = np.random.rand(50, 5)
        Y = np.random.randint(0, 2, size=(50, 3))
        transformer = LIFTTransformer(k=2, random_state=42)
        transformer.fit(X, Y)
        feature_names = transformer.get_feature_names_out()
        assert len(feature_names) == transformer.transform(X).shape[1]
        assert all('label_' in name for name in feature_names)


class TestLIFTClassifier:
    """Test cases for LIFTClassifier."""
    
    def test_basic_fit_predict(self):
        """Test basic functionality."""
        X, Y = make_multilabel_classification(
            n_samples=100, n_features=20, n_classes=5, random_state=42,
            return_indicator=True
        )[:2]  # Only take X and Y
        
        clf = LIFTClassifier(k=2, random_state=42)
        clf.fit(X, Y)
        
        # Test predictions
        y_pred = clf.predict(X)
        assert y_pred.shape == Y.shape
        assert set(np.unique(y_pred)).issubset({0, 1})
        
        # Test probabilities
        y_proba = clf.predict_proba(X)
        assert y_proba.shape == Y.shape
        assert np.all(y_proba >= 0) and np.all(y_proba <= 1)

    def test_score(self):
        """Test scoring functionality."""
        X, Y = make_multilabel_classification(
            n_samples=100, n_features=20, n_classes=3, random_state=42,
            return_indicator=True
        )[:2]  # Only take X and Y
        
        clf = LIFTClassifier(k=2, random_state=42)
        clf.fit(X, Y)
        score = clf.score(X, Y)
        assert isinstance(score, float)
        assert 0 <= score <= 1

    def test_with_custom_estimator(self):
        """Test with custom base estimator."""
        from sklearn.ensemble import RandomForestClassifier
        
        X, Y = make_multilabel_classification(
            n_samples=100, n_features=20, n_classes=3, random_state=42,
            return_indicator=True
        )[:2]  # Only take X and Y
        
        base_est = RandomForestClassifier(n_estimators=10, random_state=42)
        clf = LIFTClassifier(k=2, base_estimator=base_est, random_state=42)
        clf.fit(X, Y)
        
        y_pred = clf.predict(X)
        assert y_pred.shape == Y.shape

    @pytest.mark.skipif(
        not hasattr(pytest, 'importorskip') or 
        pytest.importorskip('skopt', minversion='0.9.0') is None,
        reason="scikit-optimize not available"
    )
    def test_auto_tune(self):
        """Test Bayesian optimization functionality."""
        try:
            from skopt.space import Integer
        except ImportError:
            pytest.skip("scikit-optimize not available")
        
        X, Y = make_multilabel_classification(
            n_samples=100, n_features=10, n_classes=2, random_state=42,
            return_indicator=True
        )[:2]  # Only take X and Y
        
        clf = LIFTClassifier(
            auto_tune=True,
            tune_params={'lift__k': Integer(1, 3)},
            n_iter=2,  # Small number for testing
            cv=2,
            random_state=42
        )
        clf.fit(X, Y)
        
        assert hasattr(clf, 'best_params_')
        assert hasattr(clf, 'best_score_')
        assert isinstance(clf.best_params_, dict)

    def test_single_label_classification(self):
        """Test with single label (binary classification)."""
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, size=(100, 1))
        
        clf = LIFTClassifier(k=2, random_state=42)
        clf.fit(X, y)
        
        y_pred = clf.predict(X)
        assert y_pred.shape == y.shape


# Ensure pytest discovers the tests
if __name__ == "__main__":
    pytest.main(["-v", "--tb=short", "--disable-warnings"])
