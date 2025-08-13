import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

try:
    from skopt import BayesSearchCV
    from skopt.space import Integer, Real, Categorical
except ImportError:
    BayesSearchCV = None


class LIFTTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, k=3, random_state=0):
        self.k = k
        self.random_state = random_state
        self.cluster_info_ = None

    def fit(self, X, Y):
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
        if self.cluster_info_ is None:
            raise ValueError("Must fit before transform.")
        transformed_features = []
        for centers in self.cluster_info_:
            pos_dist = np.linalg.norm(X[:, None, :] - centers['pos'][None, :, :], axis=2)
            neg_dist = np.linalg.norm(X[:, None, :] - centers['neg'][None, :, :], axis=2)
            transformed_features.append(pos_dist)
            transformed_features.append(neg_dist)
        return np.hstack(transformed_features)


class LIFTClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, k=3, base_estimator=None, random_state=0,
                 auto_tune=False, tune_params=None, n_iter=10, cv=3, scoring='f1_macro'):
        self.k = k
        self.base_estimator = base_estimator or LogisticRegression(max_iter=1000)
        self.random_state = random_state
        self.auto_tune = auto_tune
        self.tune_params = tune_params
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
        self.model_ = None

    def fit(self, X, Y):
        lift = LIFTTransformer(k=self.k, random_state=self.random_state)
        clf = MultiOutputClassifier(self.base_estimator)

        pipeline = Pipeline([
            ("lift", lift),
            ("clf", clf)
        ])

        if self.auto_tune:
            if BayesSearchCV is None:
                raise ImportError("Install scikit-optimize: `pip install scikit-optimize`")
            if self.tune_params is None:
                self.tune_params = {'lift__k': Integer(1, 10)}
            search = BayesSearchCV(
                estimator=pipeline,
                search_spaces=self.tune_params,
                n_iter=self.n_iter,
                cv=self.cv,
                scoring=self.scoring,
                random_state=self.random_state,
                refit=True
            )
            self.model_ = search.fit(X, Y)
        else:
            self.model_ = pipeline.fit(X, Y)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def score(self, X, Y):
        preds = self.predict(X)
        return f1_score(Y, preds, average='macro')