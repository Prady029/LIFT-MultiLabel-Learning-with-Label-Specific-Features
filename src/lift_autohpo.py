from skopt import BayesSearchCV
from skopt.space import Integer  # or Real/Categorical as needed
from .lift_sklearn_pipeline import LIFTTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_multilabel_classification

def build_lift_pipeline(k=3, random_state=42, auto_tune=False, tune_params=None, n_iter=10, cv=3, scoring='f1_macro'):
    """
    Build a pipeline for LIFT + classifier. Optionally wrap in Bayesian HPO optimizer.
    Args:
        k: number of clusters per label
        random_state: random seed
        auto_tune: bool, if True wraps pipeline in BayesSearchCV
        tune_params: dict or None, search space for tuning (must use param names)
        n_iter: int, number of BO iterations if auto_tune
        cv: int, cross-validation folds for BayesSearchCV
        scoring: string, scoring strategy
    Returns:
        pipeline or BayesSearchCV instance
    """
    # Build the LIFT transformer pipeline as before
    lift = LIFTTransformer(k=k, random_state=random_state)
    base_clf = MultiOutputClassifier(LogisticRegression(max_iter=1000))
    pipeline = Pipeline([
        ("lift", lift),
        ("clf", base_clf)
    ])

    if auto_tune:
        if tune_params is None:
            # Example search space: tune 'lift__k' between 1 and 10
            tune_params = {'lift__k': Integer(1, 10)}
        # BayesSearchCV takes the pipeline, parameter dict, and CV/scoring settings
        pipeline = BayesSearchCV(
            estimator=pipeline,
            search_spaces=tune_params,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            random_state=random_state,
            refit=True
        )
    return pipeline

# Example usage
if __name__ == "__main__":
    X, Y = make_multilabel_classification(n_samples=500, n_features=20, n_classes=5, n_labels=2, random_state=42)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    # Set auto_tune=True to enable Bayesian HPO over 'lift__k'
    pipeline = build_lift_pipeline(auto_tune=True, n_iter=20)
    pipeline.fit(X_train, Y_train)
    preds = pipeline.predict(X_test)
    print("Macro F1-score:", f1_score(Y_test, preds, average='macro'))
