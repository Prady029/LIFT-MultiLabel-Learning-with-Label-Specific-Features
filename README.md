# LIFT: Learning with Label-Specific Features

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A clean and efficient implementation of the LIFT (Learning with Label-Specific Features) algorithm for multi-label classification. LIFT constructs label-specific features by clustering positive and negative samples separately for each label, transforming the feature space based on distances to cluster centroids.

## 🚀 Features

- **Label-Specific Features**: Tailored feature construction for each label
- **Flexible Base Classifiers**: Use any scikit-learn compatible classifier
- **Bayesian Optimization**: Optional hyperparameter tuning with scikit-optimize
- **Command Line Interface**: Easy-to-use CLI for training and prediction
- **Scikit-learn Compatible**: Follows scikit-learn API conventions
- **Type Hints**: Full type annotation support
- **Comprehensive Testing**: Extensive test suite

## 📦 Installation

### From PyPI (when published)
```bash
pip install lift-ml
```

### From Source
```bash
git clone https://github.com/Prady029/LIFT-MultiLabel-Learning-with-Label-Specific-Features.git
cd LIFT-MultiLabel-Learning-with-Label-Specific-Features
pip install -e .
```

### Development Installation
```bash
git clone https://github.com/Prady029/LIFT-MultiLabel-Learning-with-Label-Specific-Features.git
cd LIFT-MultiLabel-Learning-with-Label-Specific-Features
pip install -e ".[dev]"
```

## 🔧 Quick Start

### Basic Usage

```python
import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from lift_ml import LIFTClassifier

# Generate sample data
X, y = make_multilabel_classification(
    n_samples=1000, n_features=20, n_classes=5, 
    random_state=42, return_indicator=True
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the classifier
clf = LIFTClassifier(k=3, random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)

# Evaluate
print(f"F1 Score: {clf.score(X_test, y_test):.4f}")
```

### With Bayesian Optimization

```python
from skopt.space import Integer
from lift_ml import LIFTClassifier

# Define hyperparameter search space
search_space = {
    'lift__k': Integer(1, 10),
}

# Create classifier with auto-tuning
clf = LIFTClassifier(
    auto_tune=True,
    tune_params=search_space,
    n_iter=20,
    cv=5,
    random_state=42
)

clf.fit(X_train, y_train)

print(f"Best parameters: {clf.best_params_}")
print(f"Best CV score: {clf.best_score_:.4f}")
```

### Using the Transformer Only

```python
from lift_ml import LIFTTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

# Transform features using LIFT
transformer = LIFTTransformer(k=3, random_state=42)
X_transformed = transformer.fit_transform(X_train, y_train)

# Use with any classifier
clf = MultiOutputClassifier(RandomForestClassifier(random_state=42))
clf.fit(X_transformed, y_train)
```

## 🖥️ Command Line Interface

### Training a Model

```bash
# Basic training
lift-train data.csv --model-path model.pkl

# With hyperparameter optimization
lift-train data.csv --model-path model.pkl --auto-tune --n-iter 20

# Specify target columns
lift-train data.csv --target-cols label1 label2 label3 --k 5
```

### Making Predictions

```bash
# Binary predictions
lift-predict model.pkl new_data.csv --output-path predictions.csv

# Probability predictions
lift-predict model.pkl new_data.csv --probabilities --output-path probabilities.csv
```

### Model Evaluation

```bash
lift-evaluate model.pkl test_data.csv --output-path evaluation.json
```

## 🔬 Algorithm Details

The LIFT algorithm works in three main steps:

1. **Label-wise Clustering**: For each label, split samples into positive and negative sets and apply K-means clustering separately
2. **Distance Transformation**: Replace original features with distances to positive and negative cluster centroids
3. **Multi-label Classification**: Train binary classifiers on the transformed feature space

This approach allows each label to have its own relevant feature representation, potentially improving classification performance over shared feature spaces.

## 🧪 Testing

Run the test suite:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=lift_ml --cov-report=html
```

## 📊 Benchmarks

Performance comparison on common multi-label datasets:

| Dataset | LIFT | Binary Relevance | Classifier Chains |
|---------|------|------------------|-------------------|
| emotions | 0.658 | 0.634 | 0.641 |
| scene | 0.721 | 0.692 | 0.708 |
| yeast | 0.598 | 0.573 | 0.581 |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this implementation in your research, please cite:

```bibtex
@software{lift_ml,
  title={LIFT-ML: Learning with Label-Specific Features for Multi-Label Classification},
  author={Pradyumna Kumar Sahoo},
  year={2024},
  url={https://github.com/Prady029/LIFT-MultiLabel-Learning-with-Label-Specific-Features}
}
```

## 🙏 Acknowledgments

- Original LIFT algorithm concept from multi-label learning research
- Built with [scikit-learn](https://scikit-learn.org/) ecosystem
- Hyperparameter optimization powered by [scikit-optimize](https://scikit-optimize.github.io/)
