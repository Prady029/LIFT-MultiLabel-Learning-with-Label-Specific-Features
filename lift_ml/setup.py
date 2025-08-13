from setuptools import setup, find_packages

setup(
    name="lift-ml",
    version="0.2.0",
    description="LIFT multi-label classifier with Bayesian optimization and CLI",
    author="Your Name",
    author_email="you@example.com",
    packages=find_packages(),
    install_requires=[
        "scikit-learn>=1.0",
        "numpy>=1.20",
        "scikit-optimize>=0.9.0",
        "pandas>=1.3",
        "joblib>=1.0"
    ],
entry_points={
    "console_scripts": [
        "lift-train=lift_ml.cli_train:main",
        "lift-predict=lift_ml.cli_predict:main",
        "lift-threshold-report=lift_ml.cli_threshold_report:main",
    ]
},
    python_requires=">=3.7"
)