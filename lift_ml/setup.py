from setuptools import setup, find_packages

setup(
    name="lift-ml",
    version="0.1.0",
    description="LIFT multi-label classifier with optional Bayesian optimization",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "scikit-learn>=1.0",
        "numpy>=1.20",
        "scikit-optimize>=0.9.0"
    ],
    python_requires=">=3.7",
)