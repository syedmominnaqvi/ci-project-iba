"""
Setup script for the PII/PHI detection package.
"""
from setuptools import setup, find_packages

setup(
    name="pii_detection_evolutionary",
    version="0.1.0",
    description="Evolutionary algorithm for PII/PHI detection in unstructured data",
    author="Syed Momin Naqvi, Rehmat Gul",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "spacy",
        "deap",
        "presidio-analyzer",
        "presidio-anonymizer",
        "transformers",
        "torch",
        "pytest",
        "matplotlib",
    ],
    python_requires=">=3.7",
)