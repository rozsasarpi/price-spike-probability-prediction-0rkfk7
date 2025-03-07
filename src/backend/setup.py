#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import codecs
from setuptools import setup, find_packages  # setuptools >= 61.0.0

# Get the absolute path of the directory containing this file
here = os.path.abspath(os.path.dirname(__file__))

def read(rel_path):
    """
    Read a file with proper encoding.
    
    Args:
        rel_path (str): Relative path to the file
        
    Returns:
        str: Content of the file
    """
    with codecs.open(os.path.join(here, rel_path), 'r', 'utf-8') as fp:
        return fp.read()

def get_requirements():
    """
    Parse requirements from requirements.txt file.
    
    Returns:
        List[str]: List of package requirements
    """
    try:
        requirements = read('requirements.txt')
        return [line for line in requirements.split('\n') 
                if line and not line.startswith('#')]
    except FileNotFoundError:
        # Fall back to minimum required dependencies if requirements.txt is missing
        return [
            "numpy>=1.24.0",
            "pandas>=2.0.0",
            "scikit-learn>=1.2.0",
            "xgboost>=1.7.0",
            "lightgbm>=3.3.0",
            "pandera>=0.15.0",
            "joblib>=1.2.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "plotly>=5.14.0",
            "pytest>=7.3.0",
            "pydantic>=2.0.0",
            "hydra-core>=1.3.0",
            "black>=23.3.0",
            "isort>=5.12.0",
            "mypy>=1.3.0",
            "typing-extensions>=4.5.0",
        ]

# Get long description from README.md
try:
    long_description = read('README.md')
except FileNotFoundError:
    long_description = """
    ERCOT RTLMP Spike Prediction System
    
    A system for forecasting the probability of price spikes in the 
    Real-Time Locational Marginal Price (RTLMP) market before 
    day-ahead market closure.
    """

setup(
    name="ercot-rtlmp-prediction",
    version="0.1.0",
    description="ERCOT RTLMP spike prediction system for forecasting the probability of price spikes in the Real-Time Locational Marginal Price market",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Data Science Team",
    author_email="datascience@example.com",
    url="https://github.com/example/ercot-rtlmp-prediction",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    keywords="ercot, energy, price-prediction, machine-learning, time-series",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=get_requirements(),
    entry_points={
        "console_scripts": [
            "rtlmp-predict=src.backend.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "src.backend": ["config/hydra/*.yaml"],
    },
    zip_safe=False,
)