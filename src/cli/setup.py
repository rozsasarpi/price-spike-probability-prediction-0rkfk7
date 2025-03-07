#!/usr/bin/env python

"""
Setup script for the ERCOT RTLMP spike prediction system CLI package.
This file defines package metadata, dependencies, and installation configuration using setuptools to make the CLI component installable and distributable.
"""

import codecs  # Standard
import os  # Standard

# Third-party libraries
import setuptools  # version >=61.0.0

# Internal imports
from . import __version__, __author__, __description__  # src/cli/__init__.py


here = os.path.abspath(os.path.dirname(__file__))

def read(rel_path):
    """
    Read a file with proper encoding
    """
    with codecs.open(os.path.join(here, rel_path), 'r', encoding='utf-8') as fp:
        return fp.read()

def get_requirements():
    """
    Parse requirements from requirements.txt file
    """
    requirements_path = os.path.join(here, 'requirements.txt')
    with open(requirements_path, 'r') as f:
        requirements = f.read().splitlines()
    return [req for req in requirements if req.strip() and not req.startswith('#')]

# Get the long description from the README file
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Get the requirements from the requirements.txt file
requirements = get_requirements()

setuptools.setup(
    name='ercot-rtlmp-cli',
    version=__version__,
    description=__description__,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=__author__,
    author_email='datascience@example.com',
    url='https://github.com/example/ercot-rtlmp-prediction',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis'
    ],
    keywords='ercot, energy, price-prediction, cli, command-line',
    packages=setuptools.find_packages(),
    python_requires='>=3.10',
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'rtlmp_predict=src.cli.main:main'
        ]
    },
    include_package_data=True,
    package_data={
        'src.cli': ['config/*.yaml', 'scripts/*.sh']
    },
    zip_safe=False,
)