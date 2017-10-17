from setuptools import setup, find_packages
import os

setup(
    name='robocar42',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'configparser',
        'tensorflow',
        'keras',
        'pandas',
        'scikit-learn',
        'h5py',
        'google-cloud'
    ]
)

