"""Dataflow worker setup for preprocess.py for AstroNet."""
import setuptools

NAME = 'astronet-preprocess'
VERSION = '1.0'
REQUIRED_PACKAGES = ['pandas==0.23.0', 'pandas-gbq', 'astropy', 'pydl']

setuptools.setup(
    name=NAME,
    version=VERSION,
    packages=setuptools.find_packages(),
    install_requires=REQUIRED_PACKAGES)
