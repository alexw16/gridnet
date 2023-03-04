#!/usr/bin/env python

from setuptools import setup, find_packages
import gridnet

setup(
    name="gridnet_learn",
    version=gridnet.__version__,
    description="GrID-Net: Granger causal inference on DAGs identifies genomic loci regulating transcription",
    author="Alex Wu, Rohit Singh",
    author_email="alexwu@mit.edu,rsingh@mit.edu",
    url="https://github.com/alexw16/gridnet",
    license="GPLv3",
    packages=find_packages(),
    # entry_points={
    #     "console_scripts": [
    #         "gridnet = gridnet.__main__:main",
    #     ],
    # },
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "torch",
        "scikit-learn",
        "scanpy",
        "schema_learn",

    ],
)
