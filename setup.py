#!/usr/bin/env python
from distutils.core import setup

__VERSION__ = "0.1.0"

setup(
    name="self-projection",
    version=__VERSION__,
    description="SelfProjection is a PyTorch-based neural network layer designed to transform and project high-dimensional data.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author="Konstantin Bashinskiy",
    author_email="sombressoul@gmail.com",
    keywords=["machine learning", "deep learning", "pytorch layers", "pytorch modules"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    url="https://github.com/Sombressoul/self-projection",
    license="Apache",
    packages=["self_projection"],
    install_requires=[
        "torch>=2.0.0",
    ],
    zip_safe=False,
    python_requires=">=3.10.0",
)
