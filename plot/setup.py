#!/usr/bin/env python3

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setuptools.setup(
    name="plot",
    version="0.0.0",
    author="Amery Gration",
    author_email="amerygration@hotmail.com",
    description="Plotting utilities",
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    python_requires=">=3.6.9",
    install_requires=["numpy>=1.19.5", "scipy>=1.5.4", "matplotlib>=3.3.4"],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
    ),
)
