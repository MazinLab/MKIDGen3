import setuptools
from setuptools.command.install import install
from setuptools.command.develop import develop
import subprocess
import numpy
from setuptools.extension import Extension

#pip install -e git+https://github.com/mazinlab/mkidgen3.git@develop#egg=mkidgen3


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mkidgen3",
    version="0.1",
    author="MazinLab, J. Bailey et al.",
    author_email="mazinlab@ucsb.edu",
    description="An UVOIR MKID Detector package for the ZCU111",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MazinLab/MKIDGen3",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX",
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research"],
    install_requires=[
        'fpbinary',
        'pyserial',
    ]
)
