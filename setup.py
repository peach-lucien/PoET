from setuptools import setup, find_packages

import numpy
from Cython.Build import cythonize
from setuptools import Extension

requirements = (
   'cython==0.29.35',
   'numpy==1.23.5',
   'scipy==1.10.1',
   'matplotlib==3.7.1',
   'scikit-learn==1.2.2',
   'networkx==3.1',
   'mediapipe>=0.10.10',
   'scikit-video>=1.1.11',
)

setup(name='PoET',
      version='0.1',
      packages=find_packages(exclude=["examples*"]),
      python_requires='>=3.8',
      install_requires=requirements,
      )
