# Route-opt
This repository contains a python package for solving Kaggle's Travelling Santa
problem https://www.kaggle.com/competitions/traveling-santa-2018-prime-paths/overview. 

All underlying modelling code is stored in the folder `routeopt` (also a python package).

## Set up

### Python package versions
Before using the package, ensure that your python packages match, or are 
compatible with the version list contained in environment.yaml. You can
also install a conda environment and use this prior to running any code.

To install:
`conda env create -f environment.yaml`

To activate:
`conda activate route-opt`

### PYTHONPATH environmental variable
Ensure that the directory `Route-opt` is included (appended or preappended)
to your PYTHONPATH. This is required for python to locate the package functions.

## Running
A solution to the Travelling Santa problem can be obtained by running the python script
`solve_tsp.py`. Custom problems can be solved through importing the appropriate
functions, although is not setup for conviently solving arbitary problems.
