import os
import sys
import pandas as pd
import numpy as np

def get_parent_dir(n=1):
    """ returns the n-th parent dicrectory of the current
    working directory """
    current_path = os.path.dirname(os.path.abspath(__file__))
    for k in range(n):
        current_path = os.path.dirname(current_path)
    return current_path
sys.path.append(os.path.join(get_parent_dir(1), "main"))
from MLE import PyMastic

## -------------- Validation: STart --------------- ##
q = 100.0                   # lb.
a = 5.99                    # inch
x = [0, 8]                  # number of columns in response
z = [0, 9.99, 10.01]        # number of rows in response
H = [10, 6]                 # inch
E = [500, 40, 10]           # ksi
nu = [0.35, 0.4, 0.45]
ZRO = 7*1e-7                # to avoid numerical instability
isBD= [0, 0]
it = 10

RS = PyMastic(q,a,x,z,H,E,nu, ZRO, isBounded = isBD, iteration = it, inverser = 'solve')