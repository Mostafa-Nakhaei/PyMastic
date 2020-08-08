from Main.MLE import PyMastic
import numpy as np


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


print("\nDisplacement [0, 0]: ")
print(RS['Displacement_Z'][0, 0])

print("\nSigma Z is[0, 0]: ")
print(RS['Stress_Z'][0, 0])

print("\nDisplacement_H is [0, 0]: ")
print(RS['Displacement_H'][0, 0])

print("\nSigma T is [0, 0]: ")
print(RS['Stress_T'][0, 0])

print("\nDisplacement [1, 1]: ")
print(RS['Displacement_Z'][1, 0])

print("\nSigma Z is [1, 1]: ")
print(RS['Stress_Z'][1, 0])

print("\nSigma R is [1, 1]: ")
print(RS['Stress_R'][1, 0])

print("\nSigma T is [1, 1]: ")
print(RS['Stress_T'][1, 0])