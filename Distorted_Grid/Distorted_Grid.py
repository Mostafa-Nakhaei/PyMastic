# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 11:48:17 2020, Distorted Grid (Deflecyion)

@author: Mostafa
"""


import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

UX_org = pd.read_excel('HeatMap_UX.xlsx', header=None)
UZ_org = pd.read_excel('HeatMap_UZ.xlsx', header=None)
UX = np.array(UX_org)
UZ = np.array(UZ_org)
X = UX.reshape(-1) * 1e3
Y = UZ.reshape(-1)  * 1e3
Y = Y - min(Y) #TODO: Double Check with Dr. Timm
Orginal_cross = np.arange(0, 30.1, 0.1)
original_CX = np.zeros((301, 301))
original_CY = np.zeros((301, 301))
non_loaded_cross_section = np.zeros((301, 301))

for i in range(Orginal_cross.shape[0]):
    original_CX[i] = Orginal_cross
original_CX = original_CX.reshape(-1)

for i in range(Orginal_cross.shape[0]):
    original_CY[:, i] = Orginal_cross
original_CY = original_CY.reshape(-1)

for i in range(Orginal_cross.shape[0]):
    non_loaded_cross_section[:, i] = Orginal_cross
non_loaded_cross_section[0] = Orginal_cross

X1 = original_CX + X
Y1 = original_CY + Y
fig = plt.figure()
axes1 = fig.add_axes([0.1, 0.1, 1.2, 1])
axes1.plot(X1, Y1, color='k',  alpha=0.7)
axes1.plot(-X1[1:], Y1[1:], color='k',  alpha=0.7)
axes1.plot(original_CX, original_CY, alpha=0.1, color='r')
axes1.plot(-original_CX[1:], original_CY[1:], alpha=0.1, color='r')
plt.gca().invert_yaxis()
axes1.xaxis.set_tick_params(labeltop='on')
line_X = [-40, 40]
line_Y = [9.25, 9.25]
axes1.plot(line_X, line_Y, 'k--')
line_X = [-40, 40]
line_Y = [15.3, 15.3]
axes1.plot(line_X, line_Y, 'k--')
line_X = [-40, 40]
line_Y = [20.8, 20.8]
axes1.plot(line_X, line_Y, 'k--')
axes1.set_xlim([-31, 31])
axes1.set_ylim([34, -1])
plt.show()


##----------- Distorted grid ------------##
plt.rcParams.update(plt.rcParamsDefault)
plt.style.use('bmh')

matrix_original_X = original_CX.reshape((301, 301))
matrix_original_Y = original_CY.reshape((301, 301))
X1 = original_CX + X
Y1 = original_CY + Y
Matrix_X1 = X1.reshape((301, 301))
Matrix_Y1 = Y1.reshape((301, 301))
fig = plt.figure()
ax1 = fig.add_axes([0.1, 0.1, 1.2, 1])


j = 0
DIS = 7
try:
    for i in range(1000):
        ax1.plot(Matrix_X1[:, j][0::DIS], Matrix_Y1[:, j][0::DIS], '-', color='k')
        ax1.plot(-Matrix_X1[:, j][0::DIS], Matrix_Y1[:, j][0::DIS], '-', color='k')
        ax1.plot(Matrix_X1[j, :][0::DIS], Matrix_Y1[j, :][0::DIS], '-', color='k')
        ax1.plot(-Matrix_X1[j, :][0::DIS], Matrix_Y1[j, :][0::DIS], '-', color='k')   
        # ax1.plot(matrix_original_X[:, j][0::DIS], matrix_original_Y[:, j][0::DIS], '-', color='r', alpha=0.2)
        # ax1.plot(-matrix_original_X[:, j][0::DIS], matrix_original_Y[:, j][0::DIS], '-', color='r', alpha=0.2)
        # ax1.plot(matrix_original_X[j, :][0::DIS], matrix_original_Y[j, :][0::DIS], '-', color='r', alpha=0.2)
        # ax1.plot(-matrix_original_X[j, :][0::DIS], matrix_original_Y[j, :][0::DIS], '-', color='r', alpha=0.2)        
        j=j+DIS
except:
    ax1.xaxis.tick_top()
    ax1.yaxis.tick_left()
    line_X = [-40, 40]
    line_Y = [9.25, 9.25]
    ax1.plot(line_X, line_Y, 'r--')
    line_X = [-40, 40]
    line_Y = [15.3, 15.3]
    ax1.plot(line_X, line_Y, 'r--')
    line_X = [-40, 40]
    line_Y = [20.8, 20.8]
    ax1.plot(line_X, line_Y, 'r--')
    ax1.set_xlim([-35, 35])
    ax1.grid(False)
    plt.gca().invert_yaxis()
    plt.show()

