import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
## -------------- Adding Directory --------------- ##

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
ZRO = 7*1e-7                # to avoid numerical instability (default)
ep = 1e-4 
q = 100.0                   # Load, lb.
a = 5.99                    # Tire Radius, inch
x = [0]                     # Horizontal Offset, columns in response (RS)
z = np.arange(0, 30, 1)     # Vertical Distance from surface, rows in response (RS)
H = [6, 10]                 # Thickness inch
E = [500, 40, 10]           # Elastic Modulus ksi
nu = [0.35, 0.4, 0.45]      # Posion's ratio          
isBD= [1, 1]                # interface condition: 1 for full contact, 0 for frictionless
it = 400                    # if convergence error, reduce the it
RS= PyMastic(q, a, x, z, H, E, nu, ZRO, isBounded = isBD, iteration = 40, inverser = 'solve')

## -------------- Mastic Plot: Horizontal Strain --------------- ##
plt.rcParams.update(plt.rcParamsDefault)
plt.style.use('bmh')

fig = plt.figure()
ax1 = fig.add_axes([0.1, 0.1, 1.2, 1])
lg, = ax1.plot(RS["Stress_Z"], z, 'r-o')
font = {'family': 'serif',
        'weight': 'normal',
        'size': 15}
ax1.set_xlabel("Stress (psi)", fontdict=font)
ax1.set_ylabel("Distance From The Surface (in)", fontdict=font)    
ax1.invert_yaxis()
x_dot = [-500, 500]
y_dor = [H[0], H[0]]
ax1.plot(x_dot, y_dor, 'k--', alpha=0.3)
x_dot = [-500, 500]
y_dor = [H[0]+H[1], H[0]+H[1]]
ax1.plot(x_dot, y_dor, 'k--', alpha=0.3)
y_dot = [30, 0]
x_dot = [0, 0]
ax1.plot(x_dot, y_dot, 'k--', alpha=0.9)
textstr = "AC layer"
props = dict(boxstyle='round', edgecolor='none', facecolor='none')
ax1.text(-250, 3,
         s=textstr,
         fontdict=font,
         verticalalignment='center',
         ha= 'center',
         bbox=props)
textstr = "Base layer"
props = dict(boxstyle='round', edgecolor='none', facecolor='none')
ax1.text(-250, 10,
         s=textstr,
         fontdict=font,
         verticalalignment='center',
         ha= 'center',
         bbox=props)
textstr = "Subgrade"
props = dict(boxstyle='round', edgecolor='none', facecolor='none')
ax1.text(-250, 20,
         s=textstr,
         fontdict=font,
         verticalalignment='center',
         ha= 'center',
         bbox=props)

ax1.arrow(0, 25, 40, 0, head_width=1, head_length=5, fc='r', ec='r')
textstr = "Comprissive"
props = dict(boxstyle='round', edgecolor='none', facecolor='none')
ax1.text(90, 25,
         s=textstr,
         fontdict=font,
         verticalalignment='center',
         ha= 'center',
         bbox=props)

ax1.arrow(0, 25, -40, 0, head_width=1, head_length=5, fc='r', ec='r')
textstr = "Tensile"
props = dict(boxstyle='round', edgecolor='none', facecolor='none')
ax1.text(-80, 25,
         s=textstr,
         fontdict=font,
         verticalalignment='center',
         ha= 'center',
         bbox=props)

ax1.xaxis.set_label_position('top') 
ax1.xaxis.tick_top()
# ax1.axes.get_xaxis().set_visible(False)
ax1.set_xlim([-300, 200])
ax1.set_ylim([30, 0])
lg.set_label('Mastic')
ax1.legend(loc="upper right", bbox_to_anchor=(0.95,0.9))

## -------------- Kenpave Plot:  Horizontal Strain --------------- ##
df = pd.read_csv('Kenpave.csv')
ax2 = fig.add_axes([0.1, 1.2, 1.2, 1])

lg, = ax2.plot(df["Vertical_Stress"], z, 'k-o')

ax2.set_xlabel("Stress (psi)", fontdict=font)
ax2.set_ylabel("Distance From The Surface (in)", fontdict=font)    
ax2.invert_yaxis()
x_dot = [-500, 500]
y_dor = [H[0], H[0]]
ax2.plot(x_dot, y_dor, 'k--', alpha=0.3)
x_dot = [-500, 500]
y_dor = [H[0]+H[1], H[0]+H[1]]
ax2.plot(x_dot, y_dor, 'k--', alpha=0.3)
y_dot = [30, 0]
x_dot = [0, 0]
ax2.plot(x_dot, y_dot, 'k--', alpha=0.9)
textstr = "AC layer"
props = dict(boxstyle='round', edgecolor='none', facecolor='none')
ax2.text(-250, 3,
         s=textstr,
         fontdict=font,
         verticalalignment='center',
         ha= 'center',
         bbox=props)
textstr = "Base layer"
props = dict(boxstyle='round', edgecolor='none', facecolor='none')
ax2.text(-250, 10,
         s=textstr,
         fontdict=font,
         verticalalignment='center',
         ha= 'center',
         bbox=props)
textstr = "Subgrade"
props = dict(boxstyle='round', edgecolor='none', facecolor='none')
ax2.text(-250, 20,
         s=textstr,
         fontdict=font,
         verticalalignment='center',
         ha= 'center',
         bbox=props)

ax2.arrow(0, 25, 40, 0, head_width=1, head_length=5, fc='r', ec='r')
textstr = "Comprissive"
props = dict(boxstyle='round', edgecolor='none', facecolor='none')
ax2.text(90, 25,
         s=textstr,
         fontdict=font,
         verticalalignment='center',
         ha= 'center',
         bbox=props)

ax2.arrow(0, 25, -40, 0, head_width=1, head_length=5, fc='r', ec='r')
textstr = "Tensile"
props = dict(boxstyle='round', edgecolor='none', facecolor='none')
ax2.text(-80, 25,
         s=textstr,
         fontdict=font,
         verticalalignment='center',
         ha= 'center',
         bbox=props)

ax2.xaxis.set_label_position('top') 
ax2.xaxis.tick_top()
ax2.set_xlim([-300, 200])
ax2.set_ylim([30, 0])
lg.set_label("KenPave")
ax2.legend(loc="upper right", bbox_to_anchor=(0.95,0.9))
plt.show()


