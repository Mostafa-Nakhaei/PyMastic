import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as mpl
from decimal import Decimal

############################################################
####################     U     ############################
############################################################
font = {'family': 'serif',
'weight': 'normal',
'size': 11,}
df = pd.read_excel('HeatMap_U.xlsx')
df = df * 1e3
df.set_index(np.arange(0,30.10,0.1), drop=True, inplace=True)
plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
ax = sns.heatmap(df, cmap="coolwarm")
plt.xlabel("radial offset, inch", fontdict=font)
plt.ylabel("vertical distance from the top, inch")
plt.title('Displacement Vector (milli-inch)')
x = [0, 500]
y = [92.5, 92.5]
ax.plot(x, y, 'k--')
x = [0, 500]
y = [153, 153]
ax.plot(x, y, 'k--')
ax.xaxis.set_label_position('top') 
##-------------------------Text:START-------------------------------------##
textstr = "AC"
props = dict(boxstyle='round', edgecolor='none', facecolor='none')
ax.text(156, 31.5, s=textstr,  fontdict=font,
        verticalalignment='top', bbox=props)
textstr = "CTB"
props = dict(boxstyle='round', edgecolor='none', facecolor='none')
ax.text(156, 120.5, s=textstr,  fontdict=font,
        verticalalignment='top', bbox=props)
textstr = "Subgrade"
props = dict(boxstyle='round', edgecolor='none', facecolor='none')
ax.text(156, 220.5, s=textstr,  fontdict=font,
        verticalalignment='top', bbox=props)
##-------------------------Text:END-------------------------------------##
plt.show()



############################################################
####################     UX     ############################
############################################################

font = {'family': 'serif',
'weight': 'normal',
'size': 11,}
df = pd.read_excel('HeatMap_UX.xlsx')
df = df * 1e3
df.set_index(np.arange(0,30.10,0.1), drop=True, inplace=True)
plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
ax = sns.heatmap(df, cmap="coolwarm")
plt.xlabel("radial offset, inch", fontdict=font)
plt.ylabel("vertical distance from the top, inch")
plt.title('Horizontal Displacement (milli-inch)')
x = [0, 500]
y = [92.5, 92.5]
ax.plot(x, y, 'k--')
x = [0, 500]
y = [153, 153]
ax.plot(x, y, 'k--')
ax.xaxis.set_label_position('top') 

##-------------------------Text:START-------------------------------------##
textstr = "AC"
props = dict(boxstyle='round', edgecolor='none', facecolor='none')
ax.text(156, 31.5, s=textstr,  fontdict=font,
        verticalalignment='top', bbox=props)
textstr = "CTB"
props = dict(boxstyle='round', edgecolor='none', facecolor='none')
ax.text(156, 120.5, s=textstr,  fontdict=font,
        verticalalignment='top', bbox=props)
textstr = "Subgrade"
props = dict(boxstyle='round', edgecolor='none', facecolor='none')
ax.text(156, 220.5, s=textstr,  fontdict=font,
        verticalalignment='top', bbox=props)
##-------------------------Text:END-------------------------------------##
plt.show()

############################################################
####################     UZ     ############################
############################################################
font = {'family': 'serif',
'weight': 'normal',
'size': 11,}
df = pd.read_excel('HeatMap_UZ.xlsx')
df = df * 1e3
df.set_index(np.arange(0,30.10,0.1), drop=True, inplace=True)
plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
ax = sns.heatmap(df, cmap="coolwarm")
plt.xlabel("radial offset, inch", fontdict=font)
plt.ylabel("Vertical Distance from the top, inch")
plt.title('Vertical Displacement (milli-inch)')
x = [0, 500]
y = [92.5, 92.5]
ax.plot(x, y, 'k--')
x = [0, 500]
y = [153, 153]
ax.plot(x, y, 'k--')
ax.xaxis.set_label_position('top') 

##-------------------------Text:START-------------------------------------##
textstr = "AC"
props = dict(boxstyle='round', edgecolor='none', facecolor='none')
ax.text(156, 31.5, s=textstr,  fontdict=font,
        verticalalignment='top', bbox=props)
textstr = "CTB"
props = dict(boxstyle='round', edgecolor='none', facecolor='none')
ax.text(156, 120.5, s=textstr,  fontdict=font,
        verticalalignment='top', bbox=props)
textstr = "Subgrade"
props = dict(boxstyle='round', edgecolor='none', facecolor='none')
ax.text(156, 220.5, s=textstr,  fontdict=font,
        verticalalignment='top', bbox=props)
##-------------------------Text:END-------------------------------------##
plt.show()



############################################################
####################     EX     ############################
############################################################
df_EX = pd.read_excel('HeatMap_EX.xlsx')
df_EX = df_EX * -1
df_EX.set_index(np.arange(0, 30.1, 0.1), drop=True, inplace=True)
plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
ax = sns.heatmap(df_EX, cmap="coolwarm")
plt.xlabel("radial offset, inch", size=10)
plt.ylabel("vertical distance from the top, inch", size=10)
plt.title('Horizontal Strain (1e6)')
x = [0, 500]
y = [92.5, 92.5]
ax.plot(x, y, 'k--')
x = [0, 500]
y = [153, 153]
ax.plot(x, y, 'k--')
ax.xaxis.set_label_position('top') 

##-------------------------Text:START-------------------------------------##
textstr = "AC"
props = dict(boxstyle='round', edgecolor='none', facecolor='none')
ax.text(156, 31.5, s=textstr,  fontdict=font,
        verticalalignment='top', bbox=props)
textstr = "CTB"
props = dict(boxstyle='round', edgecolor='none', facecolor='none')
ax.text(156, 120.5, s=textstr,  fontdict=font,
        verticalalignment='top', bbox=props)
textstr = "Subgrade"
props = dict(boxstyle='round', edgecolor='none', facecolor='none')
ax.text(156, 220.5, s=textstr,  fontdict=font,
        verticalalignment='top', bbox=props)

textstr = "Positive: Tension"
props = dict(boxstyle='round', edgecolor='none', facecolor='none')
ax.text(375, 75, s=textstr,  rotation=90, fontdict=font,
        verticalalignment='top', bbox=props)
##-------------------------Text:END-------------------------------------##
plt.show()


############################################################
####################     EZ     ############################
############################################################
df_EX = pd.read_excel('HeatMap_EZ.xlsx')
df_EX = df_EX * -1
df_EX.set_index(np.arange(0, 30.1, 0.1), drop=True, inplace=True)
ax = sns.heatmap(df_EX, cmap="coolwarm")
plt.xlabel("radial offset, inch", size=10)
plt.ylabel("vertical distance from the top, inch", size=10)
plt.title('Vertical Strain (1e6)')
x = [0, 500]
y = [92.5, 92.5]
ax.plot(x, y, 'r--')
x = [0, 500]
y = [153, 153]
ax.plot(x, y, 'k--')
ax.xaxis.set_label_position('top') 

##-------------------------Text:START-------------------------------------##
textstr = "AC"
props = dict(boxstyle='round', edgecolor='none', facecolor='none')
ax.text(156, 31.5, s=textstr,  fontdict=font,
        verticalalignment='top', bbox=props)
textstr = "CTB"
props = dict(boxstyle='round', edgecolor='none', facecolor='none')
ax.text(156, 120.5, s=textstr,  fontdict=font,
        verticalalignment='top', bbox=props)
textstr = "Subgrade"
props = dict(boxstyle='round', edgecolor='none', facecolor='none')
ax.text(156, 220.5, s=textstr,  fontdict=font,
        verticalalignment='top', bbox=props)
textstr = "Positive: Tension"
props = dict(boxstyle='round', edgecolor='none', facecolor='none')
ax.text(375, 75, s=textstr,  rotation=90, fontdict=font,
        verticalalignment='top', bbox=props)
##-------------------------Text:END-------------------------------------##
plt.show()



